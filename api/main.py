"""
api/main.py
-----------
Financial Fraud Detection FastAPI application.

Changes from original:
- Uses lifespan context manager (replaces deprecated @on_event)
- All artifact loading delegated to api.model_loader.ModelLoader
- Paths resolved via pathlib relative to project root (Render/Docker safe)
- transaction.dict() → transaction.model_dump()  (Pydantic V2)
- /health endpoint returns full diagnostic payload
- API starts gracefully even if model is unavailable
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import ModelLoader, CONFIG_PATH
from api.schemas import (
    TransactionInput, PredictResponse, PredictProbaResponse,
    ModelInfoResponse, LocalExplanationResponse, BatchPredictResponse,
    BatchPredictionItem, FeatureImpact,
)
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.features import FeatureEngineer
from src.explainability.explain import FraudExplainer
from src.utils.logger import get_logger

logger = get_logger("api_main")

# ── Config loading (pathlib-safe) ────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    path = CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# ── Singleton artifacts (populated by lifespan) ───────────────────────────────
_loader = ModelLoader()
fe = FeatureEngineer()
_preprocessor = DataPreprocessor()
_explainer = FraudExplainer()


# ── Lifespan (replaces deprecated @on_event("startup")) ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts at startup; log full diagnostics."""
    logger.info("=" * 60)
    logger.info("  Financial Fraud Detection API — startup")
    logger.info("=" * 60)

    _loader.load_all()

    # Sync the preprocessor wrapper used by SHAP/batch routes
    if _loader.pipeline is not None:
        _preprocessor.pipeline = _loader.pipeline
        _preprocessor.feature_names_out_ = _loader.feature_names

    # Print every diagnostic line so Render logs are searchable
    for line in _loader.diagnostics:
        if line.startswith("ERROR"):
            logger.error(line)
        else:
            logger.info(line)

    if _loader.is_healthy:
        logger.info("✅ All artifacts loaded — API is HEALTHY")
    else:
        logger.warning(
            "⚠️  API started in DEGRADED mode — model or preprocessor missing. "
            "Check /health for details."
        )

    yield  # ── application runs ──

    logger.info("API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description=(
        "Enterprise REST API for real-time Financial Fraud Detection "
        "and Explainable AI predictions."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_model():
    if _loader.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded.",
                "reason": _loader.load_errors,
                "hint": "Check /health for full diagnostics.",
            },
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Return API metadata and model load status."""
    return {
        "api_name": config["api"]["title"],
        "version": config["api"]["version"],
        "status": "online",
        "model_loaded": _loader.model is not None,
        "model_name": _loader.model_name,
        "decision_threshold": _loader.best_threshold,
    }


@app.get("/health")
async def health():
    """
    Full health check including diagnostics.
    Returns 200 always — check 'status' field for 'healthy' / 'degraded'.
    """
    return _loader.get_health_detail()


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return loaded model metadata and validation metrics."""
    _require_model()

    metrics = _loader.metrics.get(
        "metrics_at_opt",
        {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
         "roc_auc": 0.0, "pr_auc": 0.0, "mcc": 0.0},
    )

    return ModelInfoResponse(
        model_name=_loader.model_name,
        imbalance_strategy=config["models"]["imbalance_strategy"],
        best_threshold=_loader.best_threshold,
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(transaction: TransactionInput):
    """Run real-time transaction prediction."""
    _require_model()

    try:
        record = transaction.model_dump()
        engineered = fe.transform_dict(record)
        features = _preprocessor.transform_single_record(engineered)
        prob = float(_loader.model.predict_proba(features)[:, 1][0])
        is_fraud = prob >= _loader.best_threshold

        return PredictResponse(
            is_fraud=is_fraud,
            prediction=int(is_fraud),
            probability=prob,
            decision_threshold=_loader.best_threshold,
            model_version=config["api"]["version"],
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict-proba", response_model=PredictProbaResponse)
async def predict_proba(transaction: TransactionInput):
    """Return raw fraud probability for a single transaction."""
    _require_model()

    try:
        record = transaction.model_dump()
        engineered = fe.transform_dict(record)
        features = _preprocessor.transform_single_record(engineered)
        prob = float(_loader.model.predict_proba(features)[:, 1][0])
        is_fraud = prob >= _loader.best_threshold

        return PredictProbaResponse(
            probability=prob,
            decision_threshold=_loader.best_threshold,
            is_fraud=is_fraud,
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/feature-importance")
async def feature_importance():
    """Return model-specific feature importances."""
    _require_model()

    try:
        if hasattr(_loader.model, "feature_importances_"):
            importances = _loader.model.feature_importances_.tolist()
            mapped = dict(zip(_loader.feature_names, importances))
            sorted_imp = dict(
                sorted(mapped.items(), key=lambda x: x[1], reverse=True)
            )
            return {"importances": sorted_imp}
        return {"detail": "Feature importances not supported by this model class."}
    except Exception as exc:
        logger.error(f"Feature importance error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/shap", response_model=LocalExplanationResponse)
async def explain_shap(transaction: TransactionInput):
    """Return local SHAP explainability for a single transaction."""
    _require_model()

    try:
        _explainer.load_resources()
        record = transaction.model_dump()
        engineered = fe.transform_dict(record)
        features = _preprocessor.transform_single_record(engineered)

        prob = float(_loader.model.predict_proba(features)[:, 1][0])
        shap_details = _explainer.get_local_explanation(features, _loader.feature_names)

        impacts = [
            FeatureImpact(
                feature=imp["feature"],
                shap_value=imp["shap_value"],
                transformed_value=imp["transformed_value"],
            )
            for imp in shap_details["impacts"]
        ]

        return LocalExplanationResponse(
            base_value=shap_details["base_value"],
            prediction_probability=prob,
            impacts=impacts,
        )
    except Exception as exc:
        logger.error(f"SHAP explanation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"SHAP failed: {exc}")


@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Run batch predictions from an uploaded CSV file."""
    _require_model()

    try:
        contents = await file.read()
        df_upload = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        logger.error(f"CSV read failed: {exc}")
        raise HTTPException(status_code=400, detail="Invalid CSV file.")

    try:
        required = [
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        ]
        missing = [c for c in required if c not in df_upload.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing CSV columns: {missing}",
            )

        engineered_df = fe.transform_df(df_upload)
        features = _preprocessor.transform(engineered_df)
        probs = _loader.model.predict_proba(features)[:, 1]
        preds = (probs >= _loader.best_threshold).astype(int)

        predictions_list: List[BatchPredictionItem] = []
        total_fraud = 0

        for idx, (prob, pred) in enumerate(zip(probs, preds)):
            is_f = int(pred) == 1
            if is_f:
                total_fraud += 1
            predictions_list.append(
                BatchPredictionItem(
                    index=idx,
                    is_fraud=is_f,
                    prediction=int(pred),
                    probability=float(prob),
                )
            )

        return BatchPredictResponse(
            total_records=len(df_upload),
            total_fraud=total_fraud,
            predictions=predictions_list,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")
