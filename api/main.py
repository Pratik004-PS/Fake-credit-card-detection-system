import os
import io
import yaml
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from api.schemas import (
    TransactionInput, PredictResponse, PredictProbaResponse,
    ModelInfoResponse, LocalExplanationResponse, BatchPredictResponse,
    BatchPredictionItem, FeatureImpact
)
from src.preprocessing.preprocessor import DataPreprocessor
from src.feature_engineering.features import FeatureEngineer
from src.explainability.explain import FraudExplainer
from src.utils.logger import get_logger

logger = get_logger("api_main")

# Load configuration
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Instantiate global tools
fe = FeatureEngineer()
preprocessor = DataPreprocessor()
explainer = FraudExplainer()

app = FastAPI(
    title=config["api"]["title"],
    version=config["api"]["version"],
    description="Enterprise REST API for real-time Financial Fraud Detection and Explainable AI predictions."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
model = None
model_metrics = {}
best_threshold = 0.5
model_name = "Not Loaded"
feature_names = []

@app.on_event("startup")
async def startup_event():
    """Load model, preprocessing pipeline, and metrics on API startup."""
    global model, model_metrics, best_threshold, model_name, feature_names
    
    logger.info("Initializing API application resources...")
    
    # 1. Load Preprocessor
    try:
        preprocessor.load_pipeline()
        feature_names = preprocessor.feature_names_out_
        logger.info("Preprocessor pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        
    # 2. Load Model
    model_path = os.path.join(config["models"]["artifact_dir"], config["models"]["saved_model"])
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_name = type(model).__name__
            logger.info(f"Model '{model_name}' loaded successfully.")
        else:
            logger.warning(f"No trained model found at {model_path}. Please run training first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        
    # 3. Load Evaluation Metrics
    metrics_path = os.path.join(config["models"]["evaluation_dir"], "evaluation_metrics.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                model_metrics = json.load(f)
            best_threshold = model_metrics.get("best_threshold", 0.5)
            logger.info(f"Trained metrics loaded. Optimal threshold: {best_threshold:.4f}")
        else:
            logger.warning("No metrics file found. Using default threshold 0.5.")
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")

@app.get("/")
async def root():
    """Return API metadata."""
    return {
        "api_name": config["api"]["title"],
        "version": config["api"]["version"],
        "status": "online",
        "model_loaded": model is not None,
        "model_name": model_name,
        "decision_threshold": best_threshold
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None or preprocessor.pipeline is None:
        return {
            "status": "degraded",
            "reason": "Model or preprocessor pipeline not loaded yet"
        }
    return {"status": "healthy"}

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return loaded model metadata and metrics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please train it first.")
        
    metrics = model_metrics.get("metrics_at_opt", {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "roc_auc": 0.0, "pr_auc": 0.0, "mcc": 0.0
    })
    
    return ModelInfoResponse(
        model_name=model_name,
        imbalance_strategy=config["models"]["imbalance_strategy"],
        best_threshold=best_threshold,
        metrics=metrics
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(transaction: TransactionInput):
    """Run real-time transaction prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert Pydantic input to dict
        record = transaction.dict()
        
        # 1. Apply Feature Engineering
        engineered_rec = fe.transform_dict(record)
        
        # 2. Run Preprocessor
        transformed_features = preprocessor.transform_single_record(engineered_rec)
        
        # 3. Predict Probability
        prob = float(model.predict_proba(transformed_features)[:, 1][0])
        
        # 4. Check Decision Threshold
        prediction = 1 if prob >= best_threshold else 0
        is_fraud = prediction == 1
        
        return PredictResponse(
            is_fraud=is_fraud,
            prediction=prediction,
            probability=prob,
            decision_threshold=best_threshold,
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-proba", response_model=PredictProbaResponse)
async def predict_proba(transaction: TransactionInput):
    """Run real-time transaction probability check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        record = transaction.dict()
        engineered_rec = fe.transform_dict(record)
        transformed_features = preprocessor.transform_single_record(engineered_rec)
        prob = float(model.predict_proba(transformed_features)[:, 1][0])
        is_fraud = prob >= best_threshold
        
        return PredictProbaResponse(
            probability=prob,
            decision_threshold=best_threshold,
            is_fraud=is_fraud
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
async def feature_importance():
    """Return model-specific feature importances if available."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            mapped_importances = dict(zip(feature_names, importances))
            # Sort importances
            sorted_importances = dict(sorted(mapped_importances.items(), key=lambda x: x[1], reverse=True))
            return {"importances": sorted_importances}
        else:
            return {"detail": "Feature importances are not supported by this model class."}
    except Exception as e:
        logger.error(f"Feature importance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shap", response_model=LocalExplanationResponse)
async def explain_shap(transaction: TransactionInput):
    """Return local SHAP explainability mapping for the transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Load explainer resources
        explainer.load_resources()
        
        record = transaction.dict()
        engineered_rec = fe.transform_dict(record)
        transformed_features = preprocessor.transform_single_record(engineered_rec)
        
        # Calculate prob
        prob = float(model.predict_proba(transformed_features)[:, 1][0])
        
        # Compute local SHAP impacts
        shap_details = explainer.get_local_explanation(transformed_features, feature_names)
        
        impacts = [
            FeatureImpact(
                feature=imp["feature"],
                shap_value=imp["shap_value"],
                transformed_value=imp["transformed_value"]
            )
            for imp in shap_details["impacts"]
        ]
        
        return LocalExplanationResponse(
            base_value=shap_details["base_value"],
            prediction_probability=prob,
            impacts=impacts
        )
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile = File(...)):
    """Run batch predictions from an uploaded CSV file."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Read uploaded csv file
    try:
        contents = await file.read()
        df_upload = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"CSV read failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV file. Please upload a valid CSV.")
        
    try:
        # Validate schema
        required_raw_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
        missing_cols = [col for col in required_raw_cols if col not in df_upload.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns in uploaded CSV: {missing_cols}")
            
        # 1. Feature Engineering
        engineered_df = fe.transform_df(df_upload)
        
        # 2. Preprocess
        transformed_features = preprocessor.transform(engineered_df)
        
        # 3. Probabilities and Predictions
        probs = model.predict_proba(transformed_features)[:, 1]
        preds = (probs >= best_threshold).astype(int)
        
        # Format response
        predictions_list = []
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
                    probability=float(prob)
                )
            )
            
        return BatchPredictResponse(
            total_records=len(df_upload),
            total_fraud=total_fraud,
            predictions=predictions_list
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
