"""
api/model_loader.py
-------------------
Centralised artifact loading module for the Financial Fraud Detection API.

All paths are resolved using pathlib relative to the PROJECT ROOT, which is
derived from __file__ (this file lives at <project_root>/api/model_loader.py).
This guarantees correct resolution on Windows, Linux (Render), and Docker
regardless of the current working directory at startup.
"""

import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project root: two levels up from this file (api/model_loader.py → api/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Canonical artifact paths (never hard-code strings again)
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts" / "models"
EVALUATION_DIR  = PROJECT_ROOT / "artifacts" / "evaluation"
CONFIG_PATH     = PROJECT_ROOT / "config" / "config.yaml"
MODEL_PATH      = ARTIFACTS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METRICS_PATH    = EVALUATION_DIR / "evaluation_metrics.json"


class ArtifactLoadResult:
    """Holds the result of a single artifact load attempt."""

    def __init__(self, name: str):
        self.name = name
        self.success: bool = False
        self.artifact: Optional[Any] = None
        self.error: Optional[str] = None
        self.path: Optional[str] = None

    def ok(self, artifact: Any, path: Path) -> "ArtifactLoadResult":
        self.success = True
        self.artifact = artifact
        self.path = str(path)
        return self

    def fail(self, error: str, path: Path) -> "ArtifactLoadResult":
        self.success = False
        self.error = error
        self.path = str(path)
        return self


class ModelLoader:
    """
    Loads and exposes all ML artifacts needed by the FastAPI application.

    Usage
    -----
    loader = ModelLoader()
    loader.load_all()

    # Access loaded artifacts
    loader.model          → trained classifier (or None)
    loader.pipeline       → sklearn ColumnTransformer (or None)
    loader.feature_names  → list[str]
    loader.best_threshold → float
    loader.metrics        → dict

    # Access diagnostics
    loader.diagnostics    → list[str]   (human-readable status lines)
    loader.is_healthy     → bool        (True only when model + preprocessor loaded)
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.model_name: str = "Not Loaded"
        self.pipeline: Optional[Any] = None
        self.feature_names: List[str] = []
        self.best_threshold: float = 0.5
        self.metrics: Dict[str, Any] = {}
        self.diagnostics: List[str] = []
        self.load_errors: List[str] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_all(self) -> "ModelLoader":
        """Load every artifact and collect diagnostics. Never raises."""
        self.diagnostics.clear()
        self.load_errors.clear()

        self._log(f"Project root  : {PROJECT_ROOT}")
        self._log(f"Artifacts dir : {ARTIFACTS_DIR}")

        self._load_preprocessor()
        self._load_model()
        self._load_metrics()

        status = "✅ healthy" if self.is_healthy else "⚠️  degraded"
        self._log(f"Overall status: {status}")
        return self

    @property
    def is_healthy(self) -> bool:
        """True only when both model and preprocessor are available."""
        return self.model is not None and self.pipeline is not None

    def get_health_detail(self) -> Dict[str, Any]:
        """Return structured health information for /health endpoint."""
        return {
            "status": "healthy" if self.is_healthy else "degraded",
            "model_loaded": self.model is not None,
            "preprocessor_loaded": self.pipeline is not None,
            "model_name": self.model_name,
            "decision_threshold": self.best_threshold,
            "diagnostics": self.diagnostics,
            "errors": self.load_errors,
        }

    # ------------------------------------------------------------------ #
    #  Private loaders                                                     #
    # ------------------------------------------------------------------ #

    def _load_preprocessor(self) -> None:
        """Load sklearn pipeline + feature names from disk."""
        path = PREPROCESSOR_PATH
        self._log(f"Loading preprocessor from: {path}")

        if not path.exists():
            msg = f"MISSING: preprocessor not found at {path}"
            self._err(msg)
            return

        try:
            payload = joblib.load(path)
            self.pipeline = payload["pipeline"]
            self.feature_names = payload.get("feature_names_out", [])
            self._log(
                f"Preprocessor loaded ✅ — {len(self.feature_names)} output features"
            )
        except Exception as exc:
            self._err(f"Preprocessor load failed: {exc}")

    def _load_model(self) -> None:
        """Load best_model.joblib from disk."""
        path = MODEL_PATH
        self._log(f"Loading model from      : {path}")

        if not path.exists():
            msg = f"MISSING: model not found at {path}"
            self._err(msg)
            return

        try:
            self.model = joblib.load(path)
            self.model_name = type(self.model).__name__
            self._log(f"Model loaded ✅ — {self.model_name}")
        except Exception as exc:
            self._err(f"Model load failed: {exc}")

    def _load_metrics(self) -> None:
        """Load evaluation metrics + optimal threshold."""
        path = METRICS_PATH
        self._log(f"Loading metrics from    : {path}")

        if not path.exists():
            self._log(f"INFO: metrics file not found at {path} — using defaults")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
            self.best_threshold = self.metrics.get("best_threshold", 0.5)
            self._log(
                f"Metrics loaded ✅ — optimal threshold: {self.best_threshold:.4f}"
            )
        except Exception as exc:
            self._err(f"Metrics load failed: {exc}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        self.diagnostics.append(msg)

    def _err(self, msg: str) -> None:
        self.diagnostics.append(f"ERROR: {msg}")
        self.load_errors.append(msg)
