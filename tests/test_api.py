"""
tests/test_api.py
-----------------
Unit tests for the Financial Fraud Detection FastAPI application.
Mocks the ModelLoader so tests run without trained model files.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch ModelLoader.load_all before importing app so lifespan doesn't hit disk
with patch("api.model_loader.ModelLoader.load_all"):
    from api.main import app, _loader, _preprocessor

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "online"


def test_health_endpoint():
    """When model is not loaded, health should report degraded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Model was not loaded (mocked), so expect degraded
    assert data["status"] == "degraded"


def test_predict_endpoint_success():
    """Mock the loader + preprocessor to simulate a successful prediction."""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    # Temporarily inject model into loader
    original_model = _loader.model
    original_threshold = _loader.best_threshold
    original_pipeline = _preprocessor.pipeline

    try:
        _loader.model = mock_model
        _loader.model_name = "MockClassifier"
        _loader.best_threshold = 0.5

        # Mock preprocessor transform
        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = np.zeros((1, 10))
        _preprocessor.pipeline = mock_pipeline

        payload = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "nameOrig": "C12345",
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "nameDest": "C67890",
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 2000.0,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["prediction"] == 1
        assert data["probability"] == 0.9
    finally:
        # Restore originals
        _loader.model = original_model
        _loader.best_threshold = original_threshold
        _preprocessor.pipeline = original_pipeline


def test_predict_endpoint_validation_error():
    """Missing required 'amount' field should return 422."""
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "nameOrig": "C12345",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "nameDest": "C67890",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 2000.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
