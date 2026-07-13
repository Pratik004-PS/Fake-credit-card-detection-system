import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# We import the app from api.main
# To run this test without having models trained, we will mock the model and preprocessor states
with patch('joblib.load') as mock_joblib, patch('src.preprocessing.preprocessor.DataPreprocessor.load_pipeline') as mock_load_pipeline:
    from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "online"

def test_health_endpoint():
    # If model is not loaded, it should return degraded status
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"

@patch('api.main.model')
@patch('api.main.preprocessor')
def test_predict_endpoint_success(mock_prep, mock_model):
    # Mock preprocessor output
    mock_prep.transform_single_record.return_value = np.zeros((1, 10))
    
    # Mock model probability output: [ [prob_legit, prob_fraud] ]
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    
    # Mock active threshold
    import api.main
    api.main.model = mock_model
    api.main.best_threshold = 0.5
    
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "nameDest": "C67890",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 2000.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["prediction"] == 1
    assert data["probability"] == 0.9

def test_predict_endpoint_validation_error():
    # Missing required amount parameter
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "nameOrig": "C12345",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "nameDest": "C67890",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 2000.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error code
