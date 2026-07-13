from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional

class TransactionInput(BaseModel):
    step: int = Field(..., description="Hour of the simulation transaction (1-744)", ge=0)
    type: str = Field(..., description="Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)")
    amount: float = Field(..., description="Transaction amount in local currency", gt=0)
    nameOrig: str = Field(..., description="ID of customer initiating transaction")
    oldbalanceOrg: float = Field(..., description="Original balance before transaction", ge=0)
    newbalanceOrig: float = Field(..., description="New balance after transaction", ge=0)
    nameDest: str = Field(..., description="ID of transaction recipient")
    oldbalanceDest: float = Field(..., description="Recipient balance before transaction", ge=0)
    newbalanceDest: float = Field(..., description="Recipient balance after transaction", ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C130548614",
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }
    )

class PredictResponse(BaseModel):
    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    prediction: int = Field(..., description="Binary prediction: 1 for fraud, 0 for legitimate")
    probability: float = Field(..., description="Model probability of transaction being fraud (0.0 to 1.0)")
    decision_threshold: float = Field(..., description="Decision boundary used for classification")
    model_version: str = Field(..., description="Active version of the trained model")

class PredictProbaResponse(BaseModel):
    probability: float = Field(..., description="Model probability of transaction being fraud (0.0 to 1.0)")
    decision_threshold: float = Field(..., description="Decision boundary used for classification")
    is_fraud: bool = Field(..., description="Whether probability exceeds decision threshold")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Active classifier algorithm name")
    imbalance_strategy: str = Field(..., description="Resampling technique applied during training")
    best_threshold: float = Field(..., description="Optimal decision threshold to maximize F1-score")
    metrics: Dict[str, float] = Field(..., description="Validation performance metrics (Accuracy, F1, Recall, etc.)")

class FeatureImpact(BaseModel):
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value contribution")
    transformed_value: float = Field(..., description="Transformed input value used by the model")

class LocalExplanationResponse(BaseModel):
    base_value: float = Field(..., description="SHAP base value (average model log-odds/probability output)")
    prediction_probability: float = Field(..., description="Final model prediction probability")
    impacts: List[FeatureImpact] = Field(..., description="List of feature impact contributions sorted by absolute SHAP value")

class BatchPredictionItem(BaseModel):
    index: int = Field(..., description="Original row index in the CSV file")
    is_fraud: bool = Field(..., description="Whether the transaction is predicted as fraud")
    prediction: int = Field(..., description="Binary prediction (1/0)")
    probability: float = Field(..., description="Fraud probability (0.0 to 1.0)")

class BatchPredictResponse(BaseModel):
    total_records: int = Field(..., description="Total number of evaluated records")
    total_fraud: int = Field(..., description="Total fraudulent transactions identified")
    predictions: List[BatchPredictionItem] = Field(..., description="Prediction results list")
