import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.preprocessor import DataPreprocessor

@pytest.fixture
def mock_transaction_df():
    return pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [181.0, 500.0],
        "nameOrig": ["C130548614", "C214589632"],
        "oldbalanceOrg": [181.0, 1000.0],
        "newbalanceOrig": [0.0, 500.0],
        "nameDest": ["C553264065", "M852146932"],
        "oldbalanceDest": [0.0, 0.0],
        "newbalanceDest": [0.0, 0.0]
    })

def test_feature_engineer_df(mock_transaction_df):
    fe = FeatureEngineer()
    df_transformed = fe.transform_df(mock_transaction_df)
    
    # Assert features are created
    assert "hour" in df_transformed.columns
    assert "day" in df_transformed.columns
    assert "is_weekend" in df_transformed.columns
    assert "errorBalanceOrig" in df_transformed.columns
    assert "errorBalanceDest" in df_transformed.columns
    assert "isMerchantDest" in df_transformed.columns
    assert "amountOrigRatio" in df_transformed.columns
    assert "isHighValue" in df_transformed.columns
    
    # Verify values
    assert df_transformed.loc[0, "hour"] == 1
    assert df_transformed.loc[0, "isMerchantDest"] == 0 # nameDest is C553264065 (does not start with M)
    assert df_transformed.loc[1, "isMerchantDest"] == 1 # nameDest is M852146932 (starts with M)
    
    # Verify balance error
    # row 0: newbalanceOrig(0.0) + amount(181.0) - oldbalanceOrg(181.0) = 0.0
    assert df_transformed.loc[0, "errorBalanceOrig"] == 0.0

def test_feature_engineer_dict():
    fe = FeatureEngineer()
    record = {
        "step": 25,
        "type": "PAYMENT",
        "amount": 250000.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 250000.0,
        "nameDest": "M99999",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    
    rec_transformed = fe.transform_dict(record)
    
    assert rec_transformed["hour"] == 1 # 25 % 24
    assert rec_transformed["day"] == 1  # 25 // 24
    assert rec_transformed["isMerchantDest"] == 1 # nameDest starts with M
    assert rec_transformed["isHighValue"] == 1     # amount > 200,000

def test_preprocessor_pipeline():
    # Generate a larger dataset (10 rows: 5 class 0, 5 class 1) to satisfy StratifiedShuffleSplit requirements
    mock_large_df = pd.DataFrame({
        "step": [1] * 10,
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"] * 2,
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0] * 2,
        "nameOrig": [f"C_orig_{i}" for i in range(10)],
        "oldbalanceOrg": [1000.0] * 10,
        "newbalanceOrig": [900.0] * 10,
        "nameDest": [f"C_dest_{i}" for i in range(10)],
        "oldbalanceDest": [100.0] * 10,
        "newbalanceDest": [200.0] * 10,
        "isFraud": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
    
    fe = FeatureEngineer()
    df_engineered = fe.transform_df(mock_large_df)
    
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.split_data(df_engineered)
    
    X_trans = prep.fit_transform(X_train)
    
    # Shape checks (features count should match output features)
    assert X_trans.shape[0] == 8 # 10 rows * 0.8 train split
    assert X_trans.shape[1] == len(prep.feature_names_out_)
    assert "amount" in prep.feature_names_out_

