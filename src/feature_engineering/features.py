import pandas as pd
import numpy as np
from typing import Union, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("features")

class FeatureEngineer:
    """
    Engineers custom features for financial fraud detection.
    Supports both batch DataFrame input and single record dictionary input.
    """
    def __init__(self):
        pass
        
    def fit(self, df: pd.DataFrame = None):
        """No fitting needed for simple calculated features, but keeps API standard."""
        return self
        
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features for a batch pandas DataFrame.
        """
        logger.info("Engineering features on DataFrame...")
        df = df.copy()
        
        # 1. Time-based features
        df["hour"] = (df["step"] % 24).astype(np.int8)
        df["day"] = (df["step"] // 24).astype(np.int16)
        df["is_weekend"] = (((df["step"] // 24) % 7) >= 5).astype(np.int8)
        
        # 2. Balance discrepancies (Fraudsters often drain accounts completely or transactions fail to reconcile)
        df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        
        # 3. Destination features
        # In PaySim, merchant accounts start with 'M'
        # Since nameDest is categorical, we check startswith
        # Force conversion to string if it is category
        name_dest_str = df["nameDest"].astype(str)
        df["isMerchantDest"] = name_dest_str.str.startswith("M").astype(np.int8)
        
        # 4. Ratios and flag indicators
        df["amountOrigRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-5)
        df["amountDestRatio"] = df["amount"] / (df["oldbalanceDest"] + 1e-5)
        
        # 5. Empty balances indicators
        df["origZeroBalance"] = (df["oldbalanceOrg"] == 0.0).astype(np.int8)
        df["destZeroBalance"] = (df["oldbalanceDest"] == 0.0).astype(np.int8)
        
        # 6. High Value Transactions
        df["isHighValue"] = (df["amount"] > 200000).astype(np.int8)
        
        # Fill any infinity or nan values in numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(0)
        
        logger.info(f"Engineered features successfully. Columns now: {list(df.columns)}")
        return df

    def transform_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Engineers features for a single dictionary record (FastAPI endpoint input).
        """
        rec = record.copy()
        
        step = int(rec.get("step", 1))
        amount = float(rec.get("amount", 0.0))
        oldbalanceOrg = float(rec.get("oldbalanceOrg", 0.0))
        newbalanceOrig = float(rec.get("newbalanceOrig", 0.0))
        oldbalanceDest = float(rec.get("oldbalanceDest", 0.0))
        newbalanceDest = float(rec.get("newbalanceDest", 0.0))
        nameDest = str(rec.get("nameDest", ""))
        
        # 1. Time-based features
        rec["hour"] = int(step % 24)
        rec["day"] = int(step // 24)
        rec["is_weekend"] = int(((step // 24) % 7) >= 5)
        
        # 2. Balance discrepancies
        rec["errorBalanceOrig"] = float(newbalanceOrig + amount - oldbalanceOrg)
        rec["errorBalanceDest"] = float(oldbalanceDest + amount - newbalanceDest)
        
        # 3. Destination features
        rec["isMerchantDest"] = int(nameDest.startswith("M"))
        
        # 4. Ratios
        rec["amountOrigRatio"] = float(amount / (oldbalanceOrg + 1e-5))
        rec["amountDestRatio"] = float(amount / (oldbalanceDest + 1e-5))
        
        # 5. Empty balances indicators
        rec["origZeroBalance"] = int(oldbalanceOrg == 0.0)
        rec["destZeroBalance"] = int(oldbalanceDest == 0.0)
        
        # 6. High Value
        rec["isHighValue"] = int(amount > 200000)
        
        return rec

if __name__ == "__main__":
    fe = FeatureEngineer()
    test_rec = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C130548614",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    print(fe.transform_dict(test_rec))
