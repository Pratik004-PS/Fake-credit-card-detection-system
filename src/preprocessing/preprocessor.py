import os
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("preprocessor")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.target_col = self.config["features"]["target"]
        self.drop_cols = self.config["features"]["drop_cols"]
        self.artifact_dir = self.config["models"]["artifact_dir"]
        self.preprocessor_filename = self.config["models"]["saved_preprocessor"]
        
        # Categorical columns
        self.cat_cols = self.config["features"]["categorical"]
        
        # Numeric columns to scale (heavy-tailed distributions require RobustScaler)
        self.num_to_scale = [
            "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "amountOrigRatio", "amountDestRatio"
        ]
        
        # Columns that can pass through as-is
        self.pass_through_cols = [
            "step", "hour", "day", "is_weekend", "isMerchantDest", 
            "origZeroBalance", "destZeroBalance", "isHighValue"
        ]
        
        self.pipeline = None
        self.feature_names_out_ = []

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataframe into stratified train and test features and target.
        """
        logger.info("Splitting dataset into train and test splits...")
        
        X = df.drop(columns=[self.target_col] + [c for c in self.drop_cols if c in df.columns], errors="ignore")
        y = df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y
        )
        
        logger.info(f"Train split size: {X_train.shape[0]} samples (Fraud: {y_train.sum()})")
        logger.info(f"Test split size: {X_test.shape[0]} samples (Fraud: {y_test.sum()})")
        
        return X_train, X_test, y_train, y_test

    def build_pipeline(self) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.
        """
        logger.info("Building ColumnTransformer preprocessing pipeline...")
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_to_scale),
                ('cat', categorical_transformer, self.cat_cols),
                ('pass', 'passthrough', self.pass_through_cols)
            ]
        )
        
        self.pipeline = preprocessor
        return preprocessor

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit pipeline on train features and return transformed numpy array.
        """
        if self.pipeline is None:
            self.build_pipeline()
            
        logger.info("Fitting preprocessing pipeline on training data...")
        X_train_transformed = self.pipeline.fit_transform(X_train)
        
        # Save output feature names
        self.save_feature_names()
        
        return X_train_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the already fitted pipeline.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet. Call fit_transform first.")
            
        return self.pipeline.transform(X)

    def save_feature_names(self):
        """Extract and save feature names from ColumnTransformer."""
        # Get onehot categories names
        cat_encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(cat_encoder.get_feature_names_out(self.cat_cols))
        
        self.feature_names_out_ = self.num_to_scale + cat_features + self.pass_through_cols
        logger.info(f"Output features: {self.feature_names_out_}")

    def save_pipeline(self):
        """Save the fitted preprocessor pipeline to disk."""
        os.makedirs(self.artifact_dir, exist_ok=True)
        path = os.path.join(self.artifact_dir, self.preprocessor_filename)
        
        # We also save the feature names out inside a custom wrapper or dict
        payload = {
            "pipeline": self.pipeline,
            "feature_names_out": self.feature_names_out_
        }
        
        joblib.dump(payload, path)
        logger.info(f"Preprocessing pipeline saved to {path}")

    def load_pipeline(self) -> ColumnTransformer:
        """Load the preprocessor pipeline from disk."""
        path = os.path.join(self.artifact_dir, self.preprocessor_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saved pipeline not found at {path}")
            
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.feature_names_out_ = payload["feature_names_out"]
        logger.info(f"Preprocessing pipeline loaded from {path}")
        return self.pipeline

    def transform_single_record(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Transforms a single transaction record (represented as dictionary)
        into the model input format.
        """
        if self.pipeline is None:
            self.load_pipeline()
            
        # Convert single dict to DataFrame
        df = pd.DataFrame([record])
        
        # Ensure all columns required by the preprocessor are present
        # Fill missing ones with defaults
        required_cols = self.num_to_scale + self.cat_cols + self.pass_through_cols
        for col in required_cols:
            if col not in df.columns:
                if col in self.cat_cols:
                    df[col] = "TRANSFER" # logical default
                else:
                    df[col] = 0.0
                    
        # Apply transformation
        return self.transform(df)

if __name__ == "__main__":
    prep = DataPreprocessor()
    print("Preprocessor ready.")
