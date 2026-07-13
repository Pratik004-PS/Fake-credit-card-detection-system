import os
import shutil
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("data_loader")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = self.config["data"]["raw_dir"]
        self.dataset_name = self.config["data"]["dataset_name"]
        self.target_col = self.config["features"]["target"]
        
    def find_and_setup_dataset(self) -> str:
        """
        Locates the transaction CSV file in the workspace or data directories.
        Moves it to data/raw if found elsewhere. Returns the final path.
        """
        target_path = os.path.join(self.raw_dir, self.dataset_name)
        if os.path.exists(target_path):
            logger.info(f"Dataset found at target location: {target_path}")
            return target_path
            
        # Search at workspace root
        if os.path.exists(self.dataset_name):
            os.makedirs(self.raw_dir, exist_ok=True)
            shutil.move(self.dataset_name, target_path)
            logger.info(f"Dataset found at root and moved to: {target_path}")
            return target_path
            
        # Search in workspace recursively for any csv file > 100MB as a fallback
        logger.warning(f"Dataset {self.dataset_name} not found. Searching workspace for CSV files...")
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    if os.path.getsize(file_path) > 100 * 1024 * 1024: # > 100 MB
                        os.makedirs(self.raw_dir, exist_ok=True)
                        shutil.move(file_path, target_path)
                        logger.info(f"Found large CSV {file} in {root} and moved to: {target_path}")
                        return target_path
                        
        raise FileNotFoundError("Transaction dataset (CSV) could not be located in workspace.")

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcast float/integer types and convert object columns to categories
        to reduce memory consumption significantly (essential for PaySim).
        """
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                # Convert string type col to category if unique values count is small
                if col_type == object and df[col].nunique() < 50:
                    df[col] = df[col].astype('category')
                    
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB ({((start_mem - end_mem) / start_mem) * 100:.1f}% reduction)")
        return df

    def get_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a detailed summary report of the dataframe.
        """
        missing_vals = df.isnull().sum().to_dict()
        dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        
        report = {
            "dimensions": df.shape,
            "columns": list(df.columns),
            "missing_values": missing_vals,
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": float(df.memory_usage().sum() / 1024**2),
            "data_types": dtypes
        }
        
        if self.target_col in df.columns:
            class_counts = df[self.target_col].value_counts().to_dict()
            class_ratios = df[self.target_col].value_counts(normalize=True).to_dict()
            report["target_distribution"] = {
                "counts": {str(k): int(v) for k, v in class_counts.items()},
                "ratios": {str(k): float(v) for k, v in class_ratios.items()}
            }
            
        logger.info(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        if "target_distribution" in report:
            logger.info(f"Target Distribution (Class Imbalance): {report['target_distribution']['ratios']}")
            
        return report

    def load_data(self, use_full: Optional[bool] = None) -> pd.DataFrame:
        """
        Loads the dataset, optimizes types, validates the schema and
        optionally downsamples keeping stratification if configured.
        """
        file_path = self.find_and_setup_dataset()
        logger.info(f"Loading data from {file_path}...")
        
        # Load data in chunks to prevent memory spike or read all if fit in memory
        df = pd.read_csv(file_path)
        
        # Optimize types
        df = self.optimize_memory(df)
        
        # Validation
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
            
        # Report statistics
        self.get_summary_report(df)
        
        # Stratified sampling if requested
        should_use_full = use_full if use_full is not None else self.config["data"]["use_full_dataset"]
        sample_size = self.config["data"]["sample_size"]
        
        if not should_use_full and sample_size < len(df):
            logger.info(f"Performing stratified sampling to obtain {sample_size} rows...")
            # Perform stratified sample on target
            grouped = df.groupby(self.target_col, group_keys=False)
            frac = sample_size / len(df)
            df = grouped.apply(lambda x: x.sample(frac=frac, random_state=self.config["data"]["random_state"]))
            # Adjust to exactly target sample size if slight mismatch
            if len(df) != sample_size:
                df = df.sample(n=sample_size, random_state=self.config["data"]["random_state"])
            logger.info(f"Sampled dataset dimensions: {df.shape}")
            self.get_summary_report(df)
            
        return df

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()
