import os
import yaml
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("explain")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class FraudExplainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.artifact_dir = self.config["models"]["artifact_dir"]
        self.evaluation_dir = self.config["models"]["evaluation_dir"]
        self.saved_model_path = os.path.join(self.artifact_dir, self.config["models"]["saved_model"])
        
        self.model = None
        self.explainer = None
        
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def load_resources(self):
        """Loads final model and initializes TreeExplainer."""
        if self.model is None:
            if not os.path.exists(self.saved_model_path):
                raise FileNotFoundError(f"Saved model not found at {self.saved_model_path}")
            self.model = joblib.load(self.saved_model_path)
            
            logger.info("Initializing SHAP TreeExplainer...")
            # XGBoost, LightGBM, CatBoost work directly with TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self, X_transformed: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for given preprocessed features."""
        self.load_resources()
        return self.explainer.shap_values(X_transformed)

    def save_global_importance_plots(self, X_transformed: np.ndarray, feature_names: List[str]):
        """
        Generate and save global SHAP summary plots.
        """
        self.load_resources()
        logger.info("Generating global SHAP summary plots...")
        
        # Check SHAP value output shape
        # For classification models, SHAP values can be a list [class0_shap, class1_shap] or single class1 array
        shap_vals = self.get_shap_values(X_transformed)
        
        # If shape is list (e.g. from CatBoost/RF or older versions), extract class 1 (Fraud)
        if isinstance(shap_vals, list):
            # For binary classification, index 1 corresponds to positive class
            shap_vals_class1 = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            # For XGBoost/LightGBM, shap_values is often already class 1 (or 2D where last dim is 2)
            if len(shap_vals.shape) == 3: # (samples, features, classes)
                shap_vals_class1 = shap_vals[:, :, 1]
            else:
                shap_vals_class1 = shap_vals
                
        # 1. Bar Plot (Mean Absolute SHAP Value)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_class1, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)")
        bar_path = os.path.join(self.evaluation_dir, "shap_summary_bar.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary bar plot saved to {bar_path}")
        
        # 2. Beeswarm Plot (Feature impact distribution)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_class1, X_transformed, feature_names=feature_names, show=False)
        plt.title("SHAP Beeswarm Plot")
        beeswarm_path = os.path.join(self.evaluation_dir, "shap_summary_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary beeswarm plot saved to {beeswarm_path}")

    def get_local_explanation(self, x_transformed: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Compute SHAP values for a single prediction and return feature impact mappings.
        """
        self.load_resources()
        
        # Shape formatting: x_transformed should be 2D (1, n_features)
        if len(x_transformed.shape) == 1:
            x_transformed = x_transformed.reshape(1, -1)
            
        shap_vals = self.explainer.shap_values(x_transformed)
        
        if isinstance(shap_vals, list):
            shap_vals_class1 = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            if len(shap_vals.shape) == 3:
                shap_vals_class1 = shap_vals[0, :, 1]
            else:
                shap_vals_class1 = shap_vals[0]
                
        # Match features with shap value
        impacts = []
        for name, val, raw_val in zip(feature_names, shap_vals_class1, x_transformed[0]):
            impacts.append({
                "feature": name,
                "shap_value": float(val),
                "transformed_value": float(raw_val)
            })
            
        # Sort by absolute SHAP value impact
        impacts = sorted(impacts, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        base_value = float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value)
        
        return {
            "base_value": base_value,
            "impacts": impacts
        }

    def save_local_waterfall_plot(self, x_transformed: np.ndarray, feature_names: List[str], filename: str = "shap_waterfall.png") -> str:
        """
        Generates and saves a local waterfall plot for a single instance.
        """
        self.load_resources()
        if len(x_transformed.shape) == 1:
            x_transformed = x_transformed.reshape(1, -1)
            
        # Create an Explanation object required for the waterfall plot in newer SHAP versions
        shap_vals = self.explainer(x_transformed)
        
        # Format based on dimension
        # If binary classification output is 3D (samples, features, classes)
        if len(shap_vals.shape) == 3:
            exp = shap.Explanation(
                values=shap_vals.values[0, :, 1],
                base_values=shap_vals.base_values[0, 1],
                data=x_transformed[0],
                feature_names=feature_names
            )
        else:
            # If shape is (samples, features)
            exp = shap.Explanation(
                values=shap_vals.values[0],
                base_values=shap_vals.base_values[0] if isinstance(shap_vals.base_values, np.ndarray) else shap_vals.base_values,
                data=x_transformed[0],
                feature_names=feature_names
            )
            
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.title("SHAP Local Explanation (Waterfall Plot)", pad=20)
        
        path = os.path.join(self.evaluation_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Local SHAP waterfall plot saved to {path}")
        return path

if __name__ == "__main__":
    explainer = FraudExplainer()
    print("Explainer initialized.")
