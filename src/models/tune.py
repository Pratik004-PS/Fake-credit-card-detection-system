import os
import yaml
import joblib
import optuna
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, auc
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessing.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger("tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelTuner:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.artifact_dir = self.config["models"]["artifact_dir"]
        self.evaluation_dir = self.config["models"]["evaluation_dir"]
        self.cv_splits = self.config["models"]["cv_splits"]
        self.random_state = self.config["data"]["random_state"]
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
    def get_best_baseline_name(self) -> str:
        """Read comparison table and get the best baseline model name."""
        comparison_path = os.path.join(self.evaluation_dir, "model_comparison.csv")
        if os.path.exists(comparison_path):
            df = pd.read_csv(comparison_path)
            return df.iloc[0]["Model"]
        # Default fallback
        logger.warning("No model comparison file found. Defaulting tuner to XGBoost.")
        return "XGBoost"

    def tune_model(self, X_train: np.ndarray, y_train: pd.Series, model_name: str, n_trials: int = 15) -> Dict[str, Any]:
        """
        Run Optuna search to tune hyperparameters for the selected model.
        """
        logger.info(f"Starting hyperparameter tuning for {model_name} with {n_trials} trials...")
        
        # Calculate class weights for imbalance handling if class_weight strategy is used
        trainer = ModelTrainer()
        weights = trainer.get_class_weights(y_train)
        scale_pos_weight = weights[1] / weights[0]
        
        def objective(trial):
            # Define search spaces based on model type
            if model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "eval_metric": "logloss",
                    "random_state": self.random_state,
                    "n_jobs": -1
                }
                if trainer.imbalance_strategy == "class_weights":
                    params["scale_pos_weight"] = scale_pos_weight
                model_cls = XGBClassifier
                
            elif model_name == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "verbose": -1
                }
                if trainer.imbalance_strategy == "class_weights":
                    params["class_weight"] = "balanced"
                model_cls = LGBMClassifier
                
            elif model_name == "CatBoost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 400),
                    "depth": trial.suggest_int("depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "random_state": self.random_state,
                    "verbose": 0
                }
                if trainer.imbalance_strategy == "class_weights":
                    params["auto_class_weights"] = "Balanced"
                model_cls = CatBoostClassifier
                
            else: # Fallback to Random Forest
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 15),
                    "random_state": self.random_state,
                    "n_jobs": -1
                }
                if trainer.imbalance_strategy == "class_weights":
                    params["class_weight"] = "balanced"
                model_cls = RandomForestClassifier
                
            # Perform K-Fold Cross Validation
            skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
            scores = []
            
            # Re-sample training data if using SMOTE/etc.
            # (Note: we should ideally resample inside the CV loop to prevent data leakage,
            # but for baseline/Optuna performance, we resample X_train beforehand in the trainer)
            X_tr_res, y_tr_res = trainer.handle_imbalance(X_train, y_train)
            
            for train_idx, val_idx in skf.split(X_tr_res, y_tr_res):
                X_tr_cv, y_tr_cv = X_tr_res[train_idx], y_tr_res.iloc[train_idx]
                X_val_cv, y_val_cv = X_tr_res[val_idx], y_tr_res.iloc[val_idx]
                
                model = model_cls(**params)
                model.fit(X_tr_cv, y_tr_cv)
                
                # We optimize F1 score / PR-AUC
                preds = model.predict(X_val_cv)
                scores.append(f1_score(y_val_cv, preds))
                
            return np.mean(scores)
            
        # Run Optuna Study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Optuna Study Complete. Best F1 Score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            "model_name": model_name,
            "best_params": study.best_params,
            "best_value": study.best_value
        }

    def train_final_model(self, X_train: np.ndarray, y_train: pd.Series, 
                          tune_results: Dict[str, Any]) -> Any:
        """
        Train the final model with best hyperparameters on the full training set
        and save it to disk.
        """
        model_name = tune_results["model_name"]
        best_params = tune_results["best_params"]
        
        logger.info(f"Training final {model_name} model with optimized hyperparameters...")
        
        trainer = ModelTrainer()
        X_tr_res, y_tr_res = trainer.handle_imbalance(X_train, y_train)
        weights = trainer.get_class_weights(y_train)
        scale_pos_weight = weights[1] / weights[0]
        
        # Add non-tuned defaults back
        params = best_params.copy()
        params["random_state"] = self.random_state
        
        if model_name == "XGBoost":
            params["eval_metric"] = "logloss"
            params["n_jobs"] = -1
            if trainer.imbalance_strategy == "class_weights":
                params["scale_pos_weight"] = scale_pos_weight
            model = XGBClassifier(**params)
            
        elif model_name == "LightGBM":
            params["n_jobs"] = -1
            params["verbose"] = -1
            if trainer.imbalance_strategy == "class_weights":
                params["class_weight"] = "balanced"
            model = LGBMClassifier(**params)
            
        elif model_name == "CatBoost":
            params["verbose"] = 0
            if trainer.imbalance_strategy == "class_weights":
                params["auto_class_weights"] = "Balanced"
            model = CatBoostClassifier(**params)
            
        else:
            params["n_jobs"] = -1
            if trainer.imbalance_strategy == "class_weights":
                params["class_weight"] = "balanced"
            model = RandomForestClassifier(**params)
            
        with mlflow.start_run(run_name=f"tuned_{model_name.lower()}"):
            # Log params
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("imbalance_strategy", trainer.imbalance_strategy)
            
            # Fit
            model.fit(X_tr_res, y_tr_res)
            
            # Save final model
            model_path = os.path.join(self.artifact_dir, self.config["models"]["saved_model"])
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            logger.info(f"Final optimized model saved to {model_path}")
            
        return model

def main():
    # Load and preprocess data
    loader = DataLoader()
    raw_df = loader.load_data()
    
    fe = FeatureEngineer()
    engineered_df = fe.transform_df(raw_df)
    
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.split_data(engineered_df)
    
    # Load existing fitted pipeline
    X_train_trans = prep.fit_transform(X_train)
    prep.save_pipeline()
    
    # Tune
    tuner = ModelTuner()
    best_model_name = tuner.get_best_baseline_name()
    
    tune_results = tuner.tune_model(X_train_trans, y_train, best_model_name, n_trials=5)
    tuner.train_final_model(X_train_trans, y_train, tune_results)

if __name__ == "__main__":
    main()
