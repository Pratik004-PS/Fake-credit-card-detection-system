import os
import yaml
import joblib
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, recall_score, precision_score, roc_auc_score

from src.preprocessing.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger("trainer")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.imbalance_strategy = self.config["models"]["imbalance_strategy"]
        self.artifact_dir = self.config["models"]["artifact_dir"]
        self.evaluation_dir = self.config["models"]["evaluation_dir"]
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
    def handle_imbalance(self, X_train: np.ndarray, y_train: pd.Series) -> Tuple[np.ndarray, pd.Series]:
        """
        Applies class imbalance handling strategies on the training data.
        """
        logger.info(f"Handling class imbalance using strategy: {self.imbalance_strategy}...")
        
        if self.imbalance_strategy == "original":
            logger.info("Keeping original imbalanced training set.")
            return X_train, y_train
            
        elif self.imbalance_strategy == "smote":
            smote = SMOTE(random_state=self.config["data"]["random_state"])
            X_res, y_res = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied: Resampled shape {X_res.shape} (Fraud: {y_res.sum()})")
            return X_res, y_res
            
        elif self.imbalance_strategy == "rus":
            rus = RandomUnderSampler(random_state=self.config["data"]["random_state"])
            X_res, y_res = rus.fit_resample(X_train, y_train)
            logger.info(f"Random Undersampling applied: Resampled shape {X_res.shape} (Fraud: {y_res.sum()})")
            return X_res, y_res
            
        elif self.imbalance_strategy == "smoteenn":
            smoteenn = SMOTEENN(random_state=self.config["data"]["random_state"])
            X_res, y_res = smoteenn.fit_resample(X_train, y_train)
            logger.info(f"SMOTEENN applied: Resampled shape {X_res.shape} (Fraud: {y_res.sum()})")
            return X_res, y_res
            
        elif self.imbalance_strategy == "class_weights":
            logger.info("Class weights strategy selected. Weights will be passed to model constructors.")
            return X_train, y_train
            
        else:
            logger.warning(f"Unknown strategy '{self.imbalance_strategy}'. Defaulting to original dataset.")
            return X_train, y_train

    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights based on target counts."""
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        total = len(y)
        weight_neg = total / (2.0 * neg_count)
        weight_pos = total / (2.0 * pos_count)
        return {0: weight_neg, 1: weight_pos}

    def train_baseline_models(self, X_train: np.ndarray, y_train: pd.Series, 
                              X_test: np.ndarray, y_test: pd.Series,
                              feature_names: list) -> pd.DataFrame:
        """
        Train multiple baseline classifiers and return a comparative DataFrame.
        """
        # Calculate class weights for models that support it
        weights = self.get_class_weights(y_train)
        scale_pos_weight = weights[1] / weights[0]
        
        # Instantiate models
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=self.config["models"]["logistic_regression"]["max_iter"],
                solver=self.config["models"]["logistic_regression"]["solver"],
                class_weight=weights if self.imbalance_strategy == "class_weights" else None,
                random_state=self.config["data"]["random_state"]
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=self.config["models"]["random_forest"]["n_estimators"],
                max_depth=self.config["models"]["random_forest"]["max_depth"],
                class_weight="balanced" if self.imbalance_strategy == "class_weights" else None,
                n_jobs=self.config["models"]["random_forest"]["n_jobs"],
                random_state=self.config["data"]["random_state"]
            ),
            "XGBoost": XGBClassifier(
                n_estimators=self.config["models"]["xgboost"]["n_estimators"],
                max_depth=self.config["models"]["xgboost"]["max_depth"],
                learning_rate=self.config["models"]["xgboost"]["learning_rate"],
                eval_metric=self.config["models"]["xgboost"]["eval_metric"],
                scale_pos_weight=scale_pos_weight if self.imbalance_strategy == "class_weights" else 1.0,
                random_state=self.config["data"]["random_state"],
                n_jobs=self.config["models"]["xgboost"]["n_jobs"]
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=self.config["models"]["lightgbm"]["n_estimators"],
                max_depth=self.config["models"]["lightgbm"]["max_depth"],
                learning_rate=self.config["models"]["lightgbm"]["learning_rate"],
                class_weight="balanced" if self.imbalance_strategy == "class_weights" else None,
                random_state=self.config["data"]["random_state"],
                n_jobs=self.config["models"]["lightgbm"]["n_jobs"],
                verbose=self.config["models"]["lightgbm"]["verbose"]
            ),
            "CatBoost": CatBoostClassifier(
                iterations=self.config["models"]["catboost"]["iterations"],
                depth=self.config["models"]["catboost"]["depth"],
                learning_rate=self.config["models"]["catboost"]["learning_rate"],
                auto_class_weights="Balanced" if self.imbalance_strategy == "class_weights" else None,
                random_state=self.config["data"]["random_state"],
                verbose=self.config["models"]["catboost"]["verbose"]
            )
        }
        
        results = []
        
        # Prepare training data based on imbalance strategy
        X_tr_res, y_tr_res = self.handle_imbalance(X_train, y_train)
        
        for name, model in models.items():
            logger.info(f"Training model: {name}...")
            
            with mlflow.start_run(run_name=f"baseline_{name.lower().replace(' ', '_')}"):
                # Log imbalance handling strategy
                mlflow.log_param("imbalance_strategy", self.imbalance_strategy)
                
                # Fit model
                model.fit(X_tr_res, y_tr_res)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Support models without predict_proba if any, otherwise get probs
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)
                    
                # Compute metrics
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(y_test, y_prob)
                f1 = f1_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                
                # Log metrics in MLflow
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("pr_auc", pr_auc)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("precision", prec)
                
                results.append({
                    "Model": name,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1,
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc
                })
                
                logger.info(f"{name} Results - F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}, Recall: {rec:.4f}")
                
                # Save models locally
                model_path = os.path.join(self.artifact_dir, f"{name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                
        comparison_df = pd.DataFrame(results)
        # Sort models based on F1-Score and PR-AUC
        comparison_df = comparison_df.sort_values(by=["F1-Score", "PR-AUC"], ascending=False).reset_index(drop=True)
        
        # Save comparison results
        os.makedirs(self.evaluation_dir, exist_ok=True)
        comparison_path = os.path.join(self.evaluation_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")
        print("\nModel Comparison Table:")
        print(comparison_df.to_markdown())
        
        return comparison_df

    def get_best_model_name(self, comparison_df: pd.DataFrame) -> str:
        """Returns the name of the top-performing model."""
        best_model = comparison_df.iloc[0]["Model"]
        logger.info(f"Best performing baseline model selected: {best_model}")
        return best_model

def main():
    # Execute full baseline training workflow
    # 1. Load data
    loader = DataLoader()
    raw_df = loader.load_data()
    
    # 2. Engineer features
    fe = FeatureEngineer()
    engineered_df = fe.transform_df(raw_df)
    
    # 3. Preprocess
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.split_data(engineered_df)
    
    # Fit and transform preprocessor
    X_train_trans = prep.fit_transform(X_train)
    prep.save_pipeline()
    
    # Transform test set
    X_test_trans = prep.transform(X_test)
    
    # 4. Train baseline models
    trainer = ModelTrainer()
    comparison = trainer.train_baseline_models(
        X_train_trans, y_train, 
        X_test_trans, y_test, 
        prep.feature_names_out_
    )
    
    best_model_name = trainer.get_best_model_name(comparison)
    print(f"Workflow complete. Best model: {best_model_name}")

if __name__ == "__main__":
    main()
