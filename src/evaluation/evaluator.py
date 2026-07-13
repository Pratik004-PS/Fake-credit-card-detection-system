import os
import yaml
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from src.utils.logger import get_logger

logger = get_logger("evaluator")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.artifact_dir = self.config["models"]["artifact_dir"]
        self.evaluation_dir = self.config["models"]["evaluation_dir"]
        self.saved_model_path = os.path.join(self.artifact_dir, self.config["models"]["saved_model"])
        
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def load_model(self):
        """Load final tuned model."""
        if not os.path.exists(self.saved_model_path):
            raise FileNotFoundError(f"Tuned model not found at {self.saved_model_path}")
        return joblib.load(self.saved_model_path)

    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test dataset and return detailed metrics.
        """
        model = self.load_model()
        logger.info("Evaluating model on test dataset...")
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        # Optimize threshold
        best_threshold, best_f1 = self.optimize_threshold(y_test, y_prob)
        logger.info(f"Optimal decision threshold found: {best_threshold:.4f} (F1-score: {best_f1:.4f})")
        
        # Apply optimized threshold
        y_pred_opt = (y_prob >= best_threshold).astype(int)
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        # Calculate default metrics
        metrics_default = self._calculate_metrics(y_test, y_pred_default, y_prob)
        # Calculate optimized metrics
        metrics_opt = self._calculate_metrics(y_test, y_pred_opt, y_prob)
        
        # Plot evaluation curves
        self.plot_roc_curve(y_test, y_prob)
        self.plot_precision_recall_curve(y_test, y_prob, best_threshold)
        self.plot_confusion_matrix(y_test, y_pred_opt, f"Confusion Matrix (Opt Threshold: {best_threshold:.3f})", "cm_optimized.png")
        self.plot_confusion_matrix(y_test, y_pred_default, "Confusion Matrix (Default Threshold: 0.5)", "cm_default.png")
        
        report = {
            "best_threshold": float(best_threshold),
            "metrics_at_0.5": metrics_default,
            "metrics_at_opt": metrics_opt
        }
        
        # Save metrics as json
        metrics_path = os.path.join(self.evaluation_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        print("\n=== Model Performance Metrics ===")
        print(f"Decision Threshold:  0.5000 (Default)  |  {best_threshold:.4f} (Optimized)")
        print(f"Accuracy:           {metrics_default['accuracy']:.4f}            |  {metrics_opt['accuracy']:.4f}")
        print(f"Precision:          {metrics_default['precision']:.4f}            |  {metrics_opt['precision']:.4f}")
        print(f"Recall:             {metrics_default['recall']:.4f}            |  {metrics_opt['recall']:.4f}")
        print(f"F1-Score:           {metrics_default['f1']:.4f}            |  {metrics_opt['f1']:.4f}")
        print(f"ROC-AUC:            {metrics_default['roc_auc']:.4f}            |  {metrics_opt['roc_auc']:.4f}")
        print(f"PR-AUC:             {metrics_default['pr_auc']:.4f}            |  {metrics_opt['pr_auc']:.4f}")
        print(f"MCC:                {metrics_default['mcc']:.4f}            |  {metrics_opt['mcc']:.4f}")
        
        return report

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(pr_auc),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred))
        }

    def optimize_threshold(self, y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
        """
        Finds the decision threshold that maximizes F1 score.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate F1 score for each threshold
        # Precision and recall arrays have one extra element at the end corresponding to threshold=1
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_f1

    def plot_roc_curve(self, y_true: pd.Series, y_prob: np.ndarray):
        """Plot and save ROC Curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        path = os.path.join(self.evaluation_dir, "roc_curve.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {path}")

    def plot_precision_recall_curve(self, y_true: pd.Series, y_prob: np.ndarray, opt_threshold: float):
        """Plot and save Precision-Recall Curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        
        # Find precision and recall at optimal threshold
        if len(thresholds) > 0:
            opt_idx = np.argmin(np.abs(thresholds - opt_threshold))
            plt.plot(recall[opt_idx], precision[opt_idx], 'ro', markersize=8, 
                     label=f'Optimal Threshold ({opt_threshold:.3f})')
                     
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        path = os.path.join(self.evaluation_dir, "precision_recall_curve.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PR curve saved to {path}")

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, title: str, filename: str):
        """Plot and save Confusion Matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(title)
        
        path = os.path.join(self.evaluation_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {path}")

if __name__ == "__main__":
    # Test stub
    evaluator = ModelEvaluator()
    print("Evaluator ready.")
