import os
import pandas as pd
from src.preprocessing.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.preprocessing.preprocessor import DataPreprocessor
from src.evaluation.evaluator import ModelEvaluator
from src.explainability.explain import FraudExplainer
from src.utils.logger import get_logger

logger = get_logger("evaluate_and_explain")

def main():
    logger.info("Initializing evaluation and explainability workflow...")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.load_data()
    
    # 2. Feature Engineering
    fe = FeatureEngineer()
    engineered_df = fe.transform_df(raw_df)
    
    # 3. Preprocess
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.split_data(engineered_df)
    
    # Load fitted pipeline
    prep.load_pipeline()
    X_test_trans = prep.transform(X_test)
    
    # 4. Evaluate Final Model
    evaluator = ModelEvaluator()
    evaluator.evaluate(X_test_trans, y_test)
    
    # 5. Generate SHAP Plots
    explainer = FraudExplainer()
    # Compute on a representative sample of test set (e.g. 500 rows) to calculate SHAP values quickly
    logger.info("Calculating SHAP values for test set samples...")
    test_sample_size = min(500, len(X_test_trans))
    
    # Stratify sample the test set for SHAP to preserve class ratio
    test_df_tmp = pd.DataFrame(X_test_trans)
    test_df_tmp['target'] = y_test.values
    
    grouped = test_df_tmp.groupby('target', group_keys=False)
    frac = test_sample_size / len(test_df_tmp)
    shap_sample = grouped.apply(lambda x: x.sample(frac=frac, random_state=42))
    
    # Trim to exact size if needed
    if len(shap_sample) != test_sample_size:
        shap_sample = shap_sample.sample(n=test_sample_size, random_state=42)
        
    X_shap_sample = shap_sample.drop(columns=['target']).values
    
    explainer.save_global_importance_plots(X_shap_sample, prep.feature_names_out_)
    logger.info("Evaluation and Explainability assets generated successfully!")

if __name__ == "__main__":
    main()
