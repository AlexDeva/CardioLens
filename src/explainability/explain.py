# src/explainability/explain.py
import shap
import dice_ml
import pandas as pd
import joblib

def load_models():
    """Loads saved CardioLens model artifacts."""
    stage1 = joblib.load("models/stage1_obesity_xgboost.joblib")
    stage2 = joblib.load("models/stage2_cvd_lightgbm.joblib")
    return stage1, stage2

def get_shap_explanation(model, input_df):
    """Generates SHAP values for local explanation on a single patient profile."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    return explainer, shap_values

def generate_counterfactuals(model, background_df, target_col, query_instance, total_CFs=2):
    """
    Generates DiCE counterfactuals showing minimal actionable lifestyle changes 
    to shift risk from High (1) to Low (0).
    """
    d = dice_ml.Data(
        dataframe=background_df, 
        continuous_features=['age', 'ap_hi', 'ap_lo', 'bmi'], 
        outcome_name=target_col
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    
    # Generate counterfactuals aiming for target class 0 (Low CVD Risk)
    cf = exp.generate_counterfactuals(
        query_instance, 
        total_CFs=total_CFs, 
        desired_class=0,
        features_to_vary=['ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'bmi'] # Actionable features only
    )
    return cf

if __name__ == "__main__":
    print("Explainability engine initialized successfully.")