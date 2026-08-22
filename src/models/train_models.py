# src/models/train_models.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.preprocessing import prepare_splits

def train_cardiolens_pipeline():
    print("Loading cleaned dataset from data/processed/...")
    df = pd.read_csv("data/processed/cleaned_cardio_dataset.csv")
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = prepare_splits(df, target_col='cardio')
    
    # -------------------------------------------------------------
    # STAGE 1: METABOLIC / OBESITY MODEL (XGBoost)
    # Predicts obesity_class based on demographic & lifestyle features
    # -------------------------------------------------------------
    print("\n--- Training Stage 1: XGBoost Obesity Model ---")
    stage1_features = ['age', 'gender', 'height', 'weight', 'smoke', 'alco', 'active']
    
    X_train_s1 = X_train[stage1_features]
    y_train_s1 = X_train['obesity_class']
    
    X_test_s1 = X_test[stage1_features]
    y_test_s1 = X_test['obesity_class']
    
    model_stage1 = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model_stage1.fit(X_train_s1, y_train_s1)
    
    s1_preds = model_stage1.predict(X_test_s1)
    print(f"Stage 1 Accuracy (Obesity Class): {accuracy_score(y_test_s1, s1_preds):.4f}")

    # -------------------------------------------------------------
    # STAGE 2: CARDIOVASCULAR DISEASE RISK MODEL (LightGBM)
    # Uses clinical vitals + Stage 1 Obesity Predictions
    # -------------------------------------------------------------
    print("\n--- Training Stage 2: LightGBM CVD Risk Model ---")
    
    # Generate Stage 1 prediction probabilities to inject into Stage 2
    s1_train_probs = model_stage1.predict_proba(X_train_s1)[:, 1] # Probability of Overweight/Obese
    s1_test_probs = model_stage1.predict_proba(X_test_s1)[:, 1]
    
    # Construct Stage 2 Feature Set
    stage2_features = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
    
    X_train_s2 = X_train[stage2_features].copy()
    X_train_s2['pred_obesity_risk'] = s1_train_probs
    
    X_test_s2 = X_test[stage2_features].copy()
    X_test_s2['pred_obesity_risk'] = s1_test_probs
    
    model_stage2 = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model_stage2.fit(X_train_s2, y_train)
    
    s2_preds = model_stage2.predict(X_test_s2)
    s2_probs = model_stage2.predict_proba(X_test_s2)[:, 1]
    
    print(f"Stage 2 Accuracy (CVD Risk): {accuracy_score(y_test, s2_preds):.4f}")
    print(f"Stage 2 ROC-AUC Score: {roc_auc_score(y_test, s2_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, s2_preds))

    # -------------------------------------------------------------
    # SAVE MODEL ARTIFACTS
    # -------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_stage1, "models/stage1_obesity_xgboost.joblib")
    joblib.dump(model_stage2, "models/stage2_cvd_lightgbm.joblib")
    print("\nModel artifacts successfully saved to models/")

if __name__ == "__main__":
    train_cardiolens_pipeline()