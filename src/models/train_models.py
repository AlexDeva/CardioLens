# src/models/train_models.py
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, recall_score, precision_score, brier_score_loss,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.preprocessing import prepare_splits


def train_cardiolens_pipeline():
    print("Loading cleaned dataset from data/processed/...")
    df = pd.read_csv("data/processed/cleaned_cardio_dataset.csv")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = prepare_splits(df, target_col='cardio')

    # ---------------------------------------------------------------
    # STAGE 1: METABOLIC / OBESITY MODEL (XGBoost)
    # Predicts obesity_class based on demographic & lifestyle features
    # ---------------------------------------------------------------
    print("\n--- Training Stage 1: XGBoost Obesity Classification Model ---")
    stage1_features = ['age', 'gender', 'height', 'weight', 'smoke', 'alco', 'active']

    X_train_s1 = X_train[stage1_features]
    y_train_s1 = X_train['obesity_class']
    X_test_s1 = X_test[stage1_features]
    y_test_s1 = X_test['obesity_class']

    model_stage1 = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    model_stage1.fit(X_train_s1, y_train_s1)

    s1_preds = model_stage1.predict(X_test_s1)
    print(f"Stage 1 Accuracy (Obesity Class): {accuracy_score(y_test_s1, s1_preds):.4f}")

    # ---------------------------------------------------------------
    # FIX 1: CORRECT PROBABILITY CALCULATION FOR STAGE 1 OUTPUT
    # Classes: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese I, 4=Obese II/III
    # Obesity risk = P(Overweight or Obese) = P(class>=2) = sum of cols [2, 3, 4]
    # ---------------------------------------------------------------
    print("\n>>> APPLYING FIX: Summing P(class>=2) for true Overweight/Obese risk <<<")

    # FIX 2: OUT-OF-FOLD (OOF) PREDICTIONS TO ELIMINATE TARGET LEAKAGE
    # Stage 2 is trained on OOF Stage 1 probabilities so it learns from
    # realistic out-of-sample uncertainty — not optimistic in-sample predictions.
    print(">>> APPLYING FIX: Using 5-Fold OOF predictions to prevent Stage 2 target leakage <<<")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s1_oof_probs = cross_val_predict(model_stage1, X_train_s1, y_train_s1, cv=cv, method='predict_proba')
    # Sum probabilities for Overweight (2) + Obese I (3) + Obese II/III (4)
    s1_train_obesity_risk = s1_oof_probs[:, 2:].sum(axis=1)

    # For test set: predict directly with the fitted Stage 1 model
    s1_test_probs = model_stage1.predict_proba(X_test_s1)
    s1_test_obesity_risk = s1_test_probs[:, 2:].sum(axis=1)

    print(f"  OOF train obesity risk — mean: {s1_train_obesity_risk.mean():.3f}, std: {s1_train_obesity_risk.std():.3f}")
    print(f"  Test obesity risk      — mean: {s1_test_obesity_risk.mean():.3f}, std: {s1_test_obesity_risk.std():.3f}")

    # ---------------------------------------------------------------
    # STAGE 2: CARDIOVASCULAR DISEASE RISK MODEL (LightGBM)
    # Uses clinical vitals + engineered features + corrected Stage 1 obesity risk
    # ---------------------------------------------------------------
    print("\n--- Training Stage 2: LightGBM CVD Risk Model ---")

    # FIX 3: Include engineered clinical features from preprocessing
    stage2_features = [
        'age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'bmi',
        'pulse_pressure', 'map', 'high_chol_gluc'  # NEW: engineered features
    ]

    X_train_s2 = X_train[stage2_features].copy()
    X_train_s2['pred_obesity_risk'] = s1_train_obesity_risk  # OOF, corrected

    X_test_s2 = X_test[stage2_features].copy()
    X_test_s2['pred_obesity_risk'] = s1_test_obesity_risk  # corrected

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

    print(f"\nStage 2 Accuracy (CVD Risk):     {accuracy_score(y_test, s2_preds):.4f}")
    print(f"Stage 2 ROC-AUC Score:           {roc_auc_score(y_test, s2_probs):.4f}")
    print(f"Stage 2 PR-AUC Score:            {average_precision_score(y_test, s2_probs):.4f}")
    print(f"Stage 2 F1-Score:                {f1_score(y_test, s2_preds):.4f}")
    print(f"Stage 2 Sensitivity (Recall):    {recall_score(y_test, s2_preds):.4f}")
    print(f"Stage 2 Precision (PPV):         {precision_score(y_test, s2_preds):.4f}")
    print(f"Stage 2 Brier Score Loss:        {brier_score_loss(y_test, s2_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, s2_preds))

    # Feature Importance Summary
    imp_df = pd.DataFrame({
        'Feature': X_test_s2.columns,
        'Importance': model_stage2.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nStage 2 Feature Importances (Top 10):")
    print(imp_df.head(10).to_string(index=False))

    # ---------------------------------------------------------------
    # SAVE MODEL ARTIFACTS
    # ---------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_stage1, "models/stage1_obesity_xgboost.joblib")
    joblib.dump(model_stage2, "models/stage2_cvd_lightgbm.joblib")
    print("\n[OK] Model artifacts successfully saved to models/")
    print("   * models/stage1_obesity_xgboost.joblib")
    print("   * models/stage2_cvd_lightgbm.joblib")


if __name__ == "__main__":
    train_cardiolens_pipeline()