# CardioLens: Comprehensive Project Documentation & Presentation Guide

## Part 1: Executive Technical Documentation

### 1. Executive Summary & Architecture Overview
CardioLens is an end-to-end, multi-stage Machine Learning Decision Support System (CDSS) designed to evaluate cardiometabolic health. Unlike standard single-model classifiers that analyze vitals in isolation, CardioLens uses a cascade pipeline:

1. Stage 1 (Metabolic Layer): Predicts obesity and body-composition risks using demographic, dietary, and behavioral lifestyle attributes.
2. Stage 2 (Cardiovascular Layer): Ingests clinical vitals along with the Stage 1 metabolic probability output to forecast 10-year Cardiovascular Disease (CVD) event risks.
3. Stage 3 (Explainability Layer): Uses native decision-tree feature attributions to provide transparent, interpretable risk factor breakdowns.
---

### 2. Algorithmic Justification & Comparative Rationale

#### Why XGBoost for Stage 1 (Obesity Stratification)?
* Handles Multi-Class Boundaries: Body Mass Index (BMI) categories and metabolic conditions exhibit sharp, non-linear cutoffs across height, weight, and activity metrics. Gradient Boosted Trees natively isolate these non-linear thresholds better than linear classifiers.
* Regularization (L1/L2): Built-in penalty terms prevent overfitting on survey-driven lifestyle features.
* Performance: Achieved 98.72% accuracy in separating obesity classes.

#### Why LightGBM for Stage 2 (Cardiovascular Risk Scoring)?
* Histogram-Based Binning: LightGBM bins continuous features (like blood pressure readings and age) into discrete buckets, speeding up computation while handling noisy clinical data cleanly.
* Optimal Tree Growth (Leaf-Wise): Leaf-wise tree growth minimizes loss more effectively than depth-wise algorithms on tabular health data, achieving a 0.7985 ROC-AUC score (matching top academic benchmarks on the 70k Kaggle CVD dataset).
* Robustness to Categorical Ranks: Native support for ordered categorical variables (cholesterol and gluc levels 1, 2, 3) without requiring memory-intensive One-Hot Encoding.

#### Advantages Over Standard Single-Model Approaches
* Mitigates Data Leakage: Preprocessing pipelines, scaling, and feature engineering are fit strictly within stratified training splits.
* Biological Cascade Reality: Mirrors clinical pathology—behavioral and dietary factors drive metabolic changes first, which subsequently accelerate cardiovascular risk.
* High Inference Speed & Deployment Safety: Eliminates C-extension runtime dependencies (e.g., Numba/DLL blockers), ensuring smooth execution across locked-down corporate enterprise environments.
---

### 3. User Interface (UI) Interpretation Guide

When reviewing patient profiles on the CardioLens Streamlit Dashboard, use the following clinical reference ranges:

| METRIC | VALUE RANGE | CLINICAL STATUS / ACTION |
|---|---|---|
| Body Mass Index (BMI) | < 18.5 kg/m² | Underweight |
| | 18.5 - 24.9 kg/m² | Normal Weight (Optimal) |
| | 25.0 - 29.9 kg/m² | Overweight |
| | ≥ 30.0 kg/m² | Obese Class |
| Systolic BP (ap_hi) | < 120 mmHg | Normal |
| | 120 - 129 mmHg | Elevated |
| | 130 - 139 mmHg | Stage 1 Hypertension |
| | ≥ 140 mmHg | Stage 2 Hypertension |
| Diastolic BP (ap_lo) | < 80 mmHg | Normal |
| | 80 - 89 mmHg | Stage 1 Hypertension |
| | ≥ 90 mmHg | Stage 2 Hypertension |
| CVD 10-Yr Risk Score | < 30.0% | Low Risk (Green Indicator) |
| | 30.0% - 59.9% | Moderate Risk (Yellow Warning) |
| | ≥ 60.0% | High Risk (Red Alert) |

* Feature Attribution Bar Chart: Displays Gini importance weights for the Stage 2 model. Longer horizontal bars indicate which vitals (e.g., Systolic BP vs. Predicted Obesity Risk) drive the prediction calculation most heavily for that patient group.

---

### 4. Operational Run Guide (Execution Steps)

#### Environment Setup
Open PowerShell from the root `CardioLens` directory:

```powershell
# 1. Install Dependencies
python -m pip install lightgbm xgboost joblib streamlit matplotlib seaborn scikit-learn pandas numpy python-pptx

Step 1: Preprocess Dataset
PowerShell
python src/data/preprocessing.py

Step 2: Train & Serialize Multi-Stage Models
PowerShell
python src/models/train_models.py

Step 3: Launch Interactive Streamlit Dashboard
PowerShell
python -m streamlit run app/streamlit_app.py

---

### File 2: Auto-Generate PowerPoint (`create_presentation.py`)

You can generate the complete PowerPoint presentation (`.pptx`) automatically on your local machine using Python!

1. Install `python-pptx` in your PowerShell terminal:
   ```powershell
   python -m pip install python-pptx