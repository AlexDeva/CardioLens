# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainability.explain import (
    generate_wellness_recommendations,
    get_local_feature_contributions
)
from src.data.preprocessing import validate_patient_inputs

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioLens AI Engine",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LOAD TRAINED MODELS (CACHED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    s1_model = joblib.load(os.path.join(model_dir, "stage1_obesity_xgboost.joblib"))
    s2_model = joblib.load(os.path.join(model_dir, "stage2_cvd_lightgbm.joblib"))
    return s1_model, s2_model

try:
    s1_model, s2_model = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading models from models/ directory: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────
st.title("🫀 CardioLens AI Engine")
st.markdown(
    "**Multi-Stage Cardiometabolic Risk Assessment & Explainable Clinical Decision Support System**"
)
st.divider()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT VITALS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📋 Personal Clinical Inputs")

age = st.sidebar.slider("Age (Years)", 18, 80, 45)
gender_label = st.sidebar.radio("Gender", ["Female", "Male"])
gender = 2 if gender_label == "Male" else 1

height = st.sidebar.number_input("Height (cm)", 130, 210, 168)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 72)

ap_hi = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 80, 200, 125)
ap_lo = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 50, 120, 80)

cholesterol_label = st.sidebar.selectbox(
    "Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"]
)
chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}

gluc_label = st.sidebar.selectbox(
    "Glucose Level", ["Normal", "Above Normal", "Well Above Normal"]
)
gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}

st.sidebar.subheader("Lifestyle Factors")
smoke = st.sidebar.checkbox("Smoker")
alco = st.sidebar.checkbox("Alcohol Intake")
active = st.sidebar.checkbox("Physically Active", value=True)

# ─────────────────────────────────────────────────────────────
# CALCULATE ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────
bmi = round(weight / ((height / 100) ** 2), 2)
pulse_pressure = ap_hi - ap_lo
map_val = round(ap_lo + (pulse_pressure / 3.0), 2)
chol_val = chol_map[cholesterol_label]
gluc_val = gluc_map[gluc_label]
high_chol_gluc = int((chol_val > 1) and (gluc_val > 1))

# ─────────────────────────────────────────────────────────────
# INPUT VALIDATION WARNINGS
# ─────────────────────────────────────────────────────────────
input_warnings = validate_patient_inputs(
    age=age, ap_hi=ap_hi, ap_lo=ap_lo,
    height=height, weight=weight, bmi=bmi
)
if input_warnings:
    for w in input_warnings:
        st.warning(w)

# ─────────────────────────────────────────────────────────────
# STAGE 1 INFERENCE — OBESITY RISK
# ─────────────────────────────────────────────────────────────
input_s1 = pd.DataFrame([{
    'age': age, 'gender': gender, 'height': height, 'weight': weight,
    'smoke': int(smoke), 'alco': int(alco), 'active': int(active)
}])

# FIX: Sum P(Overweight) + P(Obese I) + P(Obese II/III) = P(class >= 2)
s1_probs_all = s1_model.predict_proba(input_s1)[0]
obesity_risk_pct = float(s1_probs_all[2:].sum()) * 100  # True overweight/obesity probability

# ─────────────────────────────────────────────────────────────
# STAGE 2 INFERENCE — CVD RISK
# ─────────────────────────────────────────────────────────────
input_s2 = pd.DataFrame([{
    'age': age, 'gender': gender, 'ap_hi': ap_hi, 'ap_lo': ap_lo,
    'cholesterol': chol_val, 'gluc': gluc_val,
    'smoke': int(smoke), 'alco': int(alco), 'active': int(active),
    'bmi': bmi,
    'pulse_pressure': pulse_pressure,   # FIX: engineered feature
    'map': map_val,                      # FIX: engineered feature
    'high_chol_gluc': high_chol_gluc,   # FIX: engineered feature
    'pred_obesity_risk': obesity_risk_pct / 100.0  # FIX: corrected Stage 1 output
}])

cvd_risk_prob = s2_model.predict_proba(input_s2)[0][1] * 100

# ─────────────────────────────────────────────────────────────
# SECTION 1: METABOLIC & CVD RISK SCORES
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Stage 1: Metabolic Profile")
    st.metric(label="Calculated Body Mass Index (BMI)", value=f"{bmi} kg/m²")
    st.metric(label="Pulse Pressure", value=f"{pulse_pressure} mmHg",
              help="Pulse Pressure = Systolic − Diastolic. >60 mmHg indicates arterial stiffness.")
    st.metric(label="Mean Arterial Pressure (MAP)", value=f"{map_val:.1f} mmHg",
              help="MAP = Diastolic + (Pulse Pressure / 3). Reflects average organ perfusion pressure.")

    if bmi < 18.5:
        st.info("🔵 Obesity Status: Underweight")
    elif 18.5 <= bmi < 25:
        st.success("✅ Obesity Status: Normal Weight")
    elif 25 <= bmi < 30:
        st.warning("🟡 Obesity Status: Overweight")
    elif 30 <= bmi < 35:
        st.error("🔴 Obesity Status: Obese Class I")
    else:
        st.error("🔴 Obesity Status: Obese Class II / III")

    st.metric(
        label="Predicted Overweight/Obesity Risk (Stage 1)",
        value=f"{obesity_risk_pct:.1f}%",
        help="P(BMI ≥ 25) — probability of being overweight or obese based on lifestyle inputs."
    )

with col2:
    st.subheader("🎯 Stage 2: 10-Year CVD Risk Score")
    st.metric(label="Estimated Cardiovascular Disease Risk", value=f"{cvd_risk_prob:.1f}%")

    if cvd_risk_prob < 30:
        st.success("🟢 Risk Status: LOW RISK")
    elif 30 <= cvd_risk_prob < 60:
        st.warning("🟡 Risk Status: MODERATE RISK")
    else:
        st.error("🔴 Risk Status: HIGH RISK")

    # Blood pressure reference
    st.markdown("**Blood Pressure Reading:**")
    bp_status = ""
    if ap_hi < 120 and ap_lo < 80:
        bp_status = "✅ Normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_status = "🟡 Elevated"
    elif ap_hi < 140 or (80 <= ap_lo < 90):
        bp_status = "🟠 Stage 1 Hypertension"
    else:
        bp_status = "🔴 Stage 2 Hypertension"
    st.metric(label=f"BP: {ap_hi}/{ap_lo} mmHg", value=bp_status)

st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION 2: FEATURE CONTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────
st.subheader("🔍 Feature Contribution Analysis")
st.write(
    "Relative influence of each clinical input on the Stage 2 cardiovascular risk model. "
    "Bars represent the model's learned decision weight for each feature."
)

contrib_df = get_local_feature_contributions(s2_model, input_s2, list(input_s2.columns))

# Colour bars: top-weighted features in red to flag attention
colors = []
top_3 = set(contrib_df.tail(3)['Feature'].tolist())
for feat in contrib_df['Feature']:
    if feat in top_3:
        colors.append('#e05252')   # Red for top drivers
    else:
        colors.append('#4a90d9')   # Blue for others

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(contrib_df['Feature'], contrib_df['ModelWeight'], color=colors)
ax.set_xlabel("Model Decision Weight (%)", fontsize=11)
ax.set_title("CardioLens Stage 2 Risk Drivers", fontsize=13, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
for bar, val in zip(bars, contrib_df['ModelWeight']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va='center', ha='left', fontsize=9)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION 3: PERSONALIZED WELLNESS & LIFESTYLE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
st.subheader("💡 Personalised Wellness & Lifestyle Action Plan")

recs = generate_wellness_recommendations(
    cvd_risk_pct=cvd_risk_prob,
    bmi=bmi,
    ap_hi=ap_hi,
    ap_lo=ap_lo,
    smoke=int(smoke),
    alco=int(alco),
    active=int(active),
    cholesterol=chol_val,
    gluc=gluc_val,
    obesity_risk_pct=obesity_risk_pct
)

# Headline
if recs['risk_tier'] == 'low':
    st.success(recs['headline'])
elif recs['risk_tier'] == 'moderate':
    st.warning(recs['headline'])
else:
    st.error(recs['headline'])

# Two columns: protective factors | risk factors to watch
col_pos, col_watch = st.columns(2)

with col_pos:
    st.markdown("#### ✅ Your Protective Factors")
    for pf in recs['positive_factors']:
        st.markdown(f"- {pf}")

with col_watch:
    st.markdown("#### ⚠️ Factors Requiring Attention")
    for wf in recs['watch_factors']:
        st.markdown(f"- {wf}")

st.markdown("---")
st.markdown("#### 📋 Personalised Action Plan")
for i, action in enumerate(recs['action_plan'], 1):
    with st.expander(f"Action {i}: {action[:70]}..."):
        st.markdown(action)

st.divider()

# ─────────────────────────────────────────────────────────────
# SECTION 4: MEDICAL DISCLAIMER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 16px;
    ">
        <h4 style="color: #856404; margin-top: 0;">
            ⚕️ Medical Disclaimer — Important Notice
        </h4>
        <p style="color: #533f03; font-size: 0.95rem; margin-bottom: 8px;">
            <strong>CardioLens is an AI-powered educational and research decision support tool.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.</strong>
        </p>
        <ul style="color: #533f03; font-size: 0.92rem;">
            <li>The risk scores and lifestyle recommendations generated by this system are
                <strong>statistical estimates</strong> derived from population-level data and
                are intended for <strong>informational and educational purposes only</strong>.</li>
            <li>This tool has <strong>not been reviewed or approved</strong> by any medical
                regulatory authority (e.g., FDA, CDSCO, CE Mark) for clinical use.</li>
            <li>The model may not account for your complete medical history, current medications,
                genetic predispositions, or other individualised health factors.</li>
            <li><strong>Always consult a qualified and licensed physician, cardiologist, or
                healthcare professional</strong> before making any decisions regarding your health,
                medication, or treatment plan.</li>
            <li>If you are experiencing chest pain, shortness of breath, dizziness, or any acute
                cardiac symptoms, <strong>please seek emergency medical attention immediately</strong>.</li>
        </ul>
        <p style="color: #533f03; font-size: 0.88rem; margin-bottom: 0;">
            <em>CardioLens v2.0 — For research and educational use only.
            Model trained on the Kaggle Cardiovascular Disease Dataset (70,000 records).
            ROC-AUC: ~0.80 | This system does not store any patient data.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


