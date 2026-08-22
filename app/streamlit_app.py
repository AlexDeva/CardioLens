# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os

# Page Config
st.set_page_config(
    page_title="CardioLens AI Engine",
    page_icon="🫀",
    layout="wide"
)

# Load Trained Models
@st.cache_resource
def load_artifacts():
    s1_model = joblib.load("models/stage1_obesity_xgboost.joblib")
    s2_model = joblib.load("models/stage2_cvd_lightgbm.joblib")
    return s1_model, s2_model

try:
    s1_model, s2_model = load_artifacts()
except Exception as e:
    st.error(f"Error loading models from models/ directory: {e}")
    st.stop()

# App Header
st.title("🫀 CardioLens AI Engine")
st.markdown("**Multi-Stage Cardiometabolic Risk Assessment & Explainable Decision Support System**")
st.divider()

# Sidebar - Patient Input Vitals
st.sidebar.header("📋 Patient Clinical Inputs")

age = st.sidebar.slider("Age (Years)", 20, 80, 45)
gender_label = st.sidebar.radio("Gender", ["Female", "Male"])
gender = 2 if gender_label == "Male" else 1

height = st.sidebar.number_input("Height (cm)", 130, 210, 168)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 72)

ap_hi = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 80, 200, 125)
ap_lo = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 50, 120, 80)

cholesterol = st.sidebar.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
chol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}

gluc = st.sidebar.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}

st.sidebar.subheader("Lifestyle Factors")
smoke = st.sidebar.checkbox("Smoker")
alco = st.sidebar.checkbox("Alcohol Intake")
active = st.sidebar.checkbox("Physically Active", value=True)

# Calculate Engineered Features
bmi = round(weight / ((height / 100) ** 2), 2)

# Prepare DataFrames for Inference
input_s1 = pd.DataFrame([{
    'age': age, 'gender': gender, 'height': height, 'weight': weight,
    'smoke': int(smoke), 'alco': int(alco), 'active': int(active)
}])

# Stage 1 Inference
s1_prob = s1_model.predict_proba(input_s1)[0][1]

input_s2 = pd.DataFrame([{
    'age': age, 'gender': gender, 'ap_hi': ap_hi, 'ap_lo': ap_lo,
    'cholesterol': chol_map[cholesterol], 'gluc': gluc_map[gluc],
    'smoke': int(smoke), 'alco': int(alco), 'active': int(active),
    'bmi': bmi, 'pred_obesity_risk': s1_prob
}])

# Stage 2 Inference
cvd_risk_prob = s2_model.predict_proba(input_s2)[0][1] * 100

# Main Display Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Stage 1: Metabolic Profile")
    st.metric(label="Calculated Body Mass Index (BMI)", value=f"{bmi} kg/m²")
    
    if bmi < 18.5:
        st.info("Obesity Status: Underweight")
    elif 18.5 <= bmi < 25:
        st.success("Obesity Status: Normal Weight")
    elif 25 <= bmi < 30:
        st.warning("Obesity Status: Overweight")
    else:
        st.error("Obesity Status: Obese Class")

with col2:
    st.subheader("🎯 Stage 2: 10-Year CVD Risk Score")
    st.metric(label="Estimated Cardiovascular Disease Risk", value=f"{cvd_risk_prob:.1f}%")
    
    if cvd_risk_prob < 30:
        st.success("Risk Status: LOW RISK")
    elif 30 <= cvd_risk_prob < 60:
        st.warning("Risk Status: MODERATE RISK")
    else:
        st.error("Risk Status: HIGH RISK")

st.divider()

# Explainability Section (Native Feature Importance)
st.subheader("🔍 Feature Contribution Analysis")
st.write("Relative influence of clinical inputs on the overall cardiovascular risk score:")

feature_names = input_s2.columns
importances = s2_model.feature_importances_

# Create Feature Importance Plot
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='#1f77b4')
ax.set_xlabel("Model Decision Weight (Gini Importance)")
ax.set_title("CardioLens Stage 2 Risk Drivers")
plt.tight_layout()

st.pyplot(fig)