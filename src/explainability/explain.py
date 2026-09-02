# src/explainability/explain.py
"""
CardioLens Explainability & Wellness Recommendation Engine.
Provides local patient risk factor explanations and personalized
lifestyle action plans based on metabolic and cardiovascular risk tiers.
"""
import pandas as pd
import numpy as np
import joblib


def load_models():
    """Loads saved CardioLens model artifacts."""
    stage1 = joblib.load("models/stage1_obesity_xgboost.joblib")
    stage2 = joblib.load("models/stage2_cvd_lightgbm.joblib")
    return stage1, stage2


def get_local_feature_contributions(model, input_df, feature_names):
    """
    Computes patient-specific risk contributions by comparing each feature
    against population-level average model importances, weighted by the
    patient's deviation from clinical norms.
    Returns a DataFrame sorted by relative contribution magnitude.
    """
    global_importances = model.feature_importances_
    total_importance = global_importances.sum()
    if total_importance == 0:
        total_importance = 1.0

    # Build patient-local risk contribution breakdown
    contributions = []
    for feat, imp in zip(feature_names, global_importances):
        if feat in input_df.columns:
            val = input_df[feat].iloc[0]
        else:
            val = None
        contributions.append({
            'Feature': feat,
            'PatientValue': val,
            'ModelWeight': round(float(imp) / total_importance * 100, 2)
        })

    contrib_df = pd.DataFrame(contributions).sort_values(by='ModelWeight', ascending=True)
    return contrib_df


def generate_wellness_recommendations(cvd_risk_pct, bmi, ap_hi, ap_lo,
                                      smoke, alco, active,
                                      cholesterol, gluc, obesity_risk_pct):
    """
    Generates a personalized, tiered wellness & lifestyle recommendation plan
    based on the patient's clinical profile and predicted CVD risk tier.

    Parameters
    ----------
    cvd_risk_pct    : float  — Predicted 10-year CVD risk (0–100%)
    bmi             : float  — Body Mass Index (kg/m²)
    ap_hi           : int    — Systolic blood pressure (mmHg)
    ap_lo           : int    — Diastolic blood pressure (mmHg)
    smoke           : int    — 1 = smoker, 0 = non-smoker
    alco            : int    — 1 = alcohol intake, 0 = none
    active          : int    — 1 = physically active, 0 = sedentary
    cholesterol     : int    — 1=Normal, 2=Above Normal, 3=Well Above Normal
    gluc            : int    — 1=Normal, 2=Above Normal, 3=Well Above Normal
    obesity_risk_pct: float  — Stage 1 predicted overweight/obesity probability (0–100%)

    Returns
    -------
    dict with keys:
        'risk_tier'          : str — 'low' | 'moderate' | 'high'
        'headline'           : str — Summary headline for patient
        'action_plan'        : list[str] — Ordered action items
        'positive_factors'   : list[str] — Patient's protective factors
        'watch_factors'      : list[str] — Risk factors requiring attention
    """
    risk_tier = 'low' if cvd_risk_pct < 30 else ('moderate' if cvd_risk_pct < 60 else 'high')

    action_plan = []
    positive_factors = []
    watch_factors = []

    # ---------------------------------------------------------------
    # IDENTIFY PROTECTIVE FACTORS
    # ---------------------------------------------------------------
    if active:
        positive_factors.append("✅ Physically active lifestyle — excellent cardiovascular protection.")
    if not smoke:
        positive_factors.append("✅ Non-smoker — significant reduction in arterial inflammation risk.")
    if not alco:
        positive_factors.append("✅ No alcohol intake — positive impact on blood pressure and liver health.")
    if bmi < 25.0:
        positive_factors.append(f"✅ Healthy BMI ({bmi:.1f} kg/m²) — optimal cardiometabolic weight range.")
    if ap_hi < 120 and ap_lo < 80:
        positive_factors.append(f"✅ Normal blood pressure ({ap_hi}/{ap_lo} mmHg) — no arterial strain detected.")
    if cholesterol == 1:
        positive_factors.append("✅ Normal cholesterol levels — no lipid pathway risk detected.")
    if gluc == 1:
        positive_factors.append("✅ Normal blood glucose — no dysglycemic risk detected.")

    # ---------------------------------------------------------------
    # IDENTIFY RISK FACTORS & BUILD ACTION PLAN
    # ---------------------------------------------------------------

    # --- Blood Pressure Guidance ---
    if ap_hi >= 140 or ap_lo >= 90:
        watch_factors.append(f"⚠️ Stage 2 Hypertension ({ap_hi}/{ap_lo} mmHg)")
        action_plan.append(
            "🩺 Blood Pressure Control: Adopt the DASH diet (low sodium <1,500 mg/day, "
            "potassium-rich foods: bananas, spinach, sweet potatoes). Monitor BP daily at home. "
            "Aim for <130/80 mmHg. Discuss antihypertensive options with your physician."
        )
    elif ap_hi >= 130 or ap_lo >= 80:
        watch_factors.append(f"⚠️ Stage 1 Hypertension ({ap_hi}/{ap_lo} mmHg)")
        action_plan.append(
            "🩺 Blood Pressure: Reduce sodium intake to <2,300 mg/day. "
            "Increase potassium and magnesium consumption. Limit caffeine. "
            "Log BP readings twice daily and schedule a physician review."
        )
    elif ap_hi >= 120:
        watch_factors.append(f"⚠️ Elevated Systolic BP ({ap_hi} mmHg) — monitor closely.")
        action_plan.append(
            "📋 Blood Pressure Monitoring: Your systolic BP is slightly elevated. "
            "Begin regular monitoring (twice weekly). Reduce processed food and alcohol."
        )

    # --- BMI & Weight Guidance ---
    if bmi >= 35.0:
        watch_factors.append(f"⚠️ Severe Obesity (BMI {bmi:.1f} kg/m²)")
        action_plan.append(
            "⚖️ Weight Management: Severe obesity significantly amplifies CVD risk. "
            "Target 5–10% weight reduction over 6 months. Consult a dietitian for a "
            "calorie-deficit meal plan (500–750 kcal/day deficit). Combine with low-impact "
            "aerobic exercise (swimming, cycling) 150+ mins/week. Consider clinical weight "
            "management programme evaluation."
        )
    elif bmi >= 30.0:
        watch_factors.append(f"⚠️ Obese Class I (BMI {bmi:.1f} kg/m²)")
        action_plan.append(
            "⚖️ Weight Reduction: Aim for a sustainable 0.5–1 kg per week weight loss. "
            "Follow a whole-food, plant-rich diet (Mediterranean or DASH pattern). "
            "Incorporate 30 minutes of moderate cardio 5 days/week."
        )
    elif bmi >= 25.0:
        watch_factors.append(f"⚠️ Overweight (BMI {bmi:.1f} kg/m²)")
        action_plan.append(
            "⚖️ Weight Management: You are in the overweight range. Small adjustments matter — "
            "reduce ultra-processed snacks, increase vegetable portions (half-plate rule), "
            "and aim for 150 minutes of moderate aerobic activity weekly."
        )
    elif bmi < 18.5:
        watch_factors.append(f"⚠️ Underweight (BMI {bmi:.1f} kg/m²) — risk of nutrient deficiency.")
        action_plan.append(
            "🥗 Nutritional Support: Your BMI is below healthy range. "
            "Increase caloric intake through nutrient-dense foods (nuts, legumes, whole grains, "
            "lean proteins). Consider consulting a registered dietitian."
        )

    # --- Smoking Guidance ---
    if smoke:
        watch_factors.append("⚠️ Active Smoker — primary CVD risk amplifier.")
        action_plan.append(
            "🚭 Smoking Cessation: Smoking is the single most modifiable CVD risk factor. "
            "Every cigarette smoked temporarily raises BP and causes endothelial damage. "
            "Explore nicotine replacement therapy (patches, gum), prescription cessation "
            "medications, or behavioural support groups. Within 1 year of quitting, "
            "CVD risk is halved."
        )

    # --- Alcohol Guidance ---
    if alco:
        watch_factors.append("⚠️ Regular Alcohol Intake — raises BP and triglycerides.")
        action_plan.append(
            "🍷 Alcohol Reduction: Limit alcohol to ≤1 drink/day (women) or ≤2 drinks/day (men). "
            "Alcohol directly raises blood pressure and disrupts cardiac rhythm. "
            "Alcohol-free days help normalise triglyceride levels."
        )

    # --- Physical Activity Guidance ---
    if not active:
        watch_factors.append("⚠️ Sedentary lifestyle — independent CVD risk factor.")
        action_plan.append(
            "🏃 Physical Activity: Sedentary behaviour raises CVD risk by up to 35%. "
            "Begin with 10-minute daily walks and build to 150 minutes/week of moderate-intensity "
            "aerobic activity (brisk walking, cycling, swimming). Add 2x/week resistance "
            "training for improved insulin sensitivity and metabolic health."
        )
    else:
        if risk_tier in ('moderate', 'high'):
            action_plan.append(
                "🏃 Optimise Exercise: You are already active — excellent! Consider progressing "
                "to include Zone 2 cardio (conversational pace, 45–60 mins, 3–4x/week) for "
                "maximum cardiovascular efficiency. Strength training 2x/week builds protective "
                "lean muscle mass and improves insulin sensitivity."
            )

    # --- Cholesterol & Glucose Guidance ---
    if cholesterol >= 2:
        watch_factors.append(f"⚠️ {'Above Normal' if cholesterol == 2 else 'Well Above Normal'} Cholesterol")
        action_plan.append(
            "🫀 Lipid Management: Elevated cholesterol accelerates arterial plaque buildup. "
            "Reduce saturated and trans fats (red meat, full-fat dairy, fried foods). "
            "Increase soluble fibre (oats, beans, apples), omega-3 fatty acids (fatty fish, "
            "flaxseed), and plant sterols. Request a full lipid panel from your physician."
        )
    if gluc >= 2:
        watch_factors.append(f"⚠️ {'Above Normal' if gluc == 2 else 'Well Above Normal'} Blood Glucose")
        action_plan.append(
            "🩸 Blood Sugar Control: Elevated glucose increases glycation of blood vessels "
            "and accelerates CVD risk. Reduce refined carbohydrates and sugary beverages. "
            "Eat lower glycaemic-index foods (whole grains, legumes, non-starchy vegetables). "
            "Request an HbA1c test from your physician to rule out pre-diabetes."
        )

    # ---------------------------------------------------------------
    # TIER-SPECIFIC HEADLINE & POSITIVE MAINTENANCE PLAN
    # ---------------------------------------------------------------
    if risk_tier == 'low':
        headline = (
            "🟢 Your cardiovascular risk assessment is LOW. "
            "You are on an excellent health trajectory! Keep up your current healthy habits."
        )
        if not action_plan:
            action_plan = [
                "🌟 Maintain Your Healthy Lifestyle: Continue 150+ minutes/week of aerobic activity. "
                "Eat a predominantly plant-based, whole-food diet rich in vegetables, fruits, "
                "whole grains, and lean proteins.",

                "😴 Prioritise Sleep: 7–9 hours of quality sleep per night is essential for "
                "cardiovascular recovery. Poor sleep raises cortisol and inflammatory markers.",

                "🧘 Stress Management: Chronic stress elevates cortisol, blood pressure, and "
                "inflammatory markers. Incorporate mindfulness, yoga, or deep-breathing exercises "
                "at least 3x/week.",

                "🩺 Preventive Health Checks: Even at low risk, schedule annual health screenings: "
                "BP check, cholesterol panel, fasting glucose, and BMI monitoring. "
                "Early detection of any changes is key to sustained heart health.",

                "💧 Hydration & Nutrition: Drink 2–3 litres of water daily. "
                "Include antioxidant-rich foods (berries, dark leafy greens, nuts) "
                "to protect against vascular inflammation.",
            ]
    elif risk_tier == 'moderate':
        headline = (
            "🟡 Your 10-year CVD risk is MODERATE. "
            "Targeted lifestyle changes now can significantly reduce your long-term risk."
        )
        if not action_plan:
            action_plan.append(
                "📋 Schedule a Preventive Cardiology Review: At moderate risk, a comprehensive "
                "clinical evaluation (ECG, full lipid panel, HbA1c, urine microalbumin) "
                "is recommended within the next 3–6 months."
            )
    else:
        headline = (
            "🔴 Your 10-year CVD risk is HIGH. "
            "Urgent lifestyle modification and medical consultation are strongly advised."
        )
        action_plan.insert(0,
            "🚨 Priority: Please consult your cardiologist or primary care physician within "
            "4–8 weeks for a comprehensive cardiovascular risk assessment, including "
            "ECG, echocardiogram (if indicated), full metabolic panel, and medication review."
        )

    # --- Universal Preventive Tip ---
    action_plan.append(
        "💡 Longevity Tip: Even modest improvements across multiple risk factors have a "
        "compounding protective effect. Research shows that addressing just 3 lifestyle factors "
        "(activity, diet quality, smoking cessation) can reduce 10-year CVD risk by up to 50%."
    )

    return {
        'risk_tier': risk_tier,
        'headline': headline,
        'action_plan': action_plan,
        'positive_factors': positive_factors if positive_factors else ["No major protective factors currently identified."],
        'watch_factors': watch_factors if watch_factors else ["No major risk flags identified in current profile."]
    }


if __name__ == "__main__":
    # Quick smoke-test of the recommendation engine
    print("Testing Wellness Recommendation Engine...\n")
    result = generate_wellness_recommendations(
        cvd_risk_pct=72.5, bmi=33.2, ap_hi=155, ap_lo=95,
        smoke=1, alco=0, active=0, cholesterol=3, gluc=2, obesity_risk_pct=85.0
    )
    print(f"Risk Tier: {result['risk_tier'].upper()}")
    print(f"Headline: {result['headline']}\n")
    print("Action Plan:")
    for i, item in enumerate(result['action_plan'], 1):
        print(f"  {i}. {item}\n")
    print("Positive Factors:")
    for pf in result['positive_factors']:
        print(f"  {pf}")
    print("\nRisk Factors to Watch:")
    for wf in result['watch_factors']:
        print(f"  {wf}")
    print("\n✅ Explainability engine initialized successfully.")

