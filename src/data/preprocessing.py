# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_clean_data(raw_data_path="data/raw/HeartFailureDataset.csv"):
    """
    Loads raw Kaggle CVD dataset, fixes age, handles BP outliers,
    engineers BMI, obesity classes, and key clinical cardiovascular features.
    Saves the cleaned, feature-enriched file.
    """
    # Read CSV (handles both comma and semicolon separators)
    try:
        df = pd.read_csv(raw_data_path, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(raw_data_path, sep=',')
    except Exception:
        df = pd.read_csv(raw_data_path, sep=',')

    # Drop non-predictive ID column
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 1. Convert age from days to years
    if df['age'].max() > 1000:
        df['age'] = (df['age'] / 365.25).round(1)

    # 2. Handle Blood Pressure Anomalies & Swaps
    # Swap if ap_hi < ap_lo (recorded incorrectly)
    mask = df['ap_hi'] < df['ap_lo']
    df.loc[mask, ['ap_hi', 'ap_lo']] = df.loc[mask, ['ap_lo', 'ap_hi']].values

    # Remove extreme unphysical outliers
    df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]

    # 3. Engineer BMI and Obesity Classification (Stage 1 Target)
    df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

    # Filter extreme unphysical BMI outliers
    df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]

    # Map BMI to Obesity Categories
    # 0: Underweight (<18.5), 1: Normal (18.5-24.9), 2: Overweight (25-29.9),
    # 3: Obese Class I (30-34.9), 4: Obese Class II/III (>=35)
    conditions = [
        (df['bmi'] < 18.5),
        (df['bmi'] >= 18.5) & (df['bmi'] < 25.0),
        (df['bmi'] >= 25.0) & (df['bmi'] < 30.0),
        (df['bmi'] >= 30.0) & (df['bmi'] < 35.0),
        (df['bmi'] >= 35.0)
    ]
    choices = [0, 1, 2, 3, 4]
    df['obesity_class'] = np.select(conditions, choices, default=1)

    # ----------------------------------------------------------------
    # 4. FIX: Engineer key clinical cardiovascular features
    # ----------------------------------------------------------------

    # Pulse Pressure: indicator of arterial stiffness and aortic compliance.
    # High pulse pressure (>60 mmHg) is independently associated with CVD risk.
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # Mean Arterial Pressure (MAP): average perfusion pressure to organs.
    # MAP < 70 mmHg may indicate inadequate organ perfusion.
    df['map'] = (df['ap_lo'] + (df['pulse_pressure'] / 3.0)).round(2)

    # High-Risk Metabolic Interaction Flag:
    # Combined above-normal cholesterol AND above-normal glucose doubles CVD risk.
    df['high_chol_gluc'] = ((df['cholesterol'] > 1) & (df['gluc'] > 1)).astype(int)

    # Save cleaned file
    processed_path = "data/processed/cleaned_cardio_dataset.csv"
    df.to_csv(processed_path, index=False)
    print(f"Dataset cleaned & feature-engineered! Shape: {df.shape}")
    print(f"  >> Added clinical features: pulse_pressure, map, high_chol_gluc")
    print(f"  >> Saved to: {processed_path}")

    return df


def prepare_splits(df, target_col='cardio', test_size=0.2, random_state=42):
    """Splits cleaned dataframe into stratified train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def validate_patient_inputs(age, ap_hi, ap_lo, height, weight, bmi=None):
    """
    Validates patient clinical inputs at inference time.
    Returns a list of warning strings (empty list if all valid).
    """
    warnings = []
    if ap_lo >= ap_hi:
        warnings.append(
            f"⚠️ Clinical Anomaly: Diastolic BP ({ap_lo} mmHg) ≥ Systolic BP ({ap_hi} mmHg). "
            "Please verify blood pressure readings."
        )
    if ap_hi > 180:
        warnings.append(
            f"⚠️ Extreme Systolic BP detected ({ap_hi} mmHg). "
            "This may indicate a hypertensive crisis. Please seek immediate medical attention."
        )
    if bmi is not None:
        if bmi < 10 or bmi > 60:
            warnings.append(
                f"⚠️ Calculated BMI ({bmi:.1f} kg/m²) is outside physiologically plausible range. "
                "Please check height and weight values."
            )
    if age < 18:
        warnings.append(
            "⚠️ This model was validated on adults aged ≥ 18 years. "
            "Predictions for younger patients may be unreliable."
        )
    return warnings


if __name__ == "__main__":
    cleaned_df = load_and_clean_data()
    X_train, X_test, y_train, y_test = prepare_splits(cleaned_df)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")