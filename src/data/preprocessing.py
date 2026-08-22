# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(raw_data_path="data/raw/HeartFailureDataset.csv"):
    """
    Loads raw Kaggle CVD dataset, fixes age, handles BP outliers, 
    engineers BMI and obesity classes, and saves clean file.
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
    # Swap if ap_hi < ap_lo
    mask = df['ap_hi'] < df['ap_lo']
    df.loc[mask, ['ap_hi', 'ap_lo']] = df.loc[mask, ['ap_lo', 'ap_hi']].values

    # Remove extreme unphysical outliers
    df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 140)]

    # 3. Engineer BMI and Obesity Classification (Stage 1 Feature)
    df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
    
    # Filter extreme unphysical BMI outliers
    df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]

    # Map BMI to Obesity Categories (0: Underweight, 1: Normal, 2: Overweight, 3: Obese Class I, 4: Obese Class II/III)
    conditions = [
        (df['bmi'] < 18.5),
        (df['bmi'] >= 18.5) & (df['bmi'] < 25.0),
        (df['bmi'] >= 25.0) & (df['bmi'] < 30.0),
        (df['bmi'] >= 30.0) & (df['bmi'] < 35.0),
        (df['bmi'] >= 35.0)
    ]
    choices = [0, 1, 2, 3, 4]
    df['obesity_class'] = np.select(conditions, choices, default=1)

    # Save cleaned file
    processed_path = "data/processed/cleaned_cardio_dataset.csv"
    df.to_csv(processed_path, index=False)
    print(f"Dataset cleaned successfully! Shape: {df.shape}")
    print(f"Saved to: {processed_path}")

    return df

def prepare_splits(df, target_col='cardio', test_size=0.2, random_state=42):
    """Splits cleaned dataframe into stratified train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    cleaned_df = load_and_clean_data()
    X_train, X_test, y_train, y_test = prepare_splits(cleaned_df)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")