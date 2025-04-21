"""
utils/preprocessing.py

Handles data cleaning, feature engineering, and preparation
for machine learning model input.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Converting age from days to years
    - Ensuring systolic >= diastolic BP. 
    Systolic BP (ap_hi) is how hard your heart is working. Diastolic (ap_lo) is how relaxed your blood vessels are at rest.
    Medically, systolic BP should always be greater than or equal to diastolic BP.
    - Removing unrealistic values for height, weight, and blood pressure.
    """
    df = df.copy()
    df['age_years'] = (df['age'] / 365).astype(int)
    df = df[df['ap_hi'] >= df['ap_lo']]
    df = df[(df['ap_hi'] > 60) & (df['ap_hi'] < 200)]
    df = df[(df['ap_lo'] > 40) & (df['ap_lo'] < 150)]
    df = df[(df['height'] >= 100) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 30) & (df['weight'] <= 180)]
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features:
    - Body Mass Index
    - Weight converted from kg to pounds
    """
    df = df.copy()
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['weight_lb'] = df['weight'] * 2.20462
    df.drop(columns=['weight'], inplace=True)
    return df


def split_features_labels(df: pd.DataFrame, target_column='cardio'):
    """
    Splits the dataset into input features X and target labels y.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def load_and_preprocess_data(csv_path: str):
    """
    Full pipeline to:
    - Load raw CSV
    - Create a labeled EDA copy (with BMI, weight_lb, readable labels)
    - Clean and engineer features for modeling
    - Encode categorical features (if needed)
    - Split and scale data for modeling
    Returns:
        X_train, X_test, y_train, y_test, df_eda
    """
    df = pd.read_csv(csv_path)

    # Drop unwanted columns if present
    drop_columns = ['bp_category', 'bp_category_encoded']
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    # ------------------ EDA VERSION ------------------
    df_eda = df.copy()
    df_eda = add_features(df_eda)

    # Add human-readable labels
    if 'gender' in df_eda.columns:
        df_eda['gender'] = df_eda['gender'].map({1: 'Male', 2: 'Female'})

    df_eda['cholesterol'] = df_eda['cholesterol'].map({
        1: 'Normal (<200 mg/dL)',
        2: 'Above Normal (200–239 mg/dL)',
        3: 'Well Above Normal (>240 mg/dL)'
    })

    df_eda['gluc'] = df_eda['gluc'].map({
        1: 'Normal (< 100 mg/dL)',
        2: 'Above Normal (100–125 mg/dL)',
        3: 'Well Above Normal (> 125 mg/dL)'
    })

    df_eda['smoke'] = df_eda['smoke'].map({0: 'No', 1: 'Yes'})
    df_eda['alco'] = df_eda['alco'].map({0: 'No', 1: 'Yes'})
    df_eda['active'] = df_eda['active'].map({0: 'No', 1: 'Yes'})

    # ------------------ MODELING VERSION ------------------
    df = clean_data(df)
    df = add_features(df)

    if 'hypertension' in df.columns:
        df['hypertension_encoded'] = df['hypertension'].map({
            'Normal': 0,
            'Prehypertension': 1,
            'Hypertension Stage 1': 2,
            'Hypertension Stage 2': 3
        })
        df.drop(columns=['hypertension'], inplace=True)

    # Split features and label
    X, y = split_features_labels(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("DEBUG: X shape before scaling:", X.shape)


    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, df_eda