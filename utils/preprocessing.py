import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Human-readable feature name mapping
FEATURE_NAME_MAP = {
    "age_years": "Age (Years)",
    "height": "Height (cm)",
    "weight_lb": "Weight (lbs)",
    "bmi": "Body Mass Index (BMI)",
    "ap_hi": "Systolic Blood Pressure",
    "ap_lo": "Diastolic Blood Pressure",
    "gender": "Gender",
    "cholesterol": "Cholesterol Level",
    "gluc": "Glucose Level",
    "smoke": "Smoking Status",
    "alco": "Alcohol Intake",
    "active": "Physical Activity",
    "hypertension_encoded": "Hypertension Stage",
    "cardio": "CVD Risk"
}

def rename_columns_for_eda(df):
    """
    Renames DataFrame columns using FEATURE_NAME_MAP for improved readability in EDA and visualization.

    Parameters:
        df (pd.DataFrame): Input DataFrame with original feature names.

    Returns:
        pd.DataFrame: DataFrame with human-readable column names.
    """
    return df.rename(columns=FEATURE_NAME_MAP)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and filters the dataset by removing outliers and converting age.

    Operations:
    - Converts age from days to years.
    - Filters out unrealistic systolic/diastolic blood pressure values.
    - Removes outliers in height and weight.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    df['age_years'] = (df['age'] / 365).astype(int)
    df.drop(columns=['age'], inplace=True)
    df = df[df['ap_hi'] >= df['ap_lo']]
    df = df[(df['ap_hi'] > 60) & (df['ap_hi'] < 200)]
    df = df[(df['ap_lo'] > 40) & (df['ap_lo'] < 150)]
    df = df[(df['height'] >= 100) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 30) & (df['weight'] <= 180)]
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features: BMI and weight in pounds, and drops original weight.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    df = df.copy()
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['weight_lb'] = df['weight'] * 2.20462
    df.drop(columns=['weight'], inplace=True)
    return df

def split_features_labels(df: pd.DataFrame, target_column='cardio'):
    """
    Splits the dataset into features (X) and labels (y).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and label vector y.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def load_and_preprocess_data(csv_path: str):
    """
    Loads the dataset and performs preprocessing for both EDA and modeling.

    Steps:
    - Drops unnecessary columns.
    - Prepares EDA copy with readable labels.
    - Cleans and engineers features for modeling.
    - Encodes categorical values if needed.
    - Splits and scales the data for training/testing.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        Scaled training features, scaled test features, training labels,
        test labels, and the EDA DataFrame with readable columns.
    """
    df = pd.read_csv(csv_path)

    drop_columns = ['bp_category', 'bp_category_encoded']
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    df_eda = df.copy()
    df_eda = clean_data(df_eda)
    df_eda = add_features(df_eda)
    df.drop(columns=['id', 'gender'], inplace=True, errors='ignore')
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
    df_eda['cardio'] = df_eda['cardio'].map({0: 'No CVD Risk', 1: 'CVD Risk'})

    df_eda = rename_columns_for_eda(df_eda)

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

    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, df_eda
