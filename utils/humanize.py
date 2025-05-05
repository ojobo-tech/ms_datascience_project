# Dictionary for converting technical names to readable labels
FEATURE_NAME_MAPPING = {
    "age": "Age (standardized)",
    "age_years": "Age (years)",
    "gender": "Gender",
    "height": "Height (cm)",
    "weight_lb": "Weight (pounds)",
    "bmi": "Body Mass Index (BMI)",
    "ap_hi": "Systolic Blood Pressure (Upper Number)",
    "ap_lo": "Diastolic Blood Pressure (Lower Number)",
    "cholesterol": "Cholesterol Level",
    "gluc": "Blood Glucose Level",
    "smoke": "Smokes Cigarettes",
    "alco": "Drinks Alcohol",
    "active": "Physically Active",
    "cardio": "Cardiovascular Disease (CVD) Diagnosis"
}

def get_readable_feature_names(features):
    """
    Converts features to readable labels.
    """
    return [FEATURE_NAME_MAPPING.get(feat, feat) for feat in features]