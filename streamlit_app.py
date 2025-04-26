import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from utils.recommendations import RECOMMENDATION_MAP, FEATURE_NAME_MAP

# Load the trained model and SHAP explainer
model = joblib.load("saved_models/xgboost.pkl")
explainer = shap.Explainer(model)

# Define input UI
st.set_page_config(page_title="CVD Risk Predictor", layout="wide")
st.title("🫀 Cardiovascular Risk Predictor with Explainability")

st.sidebar.header("🔎 Enter Patient Details")

input_data = {
    "age_years": st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=50),
    "height": st.sidebar.number_input("Height (cm)", min_value=120, max_value=240, value=170),
    "weight_lb": st.sidebar.number_input("Weight (lbs)", min_value=70, max_value=500, value=154),
    "ap_hi": st.sidebar.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120),
    "ap_lo": st.sidebar.number_input("Diastolic Blood Pressure", min_value=50, max_value=150, value=80),
    "cholesterol": st.sidebar.selectbox("Cholesterol Level", [1, 2, 3], format_func=lambda x: ["Normal (<200 mg/dL)", "Elevated (200-239 mg/dL)", "Critical (>240 mg/dL)"][x - 1]),
    "gluc": st.sidebar.selectbox("Glucose Level", [1, 2, 3], format_func=lambda x: ["Normal (<100 mg/dL)", "Elevated (100-125 mg/dL)", "Critical (>126 mg/dL)"][x - 1]),
    "smoke": st.sidebar.radio("Do you smoke?", [0, 1], format_func=lambda x: "Yes" if x else "No"),
    "alco": st.sidebar.radio("Consume alcohol?", [0, 1], format_func=lambda x: "Yes" if x else "No"),
    "active": st.sidebar.radio("Regular Physical activity?", [0, 1], format_func=lambda x: "Yes" if x else "No"),
    "bmi": 0.0  # Placeholder to compute
}

# Convert weight to kg for BMI calc
weight_kg = input_data["weight_lb"] / 2.20462
input_data["bmi"] = weight_kg / ((input_data["height"] / 100) ** 2)

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🔍 Predict Risk"):
    # Keep only the columns the model was trained on
    expected_features = ['height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 
                  'alco', 'active', 'age_years', 'bmi', 'weight_lb']
    input_df = input_df[expected_features]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🩺 Prediction Result")
    st.write("**CVD Risk Prediction:**", "🔴 At Risk" if prediction == 1 else "🟢 Not at Risk")
    st.write("**Probability of CVD:**", f"{probability:.2%}")

    # SHAP Explanation
    st.subheader("📊 SHAP Explanation")
    shap_values = explainer(input_df)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')

    # Top feature contributions
    st.subheader("🧠 Risk Factor Breakdown")
    shap_dict = dict(zip(input_df.columns, shap_values[0].values))
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    for feature, value in sorted_features[:3]:
        name = FEATURE_NAME_MAP.get(feature, feature)
        rec = RECOMMENDATION_MAP.get(name, {})

        st.markdown(f"### 🔹 {name}")
        if value > 0:
            st.markdown(f"**Risk Contribution:** {rec.get('risk_reason', 'N/A')}")
            st.markdown(f"**Treatment Advice:** {rec.get('treatment_advice', 'N/A')}")
            st.markdown(f"**Lifestyle Guidance:** {rec.get('lifestyle_guidance', 'N/A')}")
            st.markdown(f"**Target Range:** {rec.get('target', 'N/A')}")
        else:
            st.markdown(f"✅ {rec.get('no_risk_recommendation', 'Looks good!')}")
