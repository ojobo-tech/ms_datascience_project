import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb

import shap  # Added for SHAP explainability

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

RECOMMENDATION_MAP = {
    "Systolic Blood Pressure": {
        "risk_reason": "Elevated systolic pressure increases the strain on heart walls and arteries, which can lead to cardiovascular disease.",
        "treatment_advice": "Consult your doctor about antihypertensive medications if systolic pressure remains elevated.",
        "lifestyle_guidance": "Reduce sodium intake, avoid processed foods, increase potassium (e.g., bananas, spinach), and manage stress.",
        "target": "Aim for systolic pressure under 120 mmHg.",
        "no_risk_recommendation": "Your systolic pressure is within a healthy range. Keep monitoring it regularly and continue your heart-healthy habits."
    },
    "Diastolic Blood Pressure": {
        "risk_reason": "High diastolic pressure can damage blood vessels and increase heart disease risk.",
        "treatment_advice": "Discuss ACE inhibitors with a healthcare provider.",
        "lifestyle_guidance": "Engage in physical activity, limit alcohol, and follow a DASH diet.",
        "target": "Keep diastolic pressure below 80 mmHg.",
        "no_risk_recommendation": "Your diastolic pressure is in a healthy range. Maintain a balanced diet and active lifestyle."
    },
    "Cholesterol Level": {
        "risk_reason": "High cholesterol can lead to plaque buildup in arteries, increasing heart attack risk.",
        "treatment_advice": "Statins could be prescribed to lower cholesterol.",
        "lifestyle_guidance": "Limit saturated fats, eat more fiber, and consider omega-3 supplements.",
        "target": "Maintain total cholesterol below 200 mg/dL.",
        "no_risk_recommendation": "Great job keeping your cholesterol normal. Stay consistent with healthy eating."
    },
    "Glucose Level": {
        "risk_reason": "High glucose levels are lead to diabetes, leading to CVD risk.",
        "treatment_advice": "Medication or insulin therapy might be required for elevated glucose.",
        "lifestyle_guidance": "Cut down on sugar, increase physical activity, and monitor carbs.",
        "target": "Keep fasting glucose below 100 mg/dL.",
        "no_risk_recommendation": "Your glucose levels are normal. Maintain a balanced diet to keep it stable."
    },
    "Smoking Status": {
        "risk_reason": "Smoking damages blood vessels and reduces oxygen in the blood, raising heart disease risk.",
        "treatment_advice": "Seek cessation programs or nicotine replacement therapy.",
        "lifestyle_guidance": "Avoid exposure to secondhand smoke and join a support group.",
        "target": "Complete cessation is ideal.",
        "no_risk_recommendation": "Great that you're not smoking! This is excellent for your heart health."
    },
    "Alcohol Intake": {
        "risk_reason": "Excessive alcohol can elevate blood pressure.",
        "treatment_advice": "Limit alcohol to moderate levels or abstain if already high risk.",
        "lifestyle_guidance": "Choose non-alcoholic alternatives and avoid binge drinking.",
        "target": "Limit to no more than 1 drink/day (women) or 2 drinks/day (men).",
        "no_risk_recommendation": "No alcohol intake is a heart-healthy choice. Keep it up"
    },
    "Physical Activity": {
        "risk_reason": "Inactivity contributes to obesity and poor heart function.",
        "treatment_advice": "Include exercise in your treatment plan with medical guidance.",
        "lifestyle_guidance": "Start with brisk walking or moderate workouts 5 times/week.",
        "target": "Aim for 150 minutes/week of exercise.",
        "no_risk_recommendation": "Staying active is key to prevention. You're on the right path"
    },
    "Body Mass Index (BMI)": {
        "risk_reason": "A high BMI is associated with hypertension, diabetes, and CVD.",
        "treatment_advice": "Consider a personalized weight management program.",
        "lifestyle_guidance": "Reduce calorie intake, increase activity, and consult a dietitian.",
        "target": "Maintain BMI between 18.5 and 24.9.",
        "no_risk_recommendation": "Your BMI is in a healthy range. Continue monitoring your diet and activity."
    },
    "Age (Years)": {
        "risk_reason": "CVD risk increases naturally with age due to vascular wear and metabolic changes.",
        "treatment_advice": "Routine screenings and preventive medications may be appropriate.",
        "lifestyle_guidance": "Prioritize regular checkups, a balanced diet, and physical activity.",
        "target": "Discuss personalized age-based goals with your doctor.",
        "no_risk_recommendation": "You're managing your age-related risk well. Stay proactive with your health checks."
    }
}
