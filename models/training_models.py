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
    "hypertension_encoded": "Hypertension Stage"
}

def rename_columns_for_eda(df):
    """
    Apply human-readable names to feature columns for better interpretability during EDA.
    """
    return df.rename(columns=FEATURE_NAME_MAP)

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, model_dir='saved_models'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_dir = model_dir
        self.results = []

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def evaluate_model(self, model, X, y, model_name):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

        accuracy = accuracy_score(y, y_pred) * 100
        auc = roc_auc_score(y, y_proba) * 100
        conf_matrix = confusion_matrix(y, y_pred)

        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) * 100
        specificity = tn / (tn + fp) * 100
        precision = tp / (tp + fp) * 100

        self.results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "AUC-ROC": auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision
        })

        print(f"\n**{model_name} Performance:**")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"AUC-ROC: {auc:.2f}%")
        print(f"Sensitivity (Recall): {sensitivity:.2f}%")
        print(f"Specificity: {specificity:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print("\nClassification Report:\n", classification_report(y, y_pred))
        print("Confusion Matrix:\n", conf_matrix)

    def train_and_save_model(self, model, model_name):
        print(f"\n🔹 Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        self.evaluate_model(model, self.X_train, self.y_train, f"{model_name} (Train)")
        self.evaluate_model(model, self.X_test, self.y_test, f"{model_name} (Test)")
        joblib.dump(model, os.path.join(self.model_dir, f"{model_name.replace(' ', '_').lower()}.pkl"))

    def train_all_models(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
        }

        for name, model in models.items():
            self.train_and_save_model(model, name)

        self.explain_with_shap(models["XGBoost"])
        self.save_results_summary()
        self.save_waterfall_plot(models["XGBoost"])

    def explain_with_shap(self, model):
        X_test_named = rename_columns_for_eda(self.X_test)
        explainer = shap.Explainer(model, feature_names=X_test_named.columns)
        shap_values = explainer(X_test_named)

        plt.figure()
        shap.summary_plot(shap_values, X_test_named, show=False)
        plt.tight_layout()
        shap_path = os.path.join(self.model_dir, "xgboost_shap_summary.png")
        plt.savefig(shap_path)
        plt.close()
        print(f"📈 SHAP summary saved to {shap_path}")

    def save_waterfall_plot(self, model, index=0):
        X_test_named = rename_columns_for_eda(self.X_test)
        explainer = shap.Explainer(model, feature_names=X_test_named.columns)
        shap_values = explainer(X_test_named)

        shap.plots.waterfall(shap_values[index], show=False)
        image_path = os.path.join(self.model_dir, f"xgboost_shap_waterfall_{index}.png")
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
        print(f"📸 Waterfall plot saved to {image_path}")

    def save_results_summary(self):
        df = pd.DataFrame(self.results)
        summary_path = os.path.join(self.model_dir, "model_comparison_summary.csv")
        df.to_csv(summary_path, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Accuracy", y="Model", palette="viridis")
        plt.title("Model Accuracy Comparison")
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Model")
        plt.tight_layout()
        image_path = os.path.join(self.model_dir, "model_comparison_summary.png")
        plt.savefig(image_path)
        plt.close()

        print(f"📁 File exists? {os.path.exists(image_path)}")
        print(f"📊 Summary saved to {summary_path} and {image_path}")
