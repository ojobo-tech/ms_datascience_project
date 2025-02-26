# Personalized Treatment Recommendation System for Cardiovascular Disease

## Project Description
This project aims to develop a **Personalized Treatment Recommendation System** for cardiovascular disease (CVD) using machine learning and explainable AI. The system will analyze patient data (e.g., demographics, vital signs, lifestyle factors) to predict CVD risk and recommend tailored treatments. The goal is to improve patient outcomes and reduce healthcare costs by providing data-driven, personalized care.

## Data Sources
- **Kaggle Cardiovascular Disease Dataset**: 70,000 rows of patient data, including demographics, vital signs, and lifestyle factors.
  - Source: [Kaggle](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset))
- **MIMIC-III**: A large dataset containing de-identified health data for over 40,000 patients in critical care units.
  - Source: [MIMIC-III Website](https://mimic.physionet.org/)

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Streamlit
- **Techniques**: Machine learning, collaborative filtering, reinforcement learning, explainable AI (SHAP)

# Introduction

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for an estimated 17.9 million deaths annually (World Health Organization, 2023). Early detection and intervention can significantly reduce death rates . Traditional risk assessment models often rely on manual evaluation by health professionals, which can be time-consuming and prone to the professionals discretion.
There has been recent advancements in Artificial Intelligence (AI) and Machine Learning (ML), predictive models can now automate and enhance CVD risk assessment. By leveraging large datasets of patient health information like age, blood pressure, cholesterol levels, lifestyle factors, and more, models can identify high risk individuals with greater accuracy and efficiency.

Traditional cardiovascular disease detection methods rely on the healthcare professional and rule-based scoring systems like WHO CVD Risk Charts. These approaches, while clinically useful, have limitations like a lack of personalization, they do not account for complex interactions between multiple risk factors. Also, manual evaluation may vary among healthcare professionals and screening large populations is resource-intensive.

The goal of this project is to train and evaluate machine learning models to predict CVD risk using secondary patient data. By comparing various AI models including Logistic Regression, Decision Trees, Random Forests, XGBoost, LightGBM, and Neural Networks (MLP), we can determine the most effective approach for real-world deployment.

## Objectives
The key objectives of this study are:
To develop AI-driven models capable of predicting cardiovascular disease risk using patient health data.
To compare multiple machine learning algorithms based on accuracy, sensitivity, specificity, and AUC-ROC scores.
To ensure model explainability using SHAP (Explainable AI) to make predictions interpretable for healthcare professionals.
To explore model deployment strategies for real-time CVD risk assessment in clinical settings.

## Scope of the Project
  Dataset: A structured tabular dataset containing 68,000 patient records with features such as age, blood pressure (ap_hi, ap_lo), cholesterol levels, BMI, smoking status, and physical activity.
  Machine Learning Models: Six algorithms will be tested:
    Baseline Model: Logistic Regression
    Tree-Based Models: Decision Tree, Random Forest
    Boosting Models: XGBoost, LightGBM (Best for tabular data)
    Deep Learning Model: Multi-Layer Perceptron (MLP) Neural Network
  Evaluation Metrics: Accuracy, AUC-ROC, Sensitivity, Specificity, Precision, F1-score.
  Explainability: SHAP (SHapley Additive Explanations) for AI interpretability.
  Deployment Strategy: Model integration into a Stream-Lit for real-world applications.

## Expected Outcome
By the end of this project, I expect to:
  Identify the best-performing AI model for CVD prediction.
  Provide interpretable AI insights to support clinical decision-making.
  Develop a scalable AI-based screening tool for early CVD risk detection.
