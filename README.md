# Personalized Treatment Recommendation System for Cardiovascular Disease

## Project Description

This project aims to develop a **Personalized Treatment Recommendation System** for cardiovascular disease (CVD) using machine learning and explainable AI. The system analyzes patient data (e.g., demographics, vital signs, lifestyle factors) to predict CVD risk and recommend tailored treatments. The goal is to improve patient outcomes and reduce healthcare costs by providing data-driven, personalized care.

## Data Sources

* **Kaggle Cardiovascular Disease Dataset**: 70,000 rows of patient data, including demographics, vital signs, and lifestyle factors.

  * Source: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
* **MIMIC-III**: A large dataset containing de-identified health data for over 40,000 patients in critical care units.

  * Source: [MIMIC-III Website](https://mimic.physionet.org/)

## Tools and Technologies

* **Programming Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, Streamlit
* **Techniques**: Machine learning, model explainability (SHAP), risk-based recommendation system

# Introduction

Cardiovascular diseases (CVDs) are the leading cause of mortality worldwide, responsible for an estimated 17.9 million deaths annually (World Health Organization, 2023). Early detection and intervention can significantly reduce death rates. Traditional risk assessment models often rely on manual evaluation by health professionals, which can be time-consuming and prone to variability.

Recent advancements in Artificial Intelligence (AI) and Machine Learning (ML) have enabled predictive models that automate and enhance CVD risk assessment. These models utilize large datasets of patient health information — including age, blood pressure, cholesterol levels, and lifestyle factors — to identify high-risk individuals with greater accuracy and efficiency.

Traditional detection methods such as the WHO CVD Risk Charts and clinical heuristics do not account for complex interactions between variables and are less personalized. Manual evaluations can vary across practitioners and are resource-intensive when deployed at scale.

The goal of this project is to train and evaluate machine learning models to predict CVD risk using publicly available patient data. By comparing a range of algorithms including Logistic Regression, Decision Trees, Random Forests, XGBoost, and LightGBM, we aim to select a highly accurate and interpretable model for deployment in real-world healthcare settings.

## Objectives

The key objectives of this study are:

* Develop AI-driven models capable of predicting cardiovascular disease risk using patient health data.
* Compare multiple machine learning algorithms using accuracy, sensitivity, specificity, and AUC-ROC metrics.
* Use SHAP (SHapley Additive Explanations) to interpret model predictions for clinical transparency.
* Integrate model predictions into a web-based tool using Streamlit for ease of use in clinical settings.

## Scope of the Project

* **Dataset**: Structured tabular dataset with \~68,000 patient records and features including age, height, weight, systolic and diastolic blood pressure, cholesterol level, glucose level, smoking, alcohol intake, physical activity, and BMI.
* **Machine Learning Models**:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost
  * LightGBM
  * MLP Neural Network (optional)
* **Evaluation Metrics**: Accuracy, ROC-AUC, Sensitivity, Specificity, Precision, and F1-score.
* **Explainability**: SHAP values to visualize and understand feature contributions.
* **Deployment**: Streamlit app with embedded SHAP plots and recommendation logic.

## Expected Outcome

* Identification of the best performing AI model for CVD prediction.
* Deployment of a SHAP-powered model explainability interface.
* Personalized lifestyle and treatment recommendations based on top risk features.
* A user-friendly screening tool to aid clinicians in early intervention and treatment planning.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ojobo-tech/cvd-risk-recommender.git
cd cvd-risk-recommender

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── data/                         
├── utils/
│   ├── preprocessing.py         
│   ├── recommendations.py        
├── models/
│   └── train_models.py           
├── saved_models/               
├── streamlit_app.py            
├── run_all.ipynb              
├── README.md
```

## Running the Project

You can run the full project pipeline inside Jupyter Notebook:

1. Open `run_code.ipynb`.
2. Step through each section to:

   * Load and preprocess data
   * Train and evaluate models
   * Generate SHAP explanations
   * Launch the Streamlit app

To launch the web app locally:

```bash
python -m streamlit run streamlit_app.py
```

## Explainability with SHAP

We use SHAP (SHapley Additive Explanations) to understand the model's output:

* **SHAP Summary Plot**: Displays global feature importance.
* **SHAP Waterfall Plot**: Provides local interpretability for individual predictions.
* These are shown inside the Streamlit interface for every prediction made.

## SHAP-Driven Recommendation System

For each top contributing SHAP feature, the app:

* Shows **why** the feature contributes to CVD risk.
* Offers **lifestyle** and **treatment** recommendations.
* Gives safe **target ranges** and encouragement if within healthy bounds.


## References

* WHO Cardiovascular Disease Factsheet: [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-%28cvds%29)
* SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
* Kaggle Dataset: [https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## License

This project is licensed under the MIT License.

---

**Note:** This README is structured for direct inclusion in a GitHub repository. All sections are Markdown-formatted and Jupyter/Streamlit ready.
