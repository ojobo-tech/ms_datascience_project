import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd


def tune_xgboost(X_train, y_train, cv=3, n_iter=30, random_state=42):
    """
    Performs randomized hyperparameter tuning for XGBoost Classifier.

    Parameters:
    - X_train, y_train: Training data
    - cv: Cross-validation folds (default=3)
    - n_iter: Number of random combinations to try
    - random_state: Seed for reproducibility

    Returns:
    - Best estimator found
    - Results dataframe sorted by mean test score
    """
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'scale_pos_weight': [1, 2, 5, 10]  # useful for imbalance
    }

    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=random_state)

    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_iter=n_iter,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(X_train, y_train)

    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by='mean_test_score', ascending=False)

    return search.best_estimator_, results_df
