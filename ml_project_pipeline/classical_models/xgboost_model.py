# classical_models/xgboost_model.py

# pip install xgboost
# or
# pthon -m pip install xgboost

import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    # Handles class imbalance without changing the original data.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred))

    return {
        "Model": "XGBoost",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro"),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro"),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted"),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted")
    }