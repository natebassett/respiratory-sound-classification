# classical_models/logistic_regression.py

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=2000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__penalty": ["l2", "elasticnet"],
        "classifier__alpha": [0.0001, 0.001, 0.01],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"\nLogistic Regression best params: {grid_search.best_params_}")
    print(f"Logistic Regression best CV F1: {grid_search.best_score_:.4f}")

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    return {
        "Model": "Logistic Regression",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "best_params": str(grid_search.best_params_),
    }