from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


def run_rbf_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    # RBF kernel SVM can learn nonlinear decision boundaries.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    # Tune gamma and C as shown in the labs.
    param_grid = {
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01, 0.1, 1]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"RBF SVM best params: {grid_search.best_params_}")
    print(f"RBF SVM best CV F1: {grid_search.best_score_:.4f}")

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    return {
        "Model": "RBF SVM",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "best_params": str(grid_search.best_params_),
    }