from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_linear_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    # The scaler is fitted on training data only through the pipeline.
    # Hinge loss gives a linear SVM-style classifier, trained efficiently with SGD.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=0.0001,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    return {
        "Model": "Linear SVM",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
    }