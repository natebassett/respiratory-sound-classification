# classical_models/mlp_classifier.py

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


def run_mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.0005, 0.0001],
        "batch_size": [64, 128],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = RandomizedSearchCV(
        estimator=MLPClassifier(
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        ),
        param_distributions=param_dist,
        n_iter=8,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print(f"\nMLP best params: {search.best_params_}")
    print(f"MLP best CV F1: {search.best_score_:.4f}")

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    return {
        "Model": "MLPClassifier",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "best_params": str(search.best_params_),
    }