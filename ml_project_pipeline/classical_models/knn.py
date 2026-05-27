from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


def run_knn(X_train, y_train, X_val, y_val, X_test, y_test):
    # tried different values of k, distance metrics and weighting schemes
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "metric": ["euclidean", "manhattan"],
        "weights": ["uniform", "distance"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    print(f"KNN best params: {grid_search.best_params_}")
    print(f"KNN best CV F1: {grid_search.best_score_:.4f}")

    
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred))

    return {
        "Model": "KNN",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro"),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro"),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted"),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted"),
        "best_params": str(grid_search.best_params_),
    }
