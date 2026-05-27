# classical_models/random_forest.py

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    # Quick OOB sweep to choose a sensible number of trees
    oob_errors = []
    estimator_range = [100, 200, 300]

    with tqdm(estimator_range, desc="OOB sweep", unit="model") as pbar:
        for n in pbar:
            rf = RandomForestClassifier(
                n_estimators=n,
                class_weight="balanced",
                oob_score=True,
                random_state=42,
                n_jobs=-1,
            )

            rf.fit(X_train, y_train)

            oob_error = 1 - rf.oob_score_
            oob_errors.append(oob_error)

            pbar.set_postfix({"OOB error": f"{oob_error:.4f}"})

    best_n = estimator_range[int(np.argmin(oob_errors))]

    print(f"\nLowest OOB error at n_estimators={best_n}\n")

    # Small randomised search for practical runtime
    param_dist = {
        "n_estimators": [best_n, min(best_n + 50, 300)],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    n_iter = 8

    param_list = list(
        ParameterSampler(
            param_dist,
            n_iter=n_iter,
            random_state=42,
        )
    )

    best_score = -1
    best_params = None

    with tqdm(param_list, desc="RF hyperparameter search", unit="combo") as pbar:
        for params in pbar:
            rf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            score = cross_val_score(
                rf,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            ).mean()

            if score > best_score:
                best_score = score
                best_params = params

            pbar.set_postfix({
                "best CV F1": f"{best_score:.4f}",
                "current CV F1": f"{score:.4f}",
            })

    print(f"\nRandom Forest best params: {best_params}")
    print(f"Random Forest best CV F1: {best_score:.4f}")

    # Train final model once on the full training set
    best_model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    best_model.fit(X_train, y_train)

    # Feature importance summary
    importances = best_model.feature_importances_
    top_n = min(15, len(importances))
    top_indices = np.argsort(importances)[::-1][:top_n]

    print(f"\nTop {top_n} feature importances:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:>2d}. Feature {idx:>4d}: {importances[idx]:.4f}")

    # Evaluate on validation and test sets
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    return {
        "Model": "Random Forest",
        "Validation Accuracy": accuracy_score(y_val, val_pred),
        "Test Accuracy": accuracy_score(y_test, test_pred),
        "Validation Macro F1": f1_score(y_val, val_pred, average="macro", zero_division=0),
        "Test Macro F1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "Validation Weighted F1": f1_score(y_val, val_pred, average="weighted", zero_division=0),
        "Test Weighted F1": f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "best_params": str(best_params),
    }