"""Logistic Regression baseline for lung disease prediction.

This script trains and evaluates Logistic Regression on the patient-safe
cross-validation folds in cv_data/.
"""

"""
To run this file: 
python experiments/logistic_regression_cv.py --data_dir cv_data --folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_label_map(path):
    with open(path, "r") as f:
        raw_map = json.load(f)
    return {int(label_id): label_name for label_id, label_name in raw_map.items()}


def load_fold(fold_dir):
    X_train = np.load(fold_dir / "X_train.npy")
    X_test = np.load(fold_dir / "X_test.npy")
    y_train = np.load(fold_dir / "y_train_disease.npy")
    y_test = np.load(fold_dir / "y_test_disease.npy")
    return X_train, X_test, y_train, y_test


def make_model(max_iter, random_state):
    # Scaling is fitted inside each fold, so the test fold stays unseen.
    # SGD with logistic loss gives the same kind of linear decision boundary as
    # Logistic Regression, but it is much quicker on these cross-validation folds.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=max_iter,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_fold(model, X_train, X_test, y_train, y_test, label_ids, label_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, y_pred, labels=label_ids)
    return metrics, report, matrix


def print_metric_summary(fold_results):
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]

    print("\nCross-validation summary")
    print("------------------------")
    for metric_name in metric_names:
        values = np.array([result[metric_name] for result in fold_results])
        print(f"{metric_name}: {values.mean():.4f} +/- {values.std():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression on patient-safe lung disease CV folds."
    )
    parser.add_argument("--data_dir", default="cv_data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    label_map = load_label_map(data_dir / "disease_label_map.json")
    label_ids = sorted(label_map)
    label_names = [label_map[label_id] for label_id in label_ids]

    fold_results = []
    total_confusion_matrix = np.zeros((len(label_ids), len(label_ids)), dtype=int)

    for fold_number in range(1, args.folds + 1):
        fold_dir = data_dir / f"fold_{fold_number}"
        X_train, X_test, y_train, y_test = load_fold(fold_dir)

        model = make_model(
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        metrics, report, matrix = evaluate_fold(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_ids=label_ids,
            label_names=label_names,
        )

        fold_results.append(metrics)
        total_confusion_matrix += matrix

        print(f"\nFold {fold_number}")
        print("------")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
        print("\nClassification report")
        print(report)

    print_metric_summary(fold_results)

    print("\nCombined confusion matrix")
    print("-------------------------")
    print("Rows are true labels, columns are predicted labels.")
    print("Label order:", label_names)
    print(total_confusion_matrix)


if __name__ == "__main__":
    main()
