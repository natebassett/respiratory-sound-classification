"""Random Forest baseline for the 70/15/15 patient-level lung disease split.

This script uses only the files created by make_split.py in clean_data/.
It includes class weighting because the disease labels are heavily imbalanced.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid


def safe_filename(text):
    return text.lower().replace(" ", "_")


def load_label_map(path):
    with open(path, "r") as f:
        raw_map = json.load(f)
    return {int(label_id): label_name for label_id, label_name in raw_map.items()}


def load_feature_names(path):
    with open(path, "r") as f:
        return json.load(f)


def load_split(data_dir, split_name):
    X = np.load(data_dir / f"X_{split_name}.npy")
    y = np.load(data_dir / f"y_{split_name}_disease.npy")
    return X, y


def save_confusion_matrix_plot(matrix, label_names, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Greens")

    ax.set_title(title)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    # The counts are printed inside the cells so the image is useful in the report.
    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            colour = "white" if matrix[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, label="Number of samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_model(args, params=None):
    params = params or {}
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", args.n_estimators),
        max_depth=params.get("max_depth", args.max_depth),
        min_samples_leaf=params.get("min_samples_leaf", args.min_samples_leaf),
        max_features=params.get("max_features", args.max_features),
        class_weight=params.get("class_weight", args.class_weight),
        n_jobs=-1,
        random_state=args.random_state,
    )


def score_predictions(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }


def tune_hyperparameters(args, X_train, y_train, X_val, y_val, output_dir):
    # Keep the grid small enough to run on a laptop, but varied enough to test
    # depth, leaf size, and imbalance handling.
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt"],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    results = []
    best_model = None
    best_result = None

    print("\nHyperparameter tuning")
    print("---------------------")

    for run_number, params in enumerate(ParameterGrid(param_grid), start=1):
        model = make_model(args, params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        metrics = score_predictions(y_val, predictions)

        result = {
            "run": run_number,
            "params": params,
            **metrics,
        }
        results.append(result)

        print(
            f"Run {run_number:02d}: macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, params={params}"
        )

        # Macro F1 is the main target because the dataset is imbalanced.
        if best_result is None or (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
        ) > (
            best_result["macro_f1"],
            best_result["balanced_accuracy"],
        ):
            best_result = result
            best_model = model

    output_path = output_dir / "random_forest_tuning_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest validation setting")
    print("-----------------------")
    print(f"Macro F1:          {best_result['macro_f1']:.4f}")
    print(f"Balanced accuracy: {best_result['balanced_accuracy']:.4f}")
    print(f"Params:            {best_result['params']}")
    print(f"Saved tuning log:  {output_path}")

    return best_model, best_result


def evaluate(model, X, y, label_ids, label_names, title, output_dir):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    balanced_accuracy = balanced_accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, predictions, average="weighted", zero_division=0)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print("\nClassification report")
    print(
        classification_report(
            y,
            predictions,
            labels=label_ids,
            target_names=label_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y, predictions, labels=label_ids)
    print("Confusion matrix")
    print("Rows are true labels, columns are predicted labels.")
    print("Label order:", label_names)
    print(matrix)

    output_path = output_dir / f"random_forest_{safe_filename(title)}_confusion_matrix.png"
    save_confusion_matrix_plot(
        matrix=matrix,
        label_names=label_names,
        title=f"Random Forest {title} confusion matrix",
        output_path=output_path,
    )
    print(f"Saved graph: {output_path}")


def save_feature_importance(model, feature_names, output_path, top_n):
    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    rows = []
    for rank, feature_index in enumerate(ranked_indices[:top_n], start=1):
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[feature_index],
                "importance": float(importances[feature_index]),
            }
        )

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nTop {top_n} feature importances")
    print("---------------------------")
    for row in rows:
        print(f"{row['rank']:>2}. {row['feature']}: {row['importance']:.6f}")
    print(f"Saved feature importance: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest on the make_split.py train/val/test data."
    )
    parser.add_argument("--data_dir", default="clean_data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_features", default="sqrt")
    parser.add_argument("--class_weight", default="balanced_subsample")
    parser.add_argument("--top_features", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(data_dir / "disease_label_map.json")
    label_ids = sorted(label_map)
    label_names = [label_map[label_id] for label_id in label_ids]
    feature_names = load_feature_names(data_dir / "feature_columns.json")

    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    if args.no_tune:
        model = make_model(args)
        model.fit(X_train, y_train)
    else:
        model, _ = tune_hyperparameters(
            args=args,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_dir=output_dir,
        )

    evaluate(model, X_val, y_val, label_ids, label_names, "Validation results", output_dir)
    evaluate(model, X_test, y_test, label_ids, label_names, "Test results", output_dir)

    save_feature_importance(
        model=model,
        feature_names=feature_names,
        output_path=output_dir / "random_forest_feature_importance.json",
        top_n=args.top_features,
    )


if __name__ == "__main__":
    main()
