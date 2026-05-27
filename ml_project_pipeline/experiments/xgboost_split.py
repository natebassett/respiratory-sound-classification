"""XGBoost baseline for the 70/15/15 patient-level lung disease split.

This script uses only the files created by make_split.py in clean_data/.
Class-balanced sample weights are used because COPD dominates the dataset.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


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
    image = ax.imshow(matrix, cmap="Purples")

    ax.set_title(title)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    # The raw count inside each square makes the plot easy to read in the report.
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


def make_model(args, n_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=-1,
    )


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

    output_path = output_dir / f"xgboost_{safe_filename(title)}_confusion_matrix.png"
    save_confusion_matrix_plot(
        matrix=matrix,
        label_names=label_names,
        title=f"XGBoost {title} confusion matrix",
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
        description="Train XGBoost on the make_split.py train/val/test data."
    )
    parser.add_argument("--data_dir", default="clean_data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--reg_lambda", type=float, default=2.0)
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

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = make_model(args, n_classes=len(label_ids))
    model.fit(X_train, y_train, sample_weight=sample_weights)

    evaluate(model, X_val, y_val, label_ids, label_names, "Validation results", output_dir)
    evaluate(model, X_test, y_test, label_ids, label_names, "Test results", output_dir)

    save_feature_importance(
        model=model,
        feature_names=feature_names,
        output_path=output_dir / "xgboost_feature_importance.json",
        top_n=args.top_features,
    )


if __name__ == "__main__":
    main()
