"""SVM baseline for the 70/15/15 patient-level lung disease split.

This script uses only the files created by make_split.py in clean_data/.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


def safe_filename(text):
    return text.lower().replace(" ", "_")


def load_label_map(path):
    with open(path, "r") as f:
        raw_map = json.load(f)
    return {int(label_id): label_name for label_id, label_name in raw_map.items()}


def load_split(data_dir, split_name):
    X = np.load(data_dir / f"X_{split_name}.npy")
    y = np.load(data_dir / f"y_{split_name}_disease.npy")
    return X, y


def save_confusion_matrix_plot(matrix, label_names, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    # Put the raw count in each square so the graph matches the printed matrix.
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


def make_model(max_iter, random_state):
    # The scaler is fitted on training data only through the pipeline.
    # Hinge loss gives a linear SVM-style classifier, trained efficiently with SGD.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=0.0001,
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=1e-3,
                    random_state=random_state,
                ),
            ),
        ]
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

    print("Confusion matrix")
    print("Rows are true labels, columns are predicted labels.")
    print("Label order:", label_names)
    matrix = confusion_matrix(y, predictions, labels=label_ids)
    print(matrix)

    output_path = output_dir / f"{safe_filename(title)}_confusion_matrix.png"
    save_confusion_matrix_plot(
        matrix=matrix,
        label_names=label_names,
        title=f"{title} confusion matrix",
        output_path=output_path,
    )
    print(f"Saved graph: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear SVM-style classifier on the make_split.py data."
    )
    parser.add_argument("--data_dir", default="clean_data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(data_dir / "disease_label_map.json")
    label_ids = sorted(label_map)
    label_names = [label_map[label_id] for label_id in label_ids]

    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    model = make_model(
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, label_ids, label_names, "Validation results", output_dir)
    evaluate(model, X_test, y_test, label_ids, label_names, "Test results", output_dir)


if __name__ == "__main__":
    main()
