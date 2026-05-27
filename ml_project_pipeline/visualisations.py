# visualisations.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


FIG_DIR = "results/figures"
CNN_FIG_DIR = "results/figures/cnn"
CLASSICAL_FIG_DIR = "results/figures/classical"
COMPARISON_FIG_DIR = "results/figures/comparisons"
CM_DIR = "results/figures/confusion_matrices"

for folder in [
    FIG_DIR,
    CNN_FIG_DIR,
    CLASSICAL_FIG_DIR,
    COMPARISON_FIG_DIR,
    CM_DIR
]:
    os.makedirs(folder, exist_ok=True)


def save_figure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# Plot class balance
def plot_class_distribution(
    labels,
    class_names,
    title="Class Distribution",
    output_name="class_distribution.png"
):
    counts = np.bincount(labels, minlength=len(class_names))

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_figure(os.path.join(FIG_DIR, output_name))


# Create simplified CSV for model comparison
def create_combined_results_csv(
    classical_results_csv="results/model_results.csv",
    cnn_results_csv="results/cnn_results.csv",
    output_csv="results/combined_model_results.csv"
):
    classical_df = pd.read_csv(classical_results_csv)
    cnn_df = pd.read_csv(cnn_results_csv)

    classical_required = [
        "Model",
        "Test Accuracy",
        "Test Macro F1",
        "Test Weighted F1"
    ]

    cnn_required = [
        "Model",
        "Test Accuracy",
        "Test Macro F1",
        "Test Weighted F1"
    ]

    for column in classical_required:
        if column not in classical_df.columns:
            raise ValueError(f"Missing column in {classical_results_csv}: {column}")

    for column in cnn_required:
        if column not in cnn_df.columns:
            raise ValueError(f"Missing column in {cnn_results_csv}: {column}")

    classical_combined = pd.DataFrame({
        "model": classical_df["Model"],
        "accuracy": classical_df["Test Accuracy"],
        "macro_f1": classical_df["Test Macro F1"],
        "weighted_f1": classical_df["Test Weighted F1"],
    })

    cnn_combined = pd.DataFrame({
        "model": cnn_df["Model"],
        "accuracy": cnn_df["Test Accuracy"],
        "macro_f1": cnn_df["Test Macro F1"],
        "weighted_f1": cnn_df["Test Weighted F1"],
    })

    combined_df = pd.concat(
        [classical_combined, cnn_combined],
        ignore_index=True
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    return combined_df


# Plot model metrics from a simplified CSV
def plot_model_comparison(
    results_csv="results/combined_model_results.csv",
    output_name="model_comparison.png"
):
    df = pd.read_csv(results_csv)

    required = ["model", "accuracy", "macro_f1", "weighted_f1"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {results_csv}: {missing}")

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    df_plot = df.set_index("model")[metrics]

    ax = df_plot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_figure(os.path.join(COMPARISON_FIG_DIR, output_name))


# Plot classical ML vs CNN comparison
def plot_classical_vs_cnn(
    combined_results_csv="results/combined_model_results.csv",
    output_name="classical_vs_cnn_comparison.png"
):
    plot_model_comparison(
        results_csv=combined_results_csv,
        output_name=output_name
    )


# Save confusion matrix
def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    model_name,
    output_folder=CM_DIR
):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_folder, f"confusion_matrix_{safe_name}.png")

    save_figure(path)


# Plot CNN training curves
def plot_training_history(history, output_name="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CNN Training Curves")
    plt.tight_layout()

    save_figure(os.path.join(CNN_FIG_DIR, output_name))


# Plot Random Forest feature importance
def plot_feature_importance(
    model,
    feature_names,
    top_n=20,
    output_name="feature_importance.png"
):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not have feature_importances_.")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    save_figure(os.path.join(CLASSICAL_FIG_DIR, output_name))


# Plot PCA projection
def plot_pca_projection(
    X,
    y,
    class_names,
    output_name="pca_projection.png",
    max_samples=3000
):
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=class_name,
            alpha=0.6,
            s=18
        )

    plt.title("PCA Projection of Extracted Audio Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()

    save_figure(os.path.join(CLASSICAL_FIG_DIR, output_name))


# Plot t-SNE projection
def plot_tsne_projection(
    X,
    y,
    class_names,
    output_name="tsne_projection.png",
    max_samples=1500
):
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for class_id, class_name in enumerate(class_names):
        mask = y == class_id
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            label=class_name,
            alpha=0.6,
            s=18
        )

    plt.title("t-SNE Projection of Extracted Audio Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()

    save_figure(os.path.join(CLASSICAL_FIG_DIR, output_name))


# Plot example CNN spectrograms
def plot_spectrogram_examples(
    X,
    y,
    class_names,
    output_name="spectrogram_examples.png",
    examples_per_class=1
):
    X = np.asarray(X)
    y = np.asarray(y)

    selected = []

    for class_id in range(len(class_names)):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) > 0:
            selected.extend(class_indices[:examples_per_class])

    if len(selected) == 0:
        print("No spectrogram examples found.")
        return

    n = len(selected)
    plt.figure(figsize=(4 * n, 4))

    for i, idx in enumerate(selected):
        image = X[idx].squeeze()

        plt.subplot(1, n, i + 1)
        plt.imshow(image, aspect="auto", origin="lower")
        plt.title(class_names[y[idx]])
        plt.axis("off")

    plt.suptitle("Example Mel Spectrograms by Class")
    plt.tight_layout()

    save_figure(os.path.join(CNN_FIG_DIR, output_name))


# Plot misclassified CNN spectrograms
def plot_misclassified_spectrograms(
    X,
    y_true,
    y_pred,
    class_names,
    output_name="misclassified_examples.png",
    max_examples=6
):
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    wrong_indices = np.where(y_true != y_pred)[0]

    if len(wrong_indices) == 0:
        print("No misclassified examples found.")
        return

    wrong_indices = wrong_indices[:max_examples]

    plt.figure(figsize=(4 * len(wrong_indices), 4))

    for i, idx in enumerate(wrong_indices):
        image = X[idx].squeeze()

        plt.subplot(1, len(wrong_indices), i + 1)
        plt.imshow(image, aspect="auto", origin="lower")
        plt.title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}"
        )
        plt.axis("off")

    plt.suptitle("Misclassified CNN Spectrogram Examples")
    plt.tight_layout()

    save_figure(os.path.join(CNN_FIG_DIR, output_name))