import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from classical_models.random_forest import run_random_forest
from classical_models.xgboost_model import run_xgboost
from classical_models.mlp_classifier import run_mlp_classifier
from classical_models.knn import run_knn
from classical_models.decision_tree import run_decision_tree
from classical_models.logistic_regression import run_logistic_regression
from classical_models.svm_linear import run_linear_svm
from classical_models.svm_rbf import run_rbf_svm

from visualisations import (
    plot_model_comparison,
    plot_pca_projection,
    plot_tsne_projection,
    plot_class_distribution,
)

DATA_DIR = "clean_data"
RESULTS_DIR = "results"

DETAILED_RESULTS_FILE = "model_results.csv"
SUMMARY_RESULTS_FILE = "classical_results.csv"

CLASS_NAMES = [
    "Normal",
    "Crackle",
    "Wheeze",
    "Both"
]

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(data_dir):

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))

    y_train = np.load(os.path.join(data_dir, "y_train_cycle.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val_cycle.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test_cycle.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):

    # Fit scaler only on training data to avoid leakage
    scaler = StandardScaler()

    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test)
    )


def create_summary_results(results_df):
    # Create a clean CSV for visualisations only
    required_columns = [
        "Model",
        "Test Accuracy",
        "Test Macro F1",
        "Test Weighted F1",
    ]

    missing = [col for col in required_columns if col not in results_df.columns]

    if missing:
        raise ValueError(f"Missing expected result columns: {missing}")

    summary_df = pd.DataFrame({
        "model": results_df["Model"],
        "accuracy": results_df["Test Accuracy"],
        "macro_f1": results_df["Test Macro F1"],
        "weighted_f1": results_df["Test Weighted F1"],
    })

    return summary_df


def main():

    print("Loading cleaned data...")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATA_DIR)

    print(
        f"Train: {X_train.shape} | "
        f"Val: {X_val.shape} | "
        f"Test: {X_test.shape}"
    )

    # Plot dataset class balance
    plot_class_distribution(
        labels=y_train,
        class_names=CLASS_NAMES,
        title="Training Set Class Distribution",
        output_name="class_distribution.png"
    )

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train,
        X_val,
        X_test
    )

    model_runs = [

         (
            "Logistic Regression",
            run_logistic_regression,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled
        ),

        (
            "Random Forest",
            run_random_forest,
            X_train,
            X_val,
            X_test
        ),

        (
            "XGBoost",
            run_xgboost,
            X_train,
            X_val,
            X_test
        ),

        (
            "MLPClassifier",
            run_mlp_classifier,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled
        ),

        (
            "KNN",
            run_knn,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled
        ),

        (
            "Decision Tree",
            run_decision_tree,
            X_train,
            X_val,
            X_test
        ),

        (
            "Linear SVM",
            run_linear_svm,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled
        ),

        (
            "RBF SVM",
            run_rbf_svm,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled
        ),
    ]

    results = []

    for model_name, model_func, X_tr, X_v, X_te in model_runs:

        print(f"\nRunning {model_name}...")

        result = model_func(
            X_train=X_tr,
            y_train=y_train,
            X_val=X_v,
            y_val=y_val,
            X_test=X_te,
            y_test=y_test
        )

        results.append(result)

    # Save results CSV
    results_df = pd.DataFrame(results)

    detailed_results_path = os.path.join(RESULTS_DIR, DETAILED_RESULTS_FILE)
    summary_results_path = os.path.join(RESULTS_DIR, SUMMARY_RESULTS_FILE)

    results_df.to_csv(detailed_results_path, index=False)

    summary_df = create_summary_results(results_df)
    summary_df.to_csv(summary_results_path, index=False)

    # Generate comparison graph
    plot_model_comparison(
    results_csv=summary_results_path,
    output_name="classical_model_comparison.png"
    )

    # PCA visualisation
    plot_pca_projection(
        X=X_test_scaled,
        y=y_test,
        class_names=CLASS_NAMES,
        output_name="pca_projection.png"
    )

    # t-SNE visualisation
    plot_tsne_projection(
        X=X_test_scaled,
        y=y_test,
        class_names=CLASS_NAMES,
        output_name="tsne_projection.png"
    )

    print("\nDetailed results:")
    print(results_df)

    print("\nSummary results:")
    print(summary_df)

    print(f"\nSaved detailed results to: {detailed_results_path}")
    print(f"Saved summary results to: {summary_results_path}")


if __name__ == "__main__":
    main()