# train_cnn.py

import os
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
)

from cnn_models.cnn_model import build_cnn_model

from visualisations import (
    save_confusion_matrix,
    plot_training_history,
    plot_spectrogram_examples,
    plot_misclassified_spectrograms,
    create_combined_results_csv,
    plot_classical_vs_cnn,
)


DATA_DIR = "cnn_data"
RESULTS_DIR = "results"
MODEL_DIR = "results/models"
RESULTS_FILE = "cnn_results.csv"

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train_spec.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_spec.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_spec.npy"))

    y_train = np.load(os.path.join(DATA_DIR, "y_train_cycle.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val_cycle.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test_cycle.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalise_data(X_train, X_val, X_test):
    # Fit normalisation on training data only
    mean = X_train.mean()
    std = X_train.std()

    return (
        (X_train - mean) / (std + 1e-8),
        (X_val - mean) / (std + 1e-8),
        (X_test - mean) / (std + 1e-8),
    )


def apply_spec_augment(X, F=15, T=20, num_masks=2):
    # Apply time and frequency masking to spectrograms
    X_aug = X.copy()

    n_mels = X.shape[1]
    n_frames = X.shape[2]

    for i in range(len(X_aug)):
        for _ in range(num_masks):
            f = np.random.randint(0, F)
            f0 = np.random.randint(0, max(1, n_mels - f))
            X_aug[i, f0:f0 + f, :, :] = 0.0

        for _ in range(num_masks):
            t = np.random.randint(0, T)
            t0 = np.random.randint(0, max(1, n_frames - t))
            X_aug[i, :, t0:t0 + t, :] = 0.0

    return X_aug


def get_class_weights(y_train):
    # Balance classes based on training distribution
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate_split(split_name, y_true, y_pred):
    print(f"\n{'=' * 50}")
    print(f"{split_name} Classification Report")
    print("=" * 50)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return {
        f"{split_name} Accuracy": accuracy_score(y_true, y_pred),
        f"{split_name} Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        f"{split_name} Weighted F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def main():
    print("Loading CNN data...")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    X_train, X_val, X_test = normalise_data(
        X_train,
        X_val,
        X_test,
    )

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Num classes: {num_classes}")
    print(
        "Train class distribution:",
        np.bincount(y_train),
        "(0=normal, 1=crackle, 2=wheeze, 3=both)",
    )

    print("\nApplying SpecAugment to training data...")

    X_train_aug = apply_spec_augment(
        X_train,
        F=15,
        T=20,
        num_masks=2,
    )

    X_train_combined = np.concatenate(
        [X_train, X_train_aug],
        axis=0,
    )

    y_train_combined = np.concatenate(
        [y_train, y_train],
        axis=0,
    )

    print(f"Training set after augmentation: {X_train_combined.shape}")

    class_weights = get_class_weights(y_train_combined)
    print(f"Class weights: {class_weights}")

    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    )

    print("\nTraining...")

    history = model.fit(
        X_train_combined,
        y_train_combined,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1,
    )

    val_pred = np.argmax(
        model.predict(X_val),
        axis=1,
    )

    test_pred = np.argmax(
        model.predict(X_test),
        axis=1,
    )

    val_metrics = evaluate_split(
        "Validation",
        y_val,
        val_pred,
    )

    test_metrics = evaluate_split(
        "Test",
        y_test,
        test_pred,
    )

    save_confusion_matrix(
        y_true=y_val,
        y_pred=val_pred,
        class_names=CLASS_NAMES,
        model_name="CNN Validation",
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        class_names=CLASS_NAMES,
        model_name="CNN Test",
    )

    plot_training_history(history)

    plot_spectrogram_examples(
        X=X_test,
        y=y_test,
        class_names=CLASS_NAMES,
        output_name="spectrogram_examples.png",
    )

    plot_misclassified_spectrograms(
        X=X_test,
        y_true=y_test,
        y_pred=test_pred,
        class_names=CLASS_NAMES,
        output_name="misclassified_examples.png",
    )

    results = {
        "Model": "CNN Mel Spectrogram (Improved)",
        **val_metrics,
        **test_metrics,
        "Epochs Trained": len(history.history["loss"]),
    }

    results_df = pd.DataFrame([results])

    results_path = os.path.join(
        RESULTS_DIR,
        RESULTS_FILE,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "cnn_mel_model.keras",
    )

    model.save(model_path)

    # Create final comparison CSV and graph if classical results exist
    classical_results_path = os.path.join(
        RESULTS_DIR,
        "model_results.csv",
    )

    if os.path.exists(classical_results_path):
        create_combined_results_csv(
            classical_results_csv=classical_results_path,
            cnn_results_csv=results_path,
        )

        plot_classical_vs_cnn()
    else:
        print(
            "\nSkipping classical vs CNN comparison because "
            "results/model_results.csv was not found."
        )

    print("\n" + "=" * 50)
    print("Final CNN Results")
    print("=" * 50)
    print(results_df.T.to_string())

    print(f"\nSaved CNN results to: {results_path}")
    print(f"Saved CNN model to: {model_path}")
    print(f"All outputs saved under: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()