# train_cnn.py

import os
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

from ml_project_pipeline.old_cnn_files.cnn_model import build_cnn_model


DATA_DIR = "cnn_data"
RESULTS_DIR = "results"
RESULTS_FILE = "cnn_results.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train_spec.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_spec.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_spec.npy"))

    y_train = np.load(os.path.join(DATA_DIR, "y_train_cycle.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val_cycle.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test_cycle.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalise_data(X_train, X_val, X_test):
    # Fit normalisation on training data only to avoid leakage.
    mean = X_train.mean()
    std = X_train.std()

    return (
        (X_train - mean) / (std + 1e-8),
        (X_val - mean) / (std + 1e-8),
        (X_test - mean) / (std + 1e-8),
    )


def get_class_weights(y_train):
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }


def evaluate_split(split_name, y_true, y_pred):
    print(f"\n{split_name} classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        f"{split_name} Accuracy": accuracy_score(y_true, y_pred),
        f"{split_name} Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        f"{split_name} Weighted F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
    }


def main():
    print("Loading CNN data...")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    X_train, X_val, X_test = normalise_data(X_train, X_val, X_test)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Classes: {num_classes}")
    print(f"Train class distribution: {np.bincount(y_train)}")

    class_weights = get_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=35,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=75,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    val_pred = np.argmax(model.predict(X_val), axis=1)
    test_pred = np.argmax(model.predict(X_test), axis=1)

    val_metrics = evaluate_split("Validation", y_val, val_pred)
    test_metrics = evaluate_split("Test", y_test, test_pred)

    results = {
        "Model": "CNN Mel Spectrogram",
        **val_metrics,
        **test_metrics,
        "Epochs Trained": len(history.history["loss"])
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)

    results_df.to_csv(results_path, index=False)
    model.save(os.path.join(RESULTS_DIR, "cnn_mel_model.keras"))

    print("\nFinal CNN results:")
    print(results_df)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()

# OUTPUT:

#     Validation classification report:
#               precision    recall  f1-score   support

#            0       0.74      0.54      0.62       682
#            1       0.41      0.53      0.46       316
#            2       0.08      0.23      0.12        71
#            3       0.17      0.11      0.14        96

#     accuracy                           0.48      1165
#    macro avg       0.35      0.35      0.34      1165
# weighted avg       0.56      0.48      0.51      1165


# Test classification report:
#               precision    recall  f1-score   support

#            0       0.51      0.65      0.58       363
#            1       0.46      0.33      0.38       225
#            2       0.25      0.24      0.24       160
#            3       0.07      0.04      0.05        70

#     accuracy                           0.43       818
#    macro avg       0.32      0.32      0.31       818
# weighted avg       0.41      0.43      0.41       818


# Final CNN results:
#                  Model  Validation Accuracy  Validation Macro F1  Validation Weighted F1  Test Accuracy  Test Macro F1  Test Weighted F1  Epochs Trained
# 0  CNN Mel Spectrogram             0.483262             0.336478                 0.50927       0.430318       0.313939          0.413236              18