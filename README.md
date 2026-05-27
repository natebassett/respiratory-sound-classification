# Respiratory Sound Classification — README

This README explains how to run the Python code submitted as part of the group project. The codebase is split into two parts: **descriptive analysis / dataset statistics** and **model training and evaluation**.

---

## Requirements

Install all dependencies before running any scripts:

```bash
pip install numpy pandas scikit-learn matplotlib librosa tensorflow xgboost tqdm
```

> Python 3.9 or later is recommended.

---

## Dataset

This project uses the **ICBHI 2017 Respiratory Sound Database**. Place the raw `.wav` and `.txt` annotation files inside a folder named:

```
audio_and_txt_files/
```

at the root of the project directory before running any script.

---

## Project Structure

```
.
├── audio_and_txt_files/        # Raw ICBHI audio + annotation files (user-provided)
├── clean_data/                 # Output: processed feature arrays and split metadata
├── cnn_data/                   # Output: mel spectrogram arrays for CNN training
├── results/                    # Output: CSVs, figures, confusion matrices, saved models
│
├── classical_models/
│   ├── decision_tree.py
│   ├── knn.py
│   ├── logistic_regression.py
│   ├── mlp_classifier.py
│   ├── random_forest.py
│   ├── svm_linear.py
│   ├── svm_rbf.py
│   └── xgboost_model.py
│
├── cnn_models/
│   └── cnn_model.py            # CNN architecture definition
│
├── build_cnn_data.py           # Builds mel spectrogram dataset from raw audio
├── check_class_distribution.py # Prints class balance statistics for each split
├── make_split.py               # Creates 70/15/15 patient-level train/val/test split
├── train_cnn.py                # Trains and evaluates the CNN model
├── train_models.py             # Trains and evaluates all classical ML models
├── update_comparison.py        # Regenerates the classical vs CNN comparison plot
├── raw_mats_cache.py           # Extracts and caches ~558 handcrafted audio features
└── visualisations.py           # All plotting and figure-saving utilities
```

---

## Part 1 — Descriptive Analysis of the Dataset

These scripts generate the statistics and visualisations used in the descriptive section of the report.

### Feature Extraction and Caching

The handcrafted feature extraction pipeline is implemented in:

```bash
python raw_mats_cache.py
```

### Step 1: Build the feature dataset and create the data split

First, extract hand-crafted audio features and save them to `raw_features.npz` (feature extraction script not listed here — run it separately if provided). Then create the patient-level train/val/test split:

```bash
python make_split.py
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--in` | `raw_features.npz` | Path to the input feature file |
| `--out` | `clean_data` | Output directory for split arrays |
| `--train` | `0.70` | Fraction of patients for training |
| `--val` | `0.15` | Fraction of patients for validation |
| `--seed` | `42` | Random seed |
| `--balance_on` | `cycle` | Label to optimise class balance on (`cycle` or `disease`) |
| `--seed_search` | `1` | Number of seeds to try; the best-balanced split is kept |

This script produces the following files in `clean_data/`:

- `X_train.npy`, `X_val.npy`, `X_test.npy` — feature arrays
- `y_train_cycle.npy`, `y_val_cycle.npy`, `y_test_cycle.npy` — cycle-level labels (Normal / Crackle / Wheeze / Both)
- `y_train_disease.npy`, `y_val_disease.npy`, `y_test_disease.npy` — patient disease labels
- `split_metadata.json` — split sizes, patient IDs, class distributions, and leakage checks
- `disease_label_map.json`, `cycle_label_map.json`, `feature_columns.json`

### Step 2: Check class distribution

```bash
python check_class_distribution.py
```

Prints a per-class breakdown (sample count and percentage) for the train, validation, and test splits. Requires `clean_data/` to exist from Step 1.

---

## Part 2 — Model Training and Evaluation

### Classical Machine Learning Models

**Prerequisites:** `clean_data/` must exist (run Part 1 first).

```bash
python train_models.py
```

This script:

1. Loads the pre-split feature arrays from `clean_data/`.
2. Applies `StandardScaler` (fitted on training data only) to models that require it.
3. Trains and evaluates eight classifiers — Logistic Regression, Random Forest, XGBoost, MLP, KNN, Decision Tree, Linear SVM, and RBF SVM — each with hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`.
4. Saves results to `results/model_results.csv` and `results/classical_results.csv`.
5. Generates the following figures in `results/figures/`:
   - `classical/class_distribution.png` — training set class balance
   - `classical/pca_projection.png` — 2D PCA projection of test features
   - `classical/tsne_projection.png` — 2D t-SNE projection of test features
   - `comparisons/classical_model_comparison.png` — bar chart of accuracy and F1 across all classical models
   - `confusion_matrices/` — one confusion matrix per model

---

### CNN Model (Mel Spectrogram)

#### Step 1: Build the spectrogram dataset

```bash
python build_cnn_data.py
```

Reads the raw audio files from `audio_and_txt_files/`, converts each annotated breath cycle to a normalised log-mel spectrogram (128 × 128), and saves the patient-level split to `cnn_data/`:

- `X_train_spec.npy`, `X_val_spec.npy`, `X_test_spec.npy`
- `y_train_cycle.npy`, `y_val_cycle.npy`, `y_test_cycle.npy`

**Design notes (report-relevant):**

- Patient-level split prevents data leakage across the train/val/test sets.
- `hop_length=256` gives a frame step of ~12 ms; 128 frames covers ~3 s of audio.
- Short clips are reflect-padded; long clips are centre-cropped to 128 frames.
- Per-sample z-score normalisation removes gain differences between recording devices.

#### Step 2: Train the CNN

```bash
python train_cnn.py
```

This script:

1. Loads spectrogram arrays from `cnn_data/`.
2. Applies SpecAugment (random time and frequency masking) to the training set, doubling the effective training size.
3. Computes class weights to handle the Normal/Crackle/Wheeze/Both imbalance.
4. Trains the CNN (defined in `cnn_models/cnn_model.py`) for up to 150 epochs with early stopping (`patience=20`) and learning-rate reduction on plateau.
5. Saves the trained model to `results/models/cnn_mel_model.keras`.
6. Saves results to `results/cnn_results.csv`.
7. Generates the following figures in `results/figures/cnn/`:
   - `training_curves.png` — loss and accuracy vs epoch
   - `spectrogram_examples.png` — one example spectrogram per class
   - `misclassified_examples.png` — up to six misclassified test spectrograms
   - `confusion_matrices/` — confusion matrices for validation and test sets

If `results/model_results.csv` already exists (i.e. classical models have been run), the script also produces a combined classical-vs-CNN comparison plot.

---

### Regenerating the Comparison Plot

If classical and CNN results already exist and only the comparison figure needs updating:

```bash
python update_comparison.py
```

Reads `results/model_results.csv` and `results/cnn_results.csv`, merges them, and saves the combined bar chart to `results/figures/comparisons/classical_vs_cnn_comparison.png`.

---

## Recommended Run Order

```
1. python make_split.py
2. python check_class_distribution.py
3. python train_models.py
4. python build_cnn_data.py
5. python train_cnn.py
```

Steps 3 and 4–5 are independent of each other and can be run in parallel.

---

## Outputs Summary

| Location | Contents |
|----------|----------|
| `clean_data/` | Processed feature arrays, split metadata, label maps |
| `cnn_data/` | Mel spectrogram arrays for CNN |
| `results/model_results.csv` | Per-model validation and test metrics (classical) |
| `results/cnn_results.csv` | CNN validation and test metrics |
| `results/combined_model_results.csv` | Merged classical + CNN results |
| `results/models/` | Saved Keras CNN model |
| `results/figures/` | All plots (class distribution, PCA, t-SNE, confusion matrices, training curves, comparisons) |
