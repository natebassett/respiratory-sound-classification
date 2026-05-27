"""Patient-level k-fold cross-validation splits for raw_features.npz."""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np


def class_dist(mask, y):
    """Return class counts for the masked subset."""
    return {str(k): int(v) for k, v in sorted(Counter(y[mask].tolist()).items())}


def patient_class_counts(pids, y):
    """Aggregate sample-level labels into per-patient class counts."""
    counts = defaultdict(Counter)
    for pid, label in zip(pids.tolist(), y.tolist()):
        counts[int(pid)][int(label)] += 1
    return counts


def greedy_patient_folds(pids, y_for_balance, n_splits, seed):
    """
    Build patient-safe folds while approximately balancing the chosen label.

    The splitter assigns whole patients to folds. Patients with more samples are
    placed first, each time choosing the fold that minimises class-count
    imbalance and fold-size imbalance.
    """
    rng = random.Random(seed)
    per_patient = patient_class_counts(pids, y_for_balance)
    patients = list(per_patient.keys())
    rng.shuffle(patients)
    patients.sort(key=lambda pid: sum(per_patient[pid].values()), reverse=True)

    classes = sorted(np.unique(y_for_balance).tolist())
    total_counts = Counter(y_for_balance.tolist())
    target_counts = {
        cls: total_counts[int(cls)] / float(n_splits)
        for cls in classes
    }
    target_size = len(y_for_balance) / float(n_splits)

    fold_patients = [set() for _ in range(n_splits)]
    fold_counts = [Counter() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    def total_score(candidate_counts, candidate_sizes):
        class_error = 0.0
        size_error = 0.0
        empty_penalty = 0.0
        for fold_idx in range(n_splits):
            if candidate_sizes[fold_idx] == 0:
                empty_penalty += target_size * 10.0
            size_error += abs(candidate_sizes[fold_idx] - target_size)
            class_error += sum(
                abs(candidate_counts[fold_idx].get(int(cls), 0) - target_counts[int(cls)])
                for cls in classes
            )
        return class_error + 0.25 * size_error + empty_penalty

    for idx, pid in enumerate(patients):
        patient_counter = per_patient[pid]

        # Seed each fold with one patient before optimising global balance.
        if idx < n_splits:
            best_fold = idx
        else:
            candidates = []
            for fold_idx in range(n_splits):
                candidate_counts = [Counter(c) for c in fold_counts]
                candidate_sizes = list(fold_sizes)
                candidate_counts[fold_idx].update(patient_counter)
                candidate_sizes[fold_idx] += sum(patient_counter.values())
                candidates.append((total_score(candidate_counts, candidate_sizes), fold_idx))
            _, best_fold = min(candidates, key=lambda item: (item[0], fold_sizes[item[1]]))

        fold_patients[best_fold].add(pid)
        fold_counts[best_fold].update(patient_counter)
        fold_sizes[best_fold] += sum(patient_counter.values())

    return fold_patients


def make_kfold_splits(input_path, output_dir, n_splits=5, seed=42, balance_on="disease"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if balance_on not in {"cycle", "disease"}:
        raise ValueError("balance_on must be either 'cycle' or 'disease'")

    z = np.load(input_path, allow_pickle=True)
    X = z["X"]
    y_cycle = z["y_cycle"]
    y_disease = z["y_disease"]
    pids = z["pids"]
    feat_names = list(z["feat_names"])
    disease_to_id = z["disease_to_id"].item()

    y_for_balance = y_disease if balance_on == "disease" else y_cycle
    all_patients = set(int(p) for p in pids.tolist())

    if n_splits > len(all_patients):
        raise ValueError(
            f"n_splits={n_splits} is larger than the number of patients={len(all_patients)}"
        )

    print(
        f"Loaded {X.shape[0]} cycles, {X.shape[1]} features, "
        f"{len(all_patients)} patients"
    )
    print(f"Creating {n_splits}-fold patient-level CV split, balance_on='{balance_on}'")

    fold_patients = greedy_patient_folds(
        pids=pids,
        y_for_balance=y_for_balance,
        n_splits=n_splits,
        seed=seed
    )

    # Leakage checks: every patient appears in exactly one held-out fold.
    assigned = []
    for patients in fold_patients:
        assigned.extend(sorted(patients))
    assert len(assigned) == len(set(assigned)), "A patient was assigned to multiple folds"
    assert set(assigned) == all_patients, "Some patients were not assigned to a fold"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    disease_map = {str(v): k for k, v in disease_to_id.items()}
    cycle_map = {"0": "Normal", "1": "Crackle", "2": "Wheeze", "3": "Both"}

    with open(f"{output_dir}/disease_label_map.json", "w") as f:
        json.dump(disease_map, f, indent=2)

    with open(f"{output_dir}/cycle_label_map.json", "w") as f:
        json.dump(cycle_map, f, indent=2)

    with open(f"{output_dir}/feature_columns.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    metadata = {
        "split_method": f"{n_splits}-fold patient-level cross-validation",
        "n_splits": n_splits,
        "seed": seed,
        "balance_on": balance_on,
        "n_patients_total": len(all_patients),
        "n_cycles_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "folds": [],
        "checks": {
            "patient_overlap_between_test_folds": False,
            "every_patient_assigned_once_as_test": True,
        },
    }

    for fold_idx, test_patients in enumerate(fold_patients, start=1):
        train_patients = all_patients - test_patients
        train_mask = np.array([int(p) in train_patients for p in pids])
        test_mask = np.array([int(p) in test_patients for p in pids])

        assert set(pids[train_mask].tolist()).isdisjoint(set(pids[test_mask].tolist()))
        assignment_count = train_mask.astype(int) + test_mask.astype(int)
        assert np.all(assignment_count == 1), "Every sample must be train or test"

        fold_dir = f"{output_dir}/fold_{fold_idx}"
        os.makedirs(fold_dir)

        np.save(f"{fold_dir}/X_train.npy", X[train_mask].astype(np.float32))
        np.save(f"{fold_dir}/X_test.npy", X[test_mask].astype(np.float32))
        np.save(f"{fold_dir}/y_train_cycle.npy", y_cycle[train_mask].astype(np.int32))
        np.save(f"{fold_dir}/y_test_cycle.npy", y_cycle[test_mask].astype(np.int32))
        np.save(f"{fold_dir}/y_train_disease.npy", y_disease[train_mask].astype(np.int64))
        np.save(f"{fold_dir}/y_test_disease.npy", y_disease[test_mask].astype(np.int64))
        np.save(f"{fold_dir}/patient_ids_train.npy", pids[train_mask].astype(np.int32))
        np.save(f"{fold_dir}/patient_ids_test.npy", pids[test_mask].astype(np.int32))
        np.save(f"{fold_dir}/train_patients.npy", np.array(sorted(train_patients), dtype=np.int32))
        np.save(f"{fold_dir}/test_patients.npy", np.array(sorted(test_patients), dtype=np.int32))

        fold_info = {
            "fold": fold_idx,
            "n_patients": {
                "train": len(train_patients),
                "test": len(test_patients),
            },
            "n_cycles": {
                "train": int(train_mask.sum()),
                "test": int(test_mask.sum()),
            },
            "patient_ids": {
                "train": sorted(train_patients),
                "test": sorted(test_patients),
            },
            "cycle_distribution": {
                "train": class_dist(train_mask, y_cycle),
                "test": class_dist(test_mask, y_cycle),
            },
            "disease_distribution": {
                "train": class_dist(train_mask, y_disease),
                "test": class_dist(test_mask, y_disease),
            },
            "checks": {
                "train_test_overlap": False,
                "every_sample_assigned_once": True,
            },
        }
        metadata["folds"].append(fold_info)

        with open(f"{fold_dir}/fold_metadata.json", "w") as f:
            json.dump(fold_info, f, indent=2)

        print(
            f"Fold {fold_idx}: train={train_mask.sum()} cycles/"
            f"{len(train_patients)} patients, test={test_mask.sum()} cycles/"
            f"{len(test_patients)} patients"
        )

    with open(f"{output_dir}/kfold_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {n_splits} patient-safe folds to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default="raw_features.npz")
    parser.add_argument("--out", dest="output_dir", default="cv_data")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance_on",
        type=str,
        default="disease",
        choices=["cycle", "disease"],
        help="Which label to approximately balance across held-out folds"
    )
    args = parser.parse_args()

    make_kfold_splits(
        input_path=args.input_path,
        output_dir=args.output_dir,
        n_splits=args.k,
        seed=args.seed,
        balance_on=args.balance_on,
    )


if __name__ == "__main__":
    main()
