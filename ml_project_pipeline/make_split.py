"""70/15/15 patient-level split for raw_features.npz"""

import argparse
import json
import os
import random
import shutil
from collections import Counter

import numpy as np


def class_dist(mask, y):
    """Return class counts for the masked subset."""
    return {str(k): int(v) for k, v in sorted(Counter(y[mask].tolist()).items())}


def normalised_class_dist(mask, y):
    """Return class proportions for the masked subset."""
    values = y[mask]
    total = len(values)
    counts = Counter(values.tolist())
    if total == 0:
        return {}
    return {int(k): v / total for k, v in counts.items()}


def l1_distance(dist_a, dist_b, classes):
    """Simple distance between two class-proportion dictionaries."""
    return sum(abs(dist_a.get(c, 0.0) - dist_b.get(c, 0.0)) for c in classes)


def warn_missing_classes(name, mask, y, label_map=None):
    """Warn if any class is absent from a split."""
    present = set(np.unique(y[mask]).tolist())
    all_classes = set(np.unique(y).tolist())
    missing = sorted(all_classes - present)

    if missing:
        if label_map is not None:
            missing_readable = [label_map.get(str(c), str(c)) for c in missing]
            print(f"WARNING: {name} split is missing classes: {missing_readable}")
        else:
            print(f"WARNING: {name} split is missing classes: {missing}")


def evaluate_split(train_mask, val_mask, test_mask, y_for_balance):
    """
    Score a split based on how similar each subset's class proportions are
    to the overall class proportions. Lower is better.
    """
    classes = sorted(np.unique(y_for_balance).tolist())
    overall = normalised_class_dist(np.ones(len(y_for_balance), dtype=bool), y_for_balance)

    train_dist = normalised_class_dist(train_mask, y_for_balance)
    val_dist = normalised_class_dist(val_mask, y_for_balance)
    test_dist = normalised_class_dist(test_mask, y_for_balance)

    score = (
        l1_distance(train_dist, overall, classes)
        + l1_distance(val_dist, overall, classes)
        + l1_distance(test_dist, overall, classes)
    )
    return score


def build_patient_split(all_pids, pids, train_frac, val_frac, seed):
    """Create patient-level train/val/test masks for a given seed."""
    rng = random.Random(seed)
    shuffled = list(all_pids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_patients = set(shuffled[:n_train])
    val_patients = set(shuffled[n_train:n_train + n_val])
    test_patients = set(shuffled[n_train + n_val:])

    train_mask = np.array([p in train_patients for p in pids])
    val_mask = np.array([p in val_patients for p in pids])
    test_mask = np.array([p in test_patients for p in pids])

    return train_patients, val_patients, test_patients, train_mask, val_mask, test_mask


def make_split(
    input_path,
    output_dir,
    train_frac=0.70,
    val_frac=0.15,
    seed=42,
    balance_on="cycle",
    seed_search=1
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    z = np.load(input_path, allow_pickle=True)
    X = z["X"]
    y_cycle = z["y_cycle"]
    y_disease = z["y_disease"]
    pids = z["pids"]
    feat_names = list(z["feat_names"])
    disease_to_id = z["disease_to_id"].item()

    print(
        f"Loaded {X.shape[0]} cycles, {X.shape[1]} features, "
        f"{len(set(pids.tolist()))} patients"
    )

    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train + val must be < 1.0; got train={train_frac}, val={val_frac}"
        )

    if balance_on not in {"cycle", "disease"}:
        raise ValueError("balance_on must be either 'cycle' or 'disease'")

    y_for_balance = y_cycle if balance_on == "cycle" else y_disease

    all_pids = sorted(set(pids.tolist()))

    # Optionally try several seeds and keep the split with the closest
    # class balance to the overall dataset.
    best = None
    best_score = float("inf")

    for current_seed in range(seed, seed + seed_search):
        split = build_patient_split(
            all_pids=all_pids,
            pids=pids,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=current_seed
        )
        train_patients, val_patients, test_patients, train_mask, val_mask, test_mask = split

        score = evaluate_split(train_mask, val_mask, test_mask, y_for_balance)

        if score < best_score:
            best_score = score
            best = {
                "seed": current_seed,
                "train_patients": train_patients,
                "val_patients": val_patients,
                "test_patients": test_patients,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
            }

    chosen_seed = best["seed"]
    train_patients = best["train_patients"]
    val_patients = best["val_patients"]
    test_patients = best["test_patients"]
    train_mask = best["train_mask"]
    val_mask = best["val_mask"]
    test_mask = best["test_mask"]

    print(f"Chosen seed: {chosen_seed} (searched {seed_search} seed(s), balance_on='{balance_on}')")
    print(f"Train: {len(train_patients)} patients, {train_mask.sum()} cycles")
    print(f"Val:   {len(val_patients)} patients, {val_mask.sum()} cycles")
    print(f"Test:  {len(test_patients)} patients, {test_mask.sum()} cycles")

    # Sanity checks for leakage and assignment.
    assert train_patients.isdisjoint(val_patients), "Patient overlap between train and val"
    assert train_patients.isdisjoint(test_patients), "Patient overlap between train and test"
    assert val_patients.isdisjoint(test_patients), "Patient overlap between val and test"

    assignment_count = train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)
    assert np.all(assignment_count == 1), "Every sample must belong to exactly one split"

    disease_map = {str(v): k for k, v in disease_to_id.items()}
    cycle_map = {"0": "Normal", "1": "Crackle", "2": "Wheeze", "3": "Both"}

    warn_missing_classes("train (cycle)", train_mask, y_cycle, cycle_map)
    warn_missing_classes("val (cycle)", val_mask, y_cycle, cycle_map)
    warn_missing_classes("test (cycle)", test_mask, y_cycle, cycle_map)

    warn_missing_classes("train (disease)", train_mask, y_disease, disease_map)
    warn_missing_classes("val (disease)", val_mask, y_disease, disease_map)
    warn_missing_classes("test (disease)", test_mask, y_disease, disease_map)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Save split arrays.
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        np.save(f"{output_dir}/X_{name}.npy", X[mask].astype(np.float32))
        np.save(f"{output_dir}/y_{name}_cycle.npy", y_cycle[mask].astype(np.int32))
        np.save(f"{output_dir}/y_{name}_disease.npy", y_disease[mask].astype(np.int64))
        np.save(f"{output_dir}/patient_ids_{name}.npy", pids[mask].astype(np.int32))

    # Save unique patient lists separately as well.
    np.save(f"{output_dir}/train_patients.npy", np.array(sorted(train_patients), dtype=np.int32))
    np.save(f"{output_dir}/val_patients.npy", np.array(sorted(val_patients), dtype=np.int32))
    np.save(f"{output_dir}/test_patients.npy", np.array(sorted(test_patients), dtype=np.int32))

    with open(f"{output_dir}/disease_label_map.json", "w") as f:
        json.dump(disease_map, f, indent=2)

    with open(f"{output_dir}/cycle_label_map.json", "w") as f:
        json.dump(cycle_map, f, indent=2)

    with open(f"{output_dir}/feature_columns.json", "w") as f:
        json.dump(feat_names, f, indent=2)

    # Richer split metadata for the report and debugging.
    metadata = {
        "split_method": f"{train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%} patient-level split",
        "chosen_seed": chosen_seed,
        "seed_search": seed_search,
        "balance_on": balance_on,
        "n_patients": {
            "train": len(train_patients),
            "val": len(val_patients),
            "test": len(test_patients),
        },
        "n_cycles": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "n_features": int(X.shape[1]),
        "patient_ids": {
            "train": sorted(train_patients),
            "val": sorted(val_patients),
            "test": sorted(test_patients),
        },
        "cycle_distribution": {
            "train": class_dist(train_mask, y_cycle),
            "val": class_dist(val_mask, y_cycle),
            "test": class_dist(test_mask, y_cycle),
        },
        "disease_distribution": {
            "train": class_dist(train_mask, y_disease),
            "val": class_dist(val_mask, y_disease),
            "test": class_dist(test_mask, y_disease),
        },
        "checks": {
            "train_val_overlap": False,
            "train_test_overlap": False,
            "val_test_overlap": False,
            "every_sample_assigned_once": True,
        },
    }

    with open(f"{output_dir}/split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {len(os.listdir(output_dir))} files to {output_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", default="raw_features.npz")
    p.add_argument("--out", dest="output_dir", default="clean_data")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--balance_on",
        type=str,
        default="cycle",
        choices=["cycle", "disease"],
        help="Which label to use when searching for the most balanced split"
    )
    p.add_argument(
        "--seed_search",
        type=int,
        default=1,
        help="Number of consecutive seeds to try; best-balanced split is kept"
    )
    args = p.parse_args()

    make_split(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_frac=args.train,
        val_frac=args.val,
        seed=args.seed,
        balance_on=args.balance_on,
        seed_search=args.seed_search,
    )


if __name__ == "__main__":
    main()