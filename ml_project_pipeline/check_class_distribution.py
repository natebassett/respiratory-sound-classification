import json
from collections import Counter

import numpy as np


# Convert numeric class IDs into readable disease names
with open("clean_data/disease_label_map.json", "r") as f:
    raw_label_map = json.load(f)

label_map = {
    int(class_id): disease_name
    for class_id, disease_name in raw_label_map.items()
}


# Load disease labels for each dataset split
y_train = np.load("clean_data/y_train_disease.npy")
y_val = np.load("clean_data/y_val_disease.npy")
y_test = np.load("clean_data/y_test_disease.npy")


def print_distribution(split_name, labels):
    """Print the number of samples per disease class."""

    counts = Counter(labels)
    total_samples = len(labels)

    print(f"\n{split_name}")
    print("-" * len(split_name))

    for class_id in sorted(label_map.keys()):
        disease_name = label_map[class_id]

        count = counts.get(class_id, 0)
        percentage = (count / total_samples) * 100

        print(
            f"{disease_name:15s} "
            f"{count:5d} samples "
            f"({percentage:.2f}%)"
        )


print_distribution("TRAIN", y_train)
print_distribution("VALIDATION", y_val)
print_distribution("TEST", y_test)