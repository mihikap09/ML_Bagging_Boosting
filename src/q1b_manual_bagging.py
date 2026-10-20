"""
Question 1(b): Manual bagging on the Olivetti faces dataset.

Bagging is implemented from scratch:
1. Generate bootstrap samples.
2. Train one DecisionTreeClassifier on each sample.
3. Combine predictions using deterministic majority voting.

The sklearn BaggingClassifier is deliberately not used.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Project paths and experiment constants
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = PROJECT_ROOT / "data" / "split_indices.npz"
RESULT_FILE = PROJECT_ROOT / "results" / "q1b_bagging.csv"

SEED = 26004
N_CLASSES = 40

# Frozen using validation accuracy in Question 1(a).
TREE_MAX_DEPTH = None
TREE_MIN_SAMPLES_LEAF = 4

B_VALUES = [1, 5, 10, 25, 50, 100, 200]
MAX_B = max(B_VALUES)


# ---------------------------------------------------------
# Load the dataset and the previously saved split
# ---------------------------------------------------------

faces = fetch_olivetti_faces(
    data_home=PROJECT_ROOT / "data",
    shuffle=False,
    download_if_missing=True
)

X = np.asarray(faces.data, dtype=np.float64)
y = np.asarray(faces.target, dtype=np.int64)

split_data = np.load(SPLIT_FILE)

train_idx = np.asarray(
    split_data["train_idx"],
    dtype=np.int64
)
val_idx = np.asarray(
    split_data["val_idx"],
    dtype=np.int64
)
test_idx = np.asarray(
    split_data["test_idx"],
    dtype=np.int64
)

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# ---------------------------------------------------------
# Verify the common assignment split
# ---------------------------------------------------------

assert X.shape == (400, 4096)
assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80

# Every identity must contribute 6/2/2 observations.
for class_label in range(N_CLASSES):
    assert np.sum(y_train == class_label) == 6
    assert np.sum(y_val == class_label) == 2
    assert np.sum(y_test == class_label) == 2

# The three partitions must not overlap.
assert len(
    set(train_idx)
    & set(val_idx)
) == 0

assert len(
    set(train_idx)
    & set(test_idx)
) == 0

assert len(
    set(val_idx)
    & set(test_idx)
) == 0


# ---------------------------------------------------------
# Deterministic multiclass majority voting
# ---------------------------------------------------------

def majority_vote(prediction_matrix, n_classes):
    """
    Convert individual tree predictions into ensemble
    predictions.

    prediction_matrix has shape:
        (number_of_trees, number_of_observations)

    np.argmax deterministically chooses the smallest class
    index if two or more classes receive the same number
    of votes.
    """
    number_of_observations = prediction_matrix.shape[1]

    ensemble_predictions = np.empty(
        number_of_observations,
        dtype=np.int64
    )

    for observation_index in range(
        number_of_observations
    ):
        vote_counts = np.bincount(
            prediction_matrix[:, observation_index],
            minlength=n_classes
        )

        ensemble_predictions[observation_index] = (
            np.argmax(vote_counts)
        )

    return ensemble_predictions


# ---------------------------------------------------------
# Generate bootstrap samples and train all 200 trees once
# ---------------------------------------------------------

number_of_training_examples = len(X_train)
rng = np.random.default_rng(SEED)

train_prediction_matrix = np.empty(
    (MAX_B, len(X_train)),
    dtype=np.int64
)

val_prediction_matrix = np.empty(
    (MAX_B, len(X_val)),
    dtype=np.int64
)

test_prediction_matrix = np.empty(
    (MAX_B, len(X_test)),
    dtype=np.int64
)

distinct_counts = np.empty(
    MAX_B,
    dtype=np.int64
)

individual_fit_times = np.empty(
    MAX_B,
    dtype=np.float64
)

for tree_index in range(MAX_B):
    # Sample n positions from {0, ..., n-1} with replacement.
    bootstrap_positions = rng.integers(
        low=0,
        high=number_of_training_examples,
        size=number_of_training_examples
    )

    distinct_counts[tree_index] = len(
        np.unique(bootstrap_positions)
    )

    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=SEED + tree_index
    )

    start_time = time.perf_counter()

    tree.fit(
        X_train[bootstrap_positions],
        y_train[bootstrap_positions]
    )

    individual_fit_times[tree_index] = (
        time.perf_counter() - start_time
    )

    train_prediction_matrix[tree_index] = tree.predict(
        X_train
    )
    val_prediction_matrix[tree_index] = tree.predict(
        X_val
    )
    test_prediction_matrix[tree_index] = tree.predict(
        X_test
    )

    if (
        tree_index == 0
        or (tree_index + 1) % 25 == 0
    ):
        print(
            f"Trained {tree_index + 1}/{MAX_B} trees"
        )


# ---------------------------------------------------------
# Theoretical distinct-example expectation
# ---------------------------------------------------------

# For any one training example, the probability of never
# being selected in n bootstrap draws is (1 - 1/n)^n.
#
# Therefore, its probability of appearing at least once is:
#     1 - (1 - 1/n)^n
#
# Multiplying by n gives the expected number of distinct
# examples in one bootstrap sample.

theoretical_distinct_count = (
    number_of_training_examples
    * (
        1.0
        - (
            1.0
            - 1.0 / number_of_training_examples
        )
        ** number_of_training_examples
    )
)

theoretical_distinct_fraction = (
    theoretical_distinct_count
    / number_of_training_examples
)


# ---------------------------------------------------------
# Evaluate each requested ensemble size
# ---------------------------------------------------------

result_rows = []

for B in B_VALUES:
    train_predictions = majority_vote(
        train_prediction_matrix[:B],
        N_CLASSES
    )

    val_predictions = majority_vote(
        val_prediction_matrix[:B],
        N_CLASSES
    )

    test_predictions = majority_vote(
        test_prediction_matrix[:B],
        N_CLASSES
    )

    average_distinct_count = float(
        np.mean(distinct_counts[:B])
    )

    result_rows.append(
        {
            "B": B,
            "train_accuracy": accuracy_score(
                y_train,
                train_predictions
            ),
            "validation_accuracy": accuracy_score(
                y_val,
                val_predictions
            ),
            "test_accuracy": accuracy_score(
                y_test,
                test_predictions
            ),
            "average_distinct_examples":
                average_distinct_count,
            "average_distinct_fraction":
                average_distinct_count
                / number_of_training_examples,
            "theoretical_distinct_examples":
                theoretical_distinct_count,
            "theoretical_distinct_fraction":
                theoretical_distinct_fraction,
            "cumulative_fit_time_seconds": float(
                np.sum(individual_fit_times[:B])
            )
        }
    )

results = pd.DataFrame(result_rows)

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    RESULT_FILE,
    index=False
)


# ---------------------------------------------------------
# Report the result
# ---------------------------------------------------------

print()
print("Frozen tree configuration:")
print("max_depth:", TREE_MAX_DEPTH)
print(
    "min_samples_leaf:",
    TREE_MIN_SAMPLES_LEAF
)

print()
print(
    "Training examples:",
    number_of_training_examples
)

print(
    "Theoretical expected distinct examples:",
    theoretical_distinct_count
)

print(
    "Theoretical expected distinct fraction:",
    theoretical_distinct_fraction
)

print()
print(results.to_string(index=False))

print()
print("Saved:", RESULT_FILE)