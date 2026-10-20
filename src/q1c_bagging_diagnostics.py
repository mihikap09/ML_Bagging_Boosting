"""
Question 1(c): Bagging diagnostics.

For every requested ensemble size B, this script computes:

1. Out-of-bag accuracy.
2. Average pairwise disagreement on the validation set.
3. Mean vote margin on the validation set.
4. Validation accuracy for selecting the final ensemble size.

Bagging, bootstrap sampling and voting are implemented manually.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths and experiment constants
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = PROJECT_ROOT / "data" / "split_indices.npz"
RESULT_FILE = (
    PROJECT_ROOT / "results" / "q1c_diagnostics.csv"
)

FIGURE_FILE = (
    PROJECT_ROOT / "figures" / "q1c_diagnostics.pdf"
)

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
def calculate_oob_accuracy(
    prediction_matrix,
    in_bag_matrix,
    true_labels,
    n_classes
):
    """
    Calculate out-of-bag accuracy.

    For each training observation, only predictions from
    trees that did not use that observation for training
    are included.

    Observations with no available OOB prediction are
    excluded from the accuracy calculation.
    """
    number_of_observations = len(true_labels)

    oob_predictions = np.full(
        number_of_observations,
        fill_value=-1,
        dtype=np.int64
    )

    oob_voter_counts = np.zeros(
        number_of_observations,
        dtype=np.int64
    )

    for observation_index in range(
        number_of_observations
    ):
        # False in the in-bag matrix means that the
        # observation was absent from that bootstrap sample.
        oob_tree_mask = ~in_bag_matrix[
            :,
            observation_index
        ]

        oob_votes = prediction_matrix[
            oob_tree_mask,
            observation_index
        ]

        oob_voter_counts[
            observation_index
        ] = len(oob_votes)

        if len(oob_votes) > 0:
            vote_counts = np.bincount(
                oob_votes,
                minlength=n_classes
            )

            # np.argmax resolves a vote tie by selecting
            # the smallest class label.
            oob_predictions[
                observation_index
            ] = np.argmax(vote_counts)

    covered_mask = oob_voter_counts > 0
    coverage_count = int(np.sum(covered_mask))

    if coverage_count == 0:
        return np.nan, 0

    oob_accuracy = np.mean(
        oob_predictions[covered_mask]
        == true_labels[covered_mask]
    )

    return float(oob_accuracy), coverage_count
def calculate_pairwise_disagreement(
    prediction_matrix,
    n_classes
):
    """
    Compute the average pairwise disagreement between
    trees on a dataset.

    For a pair of trees, disagreement is the fraction of
    observations for which their predicted classes differ.

    The final result averages this quantity over every
    possible pair of trees.
    """
    number_of_trees = prediction_matrix.shape[0]
    number_of_observations = (
        prediction_matrix.shape[1]
    )

    # Pairwise disagreement is undefined for one tree
    # because no tree pair exists.
    if number_of_trees < 2:
        return np.nan

    number_of_tree_pairs = (
        number_of_trees
        * (number_of_trees - 1)
        / 2
    )

    disagreement_values = []

    for observation_index in range(
        number_of_observations
    ):
        class_counts = np.bincount(
            prediction_matrix[:, observation_index],
            minlength=n_classes
        )

        # If m trees predict one class, those trees form
        # m(m-1)/2 agreeing pairs.
        number_of_agreeing_pairs = np.sum(
            class_counts
            * (class_counts - 1)
            / 2
        )

        number_of_disagreeing_pairs = (
            number_of_tree_pairs
            - number_of_agreeing_pairs
        )

        disagreement_values.append(
            number_of_disagreeing_pairs
            / number_of_tree_pairs
        )

    return float(np.mean(disagreement_values))


def calculate_mean_vote_margin(
    prediction_matrix,
    n_classes
):
    """
    Calculate the mean vote margin.

    For each observation:

        margin =
        winner vote fraction - runner-up vote fraction

    The function returns the average margin across all
    observations.
    """
    number_of_trees = prediction_matrix.shape[0]
    number_of_observations = (
        prediction_matrix.shape[1]
    )

    margins = []

    for observation_index in range(
        number_of_observations
    ):
        vote_counts = np.bincount(
            prediction_matrix[:, observation_index],
            minlength=n_classes
        )

        sorted_vote_counts = np.sort(vote_counts)

        winner_count = sorted_vote_counts[-1]
        runner_up_count = sorted_vote_counts[-2]

        margin = (
            winner_count - runner_up_count
        ) / number_of_trees

        margins.append(margin)

    return float(np.mean(margins))


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
# in_bag_matrix[b, i] is True if training
# observation i appeared in tree b's bootstrap sample.
in_bag_matrix = np.zeros(
    (MAX_B, number_of_training_examples),
    dtype=bool
)

for tree_index in range(MAX_B):
    # Sample n positions from {0, ..., n-1} with replacement.
    bootstrap_positions = rng.integers(
        low=0,
        high=number_of_training_examples,
        size=number_of_training_examples
    )
    # Mark all observations present in this bootstrap sample.
    in_bag_matrix[
        tree_index,
        np.unique(bootstrap_positions)
    ] = True

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

results = pd.DataFrame(result_rows)


# ---------------------------------------------------------
# Compute Question 1(c) diagnostics
# ---------------------------------------------------------

diagnostic_rows = []

for B in B_VALUES:
    oob_accuracy, oob_coverage_count = (
        calculate_oob_accuracy(
            prediction_matrix=(
                train_prediction_matrix[:B]
            ),
            in_bag_matrix=in_bag_matrix[:B],
            true_labels=y_train,
            n_classes=N_CLASSES
        )
    )

    pairwise_disagreement = (
        calculate_pairwise_disagreement(
            prediction_matrix=(
                val_prediction_matrix[:B]
            ),
            n_classes=N_CLASSES
        )
    )

    mean_vote_margin = calculate_mean_vote_margin(
        prediction_matrix=(
            val_prediction_matrix[:B]
        ),
        n_classes=N_CLASSES
    )

    validation_accuracy = float(
        results.loc[
            results["B"] == B,
            "validation_accuracy"
        ].iloc[0]
    )

    diagnostic_rows.append(
        {
            "B": B,
            "validation_accuracy":
                validation_accuracy,
            "oob_accuracy":
                oob_accuracy,
            "oob_coverage_count":
                oob_coverage_count,
            "oob_coverage_fraction":
                oob_coverage_count
                / number_of_training_examples,
            "validation_pairwise_disagreement":
                pairwise_disagreement,
            "validation_mean_vote_margin":
                mean_vote_margin
        }
    )

diagnostics = pd.DataFrame(diagnostic_rows)


# ---------------------------------------------------------
# Select final B using validation accuracy only
# ---------------------------------------------------------

best_validation_accuracy = float(
    diagnostics["validation_accuracy"].max()
)

# Deterministic tie rule:
# choose the smaller B if validation accuracies tie.
selected_B = int(
    diagnostics.loc[
        diagnostics["validation_accuracy"]
        == best_validation_accuracy,
        "B"
    ].min()
)


# ---------------------------------------------------------
# Save diagnostic results
# ---------------------------------------------------------

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

diagnostics.to_csv(
    RESULT_FILE,
    index=False
)
# ---------------------------------------------------------
# Plot all three diagnostics versus B
# ---------------------------------------------------------

FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(7, 10),
    sharex=True
)

axes[0].plot(
    diagnostics["B"],
    diagnostics["oob_accuracy"],
    marker="o",
    linewidth=2
)
axes[0].axvline(
    selected_B,
    color="black",
    linestyle="--",
    alpha=0.6,
    label=f"Selected B = {selected_B}"
)
axes[0].set_ylabel("OOB accuracy")
axes[0].set_title(
    "Manual bagging diagnostics"
)
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(
    diagnostics["B"],
    diagnostics[
        "validation_pairwise_disagreement"
    ],
    marker="o",
    linewidth=2,
    color="tab:orange"
)
axes[1].axvline(
    selected_B,
    color="black",
    linestyle="--",
    alpha=0.6
)
axes[1].set_ylabel(
    "Pairwise disagreement"
)
axes[1].grid(alpha=0.3)

axes[2].plot(
    diagnostics["B"],
    diagnostics[
        "validation_mean_vote_margin"
    ],
    marker="o",
    linewidth=2,
    color="tab:green"
)
axes[2].axvline(
    selected_B,
    color="black",
    linestyle="--",
    alpha=0.6
)
axes[2].set_xlabel(
    "Number of trees, B"
)
axes[2].set_ylabel(
    "Mean vote margin"
)
axes[2].set_xticks(B_VALUES)
axes[2].grid(alpha=0.3)

figure.tight_layout()

figure.savefig(
    FIGURE_FILE,
    bbox_inches="tight"
)

plt.close(figure)

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
print("Question 1(c) diagnostics:")
print(diagnostics.to_string(index=False))

print()
print("Final B selected using validation accuracy:")
print("Selected B:", selected_B)
print(
    "Best validation accuracy:",
    best_validation_accuracy
)
print(
    "Tie rule: smaller B is selected when "
    "validation accuracies are equal."
)

print()
print("Saved diagnostic table:", RESULT_FILE)
print("Saved diagnostic figure:", FIGURE_FILE)