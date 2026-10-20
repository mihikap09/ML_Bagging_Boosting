"""
Question 3(a): Measured bias squared and variance.

The clean test set remains fixed. Thirty new training sets are
created by sampling with replacement separately within every
identity. Each resampled training set therefore contains six
observations for each of the 40 identities.

The following frozen models are refitted on every resampled
training set:

1. Selected single decision tree.
2. Final manually implemented bagging model.
3. Selected manually implemented SAMME boosting model.

For test observation i and repetition r, the class-output vector
p_r(x_i) is:

- single tree: class-probability vector;
- bagging: normalized vote-count vector;
- boosting: normalized alpha-weighted vote vector.

The script measures:

    Bias^2 =
    (1/N) sum_i ||mean_p(x_i) - e_i||_2^2

    Variance =
    (1/(N R)) sum_i sum_r
    ||p_r(x_i) - mean_p(x_i)||_2^2

A checkpoint is saved after every completed repetition. If the
script is interrupted, rerunning it resumes from the checkpoint.

sklearn's BaggingClassifier and AdaBoostClassifier are not used.
"""

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = (
    PROJECT_ROOT / "data" / "split_indices.npz"
)

Q1C_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q1c_diagnostics.csv"
)

Q2B_SELECTED_MODEL_FILE = (
    PROJECT_ROOT / "results" / "q2b_selected_model.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3a_bias_variance_checkpoint.npz"
)

TEMP_CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3a_bias_variance_checkpoint_temp.npz"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3a_bias_variance_summary.csv"
)

ACCURACY_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3a_accuracy_by_refit.csv"
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

SEED = 26004

N_CLASSES = 40
N_REPETITIONS = 30

SAMPLES_PER_CLASS = 6

# Frozen single-tree configuration from Question 1(a).
SINGLE_TREE_MAX_DEPTH = None
SINGLE_TREE_MIN_SAMPLES_LEAF = 4

RANDOM_GUESS_ERROR = 1.0 - 1.0 / N_CLASSES
ERROR_EPSILON = 1e-12


# ---------------------------------------------------------
# Load frozen bagging and boosting configurations
# ---------------------------------------------------------

q1c_results = pd.read_csv(
    Q1C_RESULT_FILE
)

required_q1c_columns = {
    "B",
    "validation_accuracy"
}

assert required_q1c_columns.issubset(
    q1c_results.columns
)

selected_bagging_row = (
    q1c_results
    .sort_values(
        by=["validation_accuracy", "B"],
        ascending=[False, True]
    )
    .iloc[0]
)

FINAL_BAGGING_SIZE = int(
    selected_bagging_row["B"]
)


q2b_selected = pd.read_csv(
    Q2B_SELECTED_MODEL_FILE
)

required_q2b_columns = {
    "selected_depth",
    "selected_rounds"
}

assert required_q2b_columns.issubset(
    q2b_selected.columns
)

FINAL_BOOSTING_DEPTH = int(
    q2b_selected.loc[
        0,
        "selected_depth"
    ]
)

FINAL_BOOSTING_ROUNDS = int(
    q2b_selected.loc[
        0,
        "selected_rounds"
    ]
)

print("Frozen model configurations:")
print(
    "Single tree:",
    f"max_depth={SINGLE_TREE_MAX_DEPTH},",
    (
        "min_samples_leaf="
        f"{SINGLE_TREE_MIN_SAMPLES_LEAF}"
    )
)
print("Bagging size:", FINAL_BAGGING_SIZE)
print(
    "Boosting:",
    f"depth={FINAL_BOOSTING_DEPTH},",
    f"rounds={FINAL_BOOSTING_ROUNDS}"
)


# ---------------------------------------------------------
# Load Olivetti Faces and the common split
# ---------------------------------------------------------

faces = fetch_olivetti_faces(
    data_home=str(PROJECT_ROOT / "data"),
    shuffle=False,
    download_if_missing=True
)

X = np.asarray(
    faces.data,
    dtype=np.float64
)

y = np.asarray(
    faces.target,
    dtype=np.int64
)

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

X_original_train = X[train_idx]
y_original_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

N_TEST = len(X_test)


# ---------------------------------------------------------
# Verify the saved split
# ---------------------------------------------------------

assert X.shape == (400, 4096)

assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

for class_label in range(N_CLASSES):
    assert (
        np.sum(
            y_original_train == class_label
        )
        == SAMPLES_PER_CLASS
    )

    assert np.sum(y_test == class_label) == 2


# ---------------------------------------------------------
# Create the one-hot test targets e_i
# ---------------------------------------------------------

test_one_hot = np.zeros(
    (N_TEST, N_CLASSES),
    dtype=np.float64
)

test_one_hot[
    np.arange(N_TEST),
    y_test
] = 1.0

assert np.allclose(
    np.sum(test_one_hot, axis=1),
    1.0
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def create_stratified_resample(
    y_train,
    repetition_index
):
    """
    Sample six positions with replacement separately within
    every identity's original six training observations.

    Returns positions relative to X_original_train.
    """
    rng = np.random.default_rng(
        SEED + 100000 + repetition_index
    )

    resampled_positions = []

    for class_label in range(N_CLASSES):
        class_positions = np.flatnonzero(
            y_train == class_label
        )

        assert (
            len(class_positions)
            == SAMPLES_PER_CLASS
        )

        sampled_class_positions = rng.choice(
            class_positions,
            size=SAMPLES_PER_CLASS,
            replace=True
        )

        resampled_positions.extend(
            sampled_class_positions.tolist()
        )

    resampled_positions = np.asarray(
        resampled_positions,
        dtype=np.int64
    )

    assert len(resampled_positions) == 240

    resampled_labels = y_train[
        resampled_positions
    ]

    for class_label in range(N_CLASSES):
        assert (
            np.sum(
                resampled_labels == class_label
            )
            == SAMPLES_PER_CLASS
        )

    return resampled_positions


def map_tree_probabilities(
    probability_matrix,
    model_classes
):
    """
    Map a tree's predict_proba output into a complete
    40-class probability vector.

    This remains correct even if a fitted model unexpectedly
    lacks one or more classes.
    """
    complete_probabilities = np.zeros(
        (
            probability_matrix.shape[0],
            N_CLASSES
        ),
        dtype=np.float64
    )

    complete_probabilities[
        :,
        model_classes.astype(np.int64)
    ] = probability_matrix

    return complete_probabilities


def train_single_tree_output(
    X_train,
    y_train,
    repetition_index
):
    """
    Fit the frozen single tree and return its test probability
    vectors and test accuracy.
    """
    tree = DecisionTreeClassifier(
        max_depth=SINGLE_TREE_MAX_DEPTH,
        min_samples_leaf=(
            SINGLE_TREE_MIN_SAMPLES_LEAF
        ),
        random_state=(
            SEED
            + 300000
            + repetition_index
        )
    )

    tree.fit(
        X_train,
        y_train
    )

    raw_probabilities = tree.predict_proba(
        X_test
    )

    probabilities = map_tree_probabilities(
        raw_probabilities,
        tree.classes_
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return probabilities, float(accuracy)


def train_manual_bagging_output(
    X_train,
    y_train,
    repetition_index
):
    """
    Fit the frozen manual bagging model and return normalized
    test vote vectors and test accuracy.
    """
    number_of_training_examples = len(X_train)

    rng = np.random.default_rng(
        SEED + 400000 + repetition_index
    )

    vote_counts = np.zeros(
        (N_TEST, N_CLASSES),
        dtype=np.int64
    )

    test_positions = np.arange(N_TEST)

    for tree_index in range(
        FINAL_BAGGING_SIZE
    ):
        bootstrap_positions = rng.integers(
            low=0,
            high=number_of_training_examples,
            size=number_of_training_examples
        )

        tree = DecisionTreeClassifier(
            max_depth=SINGLE_TREE_MAX_DEPTH,
            min_samples_leaf=(
                SINGLE_TREE_MIN_SAMPLES_LEAF
            ),
            random_state=(
                SEED
                + 500000
                + repetition_index * 1000
                + tree_index
            )
        )

        tree.fit(
            X_train[bootstrap_positions],
            y_train[bootstrap_positions]
        )

        predictions = tree.predict(
            X_test
        ).astype(np.int64)

        vote_counts[
            test_positions,
            predictions
        ] += 1

    normalized_votes = (
        vote_counts.astype(np.float64)
        / FINAL_BAGGING_SIZE
    )

    assert np.allclose(
        np.sum(normalized_votes, axis=1),
        1.0
    )

    ensemble_predictions = np.argmax(
        normalized_votes,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        ensemble_predictions
    )

    return normalized_votes, float(accuracy)


def train_manual_boosting_output(
    X_train,
    y_train,
    repetition_index
):
    """
    Fit the frozen manual SAMME model and return normalized
    alpha-weighted test-vote vectors, test accuracy and the
    number of completed boosting rounds.
    """
    number_of_training_examples = len(X_train)

    sample_weights = np.full(
        number_of_training_examples,
        fill_value=(
            1.0 / number_of_training_examples
        ),
        dtype=np.float64
    )

    weighted_scores = np.zeros(
        (N_TEST, N_CLASSES),
        dtype=np.float64
    )

    test_positions = np.arange(N_TEST)

    total_alpha = 0.0
    completed_rounds = 0

    for round_index in range(
        FINAL_BOOSTING_ROUNDS
    ):
        weak_learner = DecisionTreeClassifier(
            max_depth=FINAL_BOOSTING_DEPTH,
            random_state=(
                SEED
                + 600000
                + repetition_index * 1000
                + round_index
            )
        )

        weak_learner.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

        train_predictions = weak_learner.predict(
            X_train
        ).astype(np.int64)

        incorrect = (
            train_predictions != y_train
        )

        weighted_error = float(
            np.sum(
                sample_weights
                * incorrect.astype(np.float64)
            )
        )

        # Explicitly reject a learner that is not better than
        # random multiclass prediction.
        if weighted_error >= RANDOM_GUESS_ERROR:
            print(
                "  Boosting stopped at round",
                round_index + 1,
                "because weighted error",
                f"{weighted_error:.6f}",
                "was not below",
                f"{RANDOM_GUESS_ERROR:.6f}"
            )

            break

        error_used_for_alpha = max(
            weighted_error,
            ERROR_EPSILON
        )

        alpha = float(
            np.log(
                (1.0 - error_used_for_alpha)
                / error_used_for_alpha
            )
            + np.log(N_CLASSES - 1.0)
        )

        test_predictions = weak_learner.predict(
            X_test
        ).astype(np.int64)

        weighted_scores[
            test_positions,
            test_predictions
        ] += alpha

        total_alpha += alpha
        completed_rounds += 1

        sample_weights = (
            sample_weights
            * np.exp(
                alpha
                * incorrect.astype(np.float64)
            )
        )

        normalization_constant = float(
            np.sum(sample_weights)
        )

        if (
            not np.isfinite(
                normalization_constant
            )
            or normalization_constant <= 0.0
        ):
            raise RuntimeError(
                "Invalid SAMME normalization during "
                f"repetition {repetition_index + 1}, "
                f"round {round_index + 1}."
            )

        sample_weights = (
            sample_weights
            / normalization_constant
        )

        assert np.isclose(
            np.sum(sample_weights),
            1.0
        )

        # A perfect weak learner makes further rounds
        # unnecessary.
        if weighted_error <= ERROR_EPSILON:
            print(
                "  Boosting encountered a perfect weak "
                f"learner at round {round_index + 1}."
            )

            break

    if completed_rounds == 0:
        raise RuntimeError(
            "Boosting accepted no weak learners during "
            f"repetition {repetition_index + 1}."
        )

    normalized_weighted_votes = (
        weighted_scores / total_alpha
    )

    assert np.allclose(
        np.sum(
            normalized_weighted_votes,
            axis=1
        ),
        1.0
    )

    ensemble_predictions = np.argmax(
        normalized_weighted_votes,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        ensemble_predictions
    )

    return (
        normalized_weighted_votes,
        float(accuracy),
        completed_rounds
    )


def save_checkpoint(
    completed,
    single_outputs,
    bagging_outputs,
    boosting_outputs,
    single_accuracies,
    bagging_accuracies,
    boosting_accuracies,
    boosting_round_counts,
    repetition_times
):
    """
    Save the checkpoint atomically.

    Data are first written to a temporary file and then moved
    over the previous checkpoint.
    """
    CHECKPOINT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        TEMP_CHECKPOINT_FILE,
        completed=completed,
        single_outputs=single_outputs,
        bagging_outputs=bagging_outputs,
        boosting_outputs=boosting_outputs,
        single_accuracies=single_accuracies,
        bagging_accuracies=bagging_accuracies,
        boosting_accuracies=boosting_accuracies,
        boosting_round_counts=(
            boosting_round_counts
        ),
        repetition_times=repetition_times,
        n_repetitions=np.asarray(
            [N_REPETITIONS],
            dtype=np.int64
        ),
        n_test=np.asarray(
            [N_TEST],
            dtype=np.int64
        ),
        n_classes=np.asarray(
            [N_CLASSES],
            dtype=np.int64
        ),
        final_bagging_size=np.asarray(
            [FINAL_BAGGING_SIZE],
            dtype=np.int64
        ),
        final_boosting_depth=np.asarray(
            [FINAL_BOOSTING_DEPTH],
            dtype=np.int64
        ),
        final_boosting_rounds=np.asarray(
            [FINAL_BOOSTING_ROUNDS],
            dtype=np.int64
        )
    )

    os.replace(
        TEMP_CHECKPOINT_FILE,
        CHECKPOINT_FILE
    )


# ---------------------------------------------------------
# Create or load checkpoint arrays
# ---------------------------------------------------------

if CHECKPOINT_FILE.exists():
    print()
    print("Existing Q3(a) checkpoint found.")
    print("Loading completed repetitions...")

    checkpoint = np.load(
        CHECKPOINT_FILE
    )

    assert int(
        checkpoint["n_repetitions"][0]
    ) == N_REPETITIONS

    assert int(
        checkpoint["n_test"][0]
    ) == N_TEST

    assert int(
        checkpoint["n_classes"][0]
    ) == N_CLASSES

    assert int(
        checkpoint["final_bagging_size"][0]
    ) == FINAL_BAGGING_SIZE

    assert int(
        checkpoint[
            "final_boosting_depth"
        ][0]
    ) == FINAL_BOOSTING_DEPTH

    assert int(
        checkpoint[
            "final_boosting_rounds"
        ][0]
    ) == FINAL_BOOSTING_ROUNDS

    completed = np.asarray(
        checkpoint["completed"],
        dtype=bool
    )

    single_outputs = np.asarray(
        checkpoint["single_outputs"],
        dtype=np.float64
    )

    bagging_outputs = np.asarray(
        checkpoint["bagging_outputs"],
        dtype=np.float64
    )

    boosting_outputs = np.asarray(
        checkpoint["boosting_outputs"],
        dtype=np.float64
    )

    single_accuracies = np.asarray(
        checkpoint["single_accuracies"],
        dtype=np.float64
    )

    bagging_accuracies = np.asarray(
        checkpoint["bagging_accuracies"],
        dtype=np.float64
    )

    boosting_accuracies = np.asarray(
        checkpoint["boosting_accuracies"],
        dtype=np.float64
    )

    boosting_round_counts = np.asarray(
        checkpoint["boosting_round_counts"],
        dtype=np.int64
    )

    repetition_times = np.asarray(
        checkpoint["repetition_times"],
        dtype=np.float64
    )

    print(
        "Completed repetitions:",
        int(np.sum(completed)),
        "of",
        N_REPETITIONS
    )

else:
    print()
    print("No existing Q3(a) checkpoint found.")
    print("Starting all repetitions from the beginning.")

    completed = np.zeros(
        N_REPETITIONS,
        dtype=bool
    )

    output_shape = (
        N_REPETITIONS,
        N_TEST,
        N_CLASSES
    )

    single_outputs = np.full(
        output_shape,
        fill_value=np.nan,
        dtype=np.float64
    )

    bagging_outputs = np.full(
        output_shape,
        fill_value=np.nan,
        dtype=np.float64
    )

    boosting_outputs = np.full(
        output_shape,
        fill_value=np.nan,
        dtype=np.float64
    )

    single_accuracies = np.full(
        N_REPETITIONS,
        fill_value=np.nan,
        dtype=np.float64
    )

    bagging_accuracies = np.full(
        N_REPETITIONS,
        fill_value=np.nan,
        dtype=np.float64
    )

    boosting_accuracies = np.full(
        N_REPETITIONS,
        fill_value=np.nan,
        dtype=np.float64
    )

    boosting_round_counts = np.zeros(
        N_REPETITIONS,
        dtype=np.int64
    )

    repetition_times = np.full(
        N_REPETITIONS,
        fill_value=np.nan,
        dtype=np.float64
    )


# ---------------------------------------------------------
# Run or resume the 30 refits
# ---------------------------------------------------------

for repetition_index in range(
    N_REPETITIONS
):
    repetition_number = repetition_index + 1

    if completed[repetition_index]:
        print(
            f"Repetition {repetition_number}/"
            f"{N_REPETITIONS} already complete; "
            "skipping."
        )

        continue

    print()
    print(
        "=" * 60
    )
    print(
        f"Starting repetition {repetition_number}/"
        f"{N_REPETITIONS}"
    )

    repetition_start = time.perf_counter()

    resampled_positions = (
        create_stratified_resample(
            y_original_train,
            repetition_index
        )
    )

    X_resampled = X_original_train[
        resampled_positions
    ]

    y_resampled = y_original_train[
        resampled_positions
    ]

    print("  Fitting single tree...")

    (
        single_output,
        single_accuracy
    ) = train_single_tree_output(
        X_resampled,
        y_resampled,
        repetition_index
    )

    print(
        "  Single-tree test accuracy:",
        f"{single_accuracy:.4f}"
    )

    print(
        "  Fitting",
        FINAL_BAGGING_SIZE,
        "manual bagging trees..."
    )

    (
        bagging_output,
        bagging_accuracy
    ) = train_manual_bagging_output(
        X_resampled,
        y_resampled,
        repetition_index
    )

    print(
        "  Bagging test accuracy:",
        f"{bagging_accuracy:.4f}"
    )

    print(
        "  Fitting up to",
        FINAL_BOOSTING_ROUNDS,
        "manual boosting trees..."
    )

    (
        boosting_output,
        boosting_accuracy,
        completed_boosting_rounds
    ) = train_manual_boosting_output(
        X_resampled,
        y_resampled,
        repetition_index
    )

    print(
        "  Boosting test accuracy:",
        f"{boosting_accuracy:.4f}"
    )

    print(
        "  Completed boosting rounds:",
        completed_boosting_rounds
    )

    single_outputs[
        repetition_index
    ] = single_output

    bagging_outputs[
        repetition_index
    ] = bagging_output

    boosting_outputs[
        repetition_index
    ] = boosting_output

    single_accuracies[
        repetition_index
    ] = single_accuracy

    bagging_accuracies[
        repetition_index
    ] = bagging_accuracy

    boosting_accuracies[
        repetition_index
    ] = boosting_accuracy

    boosting_round_counts[
        repetition_index
    ] = completed_boosting_rounds

    repetition_times[
        repetition_index
    ] = (
        time.perf_counter()
        - repetition_start
    )

    # Mark complete only after every model result has been
    # stored successfully.
    completed[
        repetition_index
    ] = True

    save_checkpoint(
        completed=completed,
        single_outputs=single_outputs,
        bagging_outputs=bagging_outputs,
        boosting_outputs=boosting_outputs,
        single_accuracies=single_accuracies,
        bagging_accuracies=bagging_accuracies,
        boosting_accuracies=boosting_accuracies,
        boosting_round_counts=(
            boosting_round_counts
        ),
        repetition_times=repetition_times
    )

    print(
        "  Repetition time:",
        f"{repetition_times[repetition_index]:.2f}",
        "seconds"
    )

    print(
        "  Checkpoint saved. Total completed:",
        int(np.sum(completed)),
        "of",
        N_REPETITIONS
    )


# ---------------------------------------------------------
# Verify all repetitions completed
# ---------------------------------------------------------

if not np.all(completed):
    raise RuntimeError(
        "Not all repetitions are complete."
    )

assert not np.isnan(single_outputs).any()
assert not np.isnan(bagging_outputs).any()
assert not np.isnan(boosting_outputs).any()

assert np.allclose(
    np.sum(single_outputs, axis=2),
    1.0
)

assert np.allclose(
    np.sum(bagging_outputs, axis=2),
    1.0
)

assert np.allclose(
    np.sum(boosting_outputs, axis=2),
    1.0
)


# ---------------------------------------------------------
# Calculate measured bias squared and variance
# ---------------------------------------------------------

def calculate_bias_squared_and_variance(
    model_outputs,
    one_hot_targets
):
    """
    model_outputs has shape:
        repetitions x test observations x classes
    """
    mean_output = np.mean(
        model_outputs,
        axis=0
    )

    squared_bias_per_observation = np.sum(
        (
            mean_output
            - one_hot_targets
        ) ** 2,
        axis=1
    )

    bias_squared = float(
        np.mean(
            squared_bias_per_observation
        )
    )

    squared_deviation = np.sum(
        (
            model_outputs
            - mean_output[np.newaxis, :, :]
        ) ** 2,
        axis=2
    )

    variance = float(
        np.mean(squared_deviation)
    )

    return bias_squared, variance


single_bias_squared, single_variance = (
    calculate_bias_squared_and_variance(
        single_outputs,
        test_one_hot
    )
)

bagging_bias_squared, bagging_variance = (
    calculate_bias_squared_and_variance(
        bagging_outputs,
        test_one_hot
    )
)

boosting_bias_squared, boosting_variance = (
    calculate_bias_squared_and_variance(
        boosting_outputs,
        test_one_hot
    )
)


# ---------------------------------------------------------
# Build final result tables
# ---------------------------------------------------------

summary_results = pd.DataFrame(
    [
        {
            "model": "single_tree",
            "bias_squared":
                single_bias_squared,
            "variance":
                single_variance,
            "mean_test_accuracy":
                float(
                    np.mean(
                        single_accuracies
                    )
                ),
            "test_accuracy_std_ddof1":
                float(
                    np.std(
                        single_accuracies,
                        ddof=1
                    )
                ),
            "minimum_test_accuracy":
                float(
                    np.min(
                        single_accuracies
                    )
                ),
            "maximum_test_accuracy":
                float(
                    np.max(
                        single_accuracies
                    )
                ),
            "number_of_refits":
                N_REPETITIONS
        },
        {
            "model": "manual_bagging",
            "bias_squared":
                bagging_bias_squared,
            "variance":
                bagging_variance,
            "mean_test_accuracy":
                float(
                    np.mean(
                        bagging_accuracies
                    )
                ),
            "test_accuracy_std_ddof1":
                float(
                    np.std(
                        bagging_accuracies,
                        ddof=1
                    )
                ),
            "minimum_test_accuracy":
                float(
                    np.min(
                        bagging_accuracies
                    )
                ),
            "maximum_test_accuracy":
                float(
                    np.max(
                        bagging_accuracies
                    )
                ),
            "number_of_refits":
                N_REPETITIONS
        },
        {
            "model": "manual_boosting",
            "bias_squared":
                boosting_bias_squared,
            "variance":
                boosting_variance,
            "mean_test_accuracy":
                float(
                    np.mean(
                        boosting_accuracies
                    )
                ),
            "test_accuracy_std_ddof1":
                float(
                    np.std(
                        boosting_accuracies,
                        ddof=1
                    )
                ),
            "minimum_test_accuracy":
                float(
                    np.min(
                        boosting_accuracies
                    )
                ),
            "maximum_test_accuracy":
                float(
                    np.max(
                        boosting_accuracies
                    )
                ),
            "number_of_refits":
                N_REPETITIONS
        }
    ]
)

accuracy_by_refit = pd.DataFrame(
    {
        "repetition": np.arange(
            1,
            N_REPETITIONS + 1
        ),
        "single_tree_test_accuracy":
            single_accuracies,
        "bagging_test_accuracy":
            bagging_accuracies,
        "boosting_test_accuracy":
            boosting_accuracies,
        "boosting_completed_rounds":
            boosting_round_counts,
        "total_repetition_time_seconds":
            repetition_times
    }
)


# ---------------------------------------------------------
# Save final tables
# ---------------------------------------------------------

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary_results.to_csv(
    SUMMARY_FILE,
    index=False
)

accuracy_by_refit.to_csv(
    ACCURACY_FILE,
    index=False
)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("=" * 60)
print("QUESTION 3(a): MEASURED BIAS AND VARIANCE")
print()

print(
    summary_results.to_string(
        index=False
    )
)

print()
print("Per-refit test accuracies:")
print(
    accuracy_by_refit.to_string(
        index=False
    )
)

print()
print(
    "Accuracy standard deviation uses ddof=1 "
    "(sample standard deviation across the 30 refits)."
)

print()
print("Saved checkpoint:", CHECKPOINT_FILE)
print("Saved summary:", SUMMARY_FILE)
print("Saved per-refit results:", ACCURACY_FILE)