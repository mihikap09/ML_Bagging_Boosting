"""
Question 2(b): Manual SAMME AdaBoost on Olivetti Faces.

Weak learner depths:
    1, 2, 3, 4

Requested boosting rounds:
    10, 25, 50, 100

Depth and number of rounds are selected using clean validation
accuracy only.

For the selected depth, the script plots over boosting rounds:

1. Weighted training error.
2. SAMME learner weight alpha.
3. Maximum training-sample weight.
4. Total weight carried by the 10 heaviest training images.

SAMME is implemented manually. sklearn's AdaBoostClassifier
is not used.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Project paths and constants
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = (
    PROJECT_ROOT / "data" / "split_indices.npz"
)

MODEL_SELECTION_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2b_model_selection.csv"
)

SELECTED_ROUND_LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2b_selected_round_log.csv"
)

SELECTED_MODEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2b_selected_model.csv"
)

DIAGNOSTIC_FIGURE_FILE = (
    PROJECT_ROOT
    / "figures"
    / "q2b_selected_diagnostics.pdf"
)

SEED = 26004
N_CLASSES = 40

DEPTH_VALUES = [1, 2, 3, 4]
ROUND_VALUES = [10, 25, 50, 100]
MAX_ROUNDS = max(ROUND_VALUES)

# A valid SAMME learner must have an error below this value.
RANDOM_GUESS_ERROR = 1.0 - 1.0 / N_CLASSES

# Prevent log(0) if a weak learner achieves zero error.
ERROR_EPSILON = 1e-12


# ---------------------------------------------------------
# Load Olivetti Faces and the common saved split
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

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]


# ---------------------------------------------------------
# Verify the common assignment split
# ---------------------------------------------------------

assert X.shape == (400, 4096)

assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

for class_label in range(N_CLASSES):
    assert np.sum(y_train == class_label) == 6
    assert np.sum(y_val == class_label) == 2


# ---------------------------------------------------------
# Manual SAMME helper functions
# ---------------------------------------------------------

def predictions_from_scores(class_scores):
    """
    Convert accumulated SAMME class scores into predictions.

    np.argmax resolves a score tie by selecting the smallest
    class label.
    """
    return np.argmax(
        class_scores,
        axis=1
    ).astype(np.int64)


def add_weighted_votes(
    class_scores,
    predictions,
    alpha
):
    """
    Add alpha to the predicted class score of every sample.

    This implements:

        S_k(x) =
        sum_t alpha_t * 1[h_t(x) = k]
    """
    observation_positions = np.arange(
        len(predictions)
    )

    class_scores[
        observation_positions,
        predictions
    ] += alpha


def run_manual_samme(
    X_train,
    y_train,
    X_val,
    y_val,
    max_depth,
    max_rounds,
    checkpoint_rounds,
    seed,
    verbose=True
):
    """
    Train a multiclass SAMME ensemble manually.

    Returns:
        checkpoint_rows:
            Accuracy and timing at requested rounds.

        round_rows:
            Per-round error, alpha and weight diagnostics.

        stop_reason:
            Reason training completed or stopped.
    """
    number_of_training_examples = len(X_train)

    sample_weights = np.full(
        number_of_training_examples,
        fill_value=(
            1.0 / number_of_training_examples
        ),
        dtype=np.float64
    )

    train_class_scores = np.zeros(
        (
            len(X_train),
            N_CLASSES
        ),
        dtype=np.float64
    )

    val_class_scores = np.zeros(
        (
            len(X_val),
            N_CLASSES
        ),
        dtype=np.float64
    )

    checkpoint_rows = []
    round_rows = []

    cumulative_training_time = 0.0
    cumulative_alpha = 0.0

    stop_reason = "maximum_rounds_reached"

    for round_index in range(max_rounds):
        round_number = round_index + 1

        weak_learner = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=(
                seed
                + max_depth * 1000
                + round_index
            )
        )

        fit_start = time.perf_counter()

        weak_learner.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

        round_fit_time = (
            time.perf_counter() - fit_start
        )

        cumulative_training_time += round_fit_time

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

        # A weak learner at or above the random multiclass
        # threshold must not be added to the ensemble.
        if weighted_error >= RANDOM_GUESS_ERROR:
            stop_reason = (
                "weak_learner_not_better_than_random"
            )

            if verbose:
                print(
                    f"Depth {max_depth}, round "
                    f"{round_number}: stopped because "
                    f"error={weighted_error:.6f} is not "
                    f"below {RANDOM_GUESS_ERROR:.6f}"
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

        val_predictions = weak_learner.predict(
            X_val
        ).astype(np.int64)

        add_weighted_votes(
            train_class_scores,
            train_predictions,
            alpha
        )

        add_weighted_votes(
            val_class_scores,
            val_predictions,
            alpha
        )

        cumulative_alpha += alpha

        # Apply exactly the same update used in Q2(a).
        weight_multipliers = np.exp(
            alpha
            * incorrect.astype(np.float64)
        )

        sample_weights = (
            sample_weights
            * weight_multipliers
        )

        normalization_constant = float(
            np.sum(sample_weights)
        )

        if (
            not np.isfinite(normalization_constant)
            or normalization_constant <= 0.0
        ):
            raise RuntimeError(
                "Invalid SAMME weight normalization "
                f"at depth={max_depth}, "
                f"round={round_number}."
            )

        sample_weights = (
            sample_weights
            / normalization_constant
        )

        assert np.isclose(
            np.sum(sample_weights),
            1.0
        )

        sorted_weights = np.sort(
            sample_weights
        )

        maximum_sample_weight = float(
            sorted_weights[-1]
        )

        top_10_weight_total = float(
            np.sum(sorted_weights[-10:])
        )

        round_rows.append(
            {
                "depth": max_depth,
                "round": round_number,
                "weighted_training_error":
                    weighted_error,
                "alpha": alpha,
                "normalization_constant":
                    normalization_constant,
                "maximum_sample_weight":
                    maximum_sample_weight,
                "top_10_weight_total":
                    top_10_weight_total,
                "cumulative_alpha":
                    cumulative_alpha,
                "round_fit_time_seconds":
                    round_fit_time,
                "cumulative_fit_time_seconds":
                    cumulative_training_time
            }
        )

        if round_number in checkpoint_rounds:
            ensemble_train_predictions = (
                predictions_from_scores(
                    train_class_scores
                )
            )

            ensemble_val_predictions = (
                predictions_from_scores(
                    val_class_scores
                )
            )

            checkpoint_rows.append(
                {
                    "depth": max_depth,
                    "rounds": round_number,
                    "train_accuracy": accuracy_score(
                        y_train,
                        ensemble_train_predictions
                    ),
                    "validation_accuracy":
                        accuracy_score(
                            y_val,
                            ensemble_val_predictions
                        ),
                    "cumulative_alpha":
                        cumulative_alpha,
                    "training_time_seconds":
                        cumulative_training_time,
                    "completed_rounds":
                        round_number,
                    "status": "completed"
                }
            )

        if verbose and (
            round_number == 1
            or round_number % 10 == 0
        ):
            print(
                f"Depth {max_depth}: completed "
                f"{round_number}/{max_rounds} rounds"
            )

        # A perfect weak learner already separates all
        # training observations. Continuing would only repeat
        # a numerically extreme learner.
        if weighted_error <= ERROR_EPSILON:
            stop_reason = "perfect_weak_learner"

            if verbose:
                print(
                    f"Depth {max_depth}, round "
                    f"{round_number}: stopped after a "
                    "perfect weak learner."
                )

            break

    completed_checkpoint_rounds = {
        row["rounds"]
        for row in checkpoint_rows
    }

    # If training stopped before a requested checkpoint, record
    # that configuration explicitly rather than silently using
    # fewer learners.
    for requested_rounds in checkpoint_rounds:
        if requested_rounds not in completed_checkpoint_rounds:
            checkpoint_rows.append(
                {
                    "depth": max_depth,
                    "rounds": requested_rounds,
                    "train_accuracy": np.nan,
                    "validation_accuracy": np.nan,
                    "cumulative_alpha": np.nan,
                    "training_time_seconds":
                        cumulative_training_time,
                    "completed_rounds":
                        len(round_rows),
                    "status": stop_reason
                }
            )

    return (
        checkpoint_rows,
        round_rows,
        stop_reason
    )


# ---------------------------------------------------------
# Evaluate every depth and requested round count
# ---------------------------------------------------------

all_checkpoint_rows = []
round_logs_by_depth = {}
stop_reasons_by_depth = {}

for depth in DEPTH_VALUES:
    print()
    print(
        f"Training manual SAMME with depth={depth}"
    )

    (
        depth_checkpoint_rows,
        depth_round_rows,
        depth_stop_reason
    ) = run_manual_samme(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_depth=depth,
        max_rounds=MAX_ROUNDS,
        checkpoint_rounds=ROUND_VALUES,
        seed=SEED,
        verbose=True
    )

    all_checkpoint_rows.extend(
        depth_checkpoint_rows
    )

    round_logs_by_depth[depth] = pd.DataFrame(
        depth_round_rows
    )

    stop_reasons_by_depth[depth] = (
        depth_stop_reason
    )


# ---------------------------------------------------------
# Build the complete model-selection table
# ---------------------------------------------------------

model_selection_results = pd.DataFrame(
    all_checkpoint_rows
)

model_selection_results = (
    model_selection_results
    .sort_values(
        by=["depth", "rounds"]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Select depth and rounds using validation accuracy only
# ---------------------------------------------------------

valid_candidates = model_selection_results.dropna(
    subset=["validation_accuracy"]
).copy()

if len(valid_candidates) == 0:
    raise RuntimeError(
        "No SAMME configuration reached a requested "
        "round checkpoint."
    )

# Deterministic tie rule:
# 1. Highest validation accuracy.
# 2. Fewer boosting rounds.
# 3. Smaller weak-learner depth.
selected_row = (
    valid_candidates
    .sort_values(
        by=[
            "validation_accuracy",
            "rounds",
            "depth"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )
    .iloc[0]
)

SELECTED_DEPTH = int(
    selected_row["depth"]
)

SELECTED_ROUNDS = int(
    selected_row["rounds"]
)

BEST_VALIDATION_ACCURACY = float(
    selected_row["validation_accuracy"]
)

selected_round_log = (
    round_logs_by_depth[SELECTED_DEPTH]
    .loc[
        lambda table:
            table["round"] <= SELECTED_ROUNDS
    ]
    .copy()
    .reset_index(drop=True)
)

if len(selected_round_log) != SELECTED_ROUNDS:
    raise RuntimeError(
        "Selected model does not contain the expected "
        "number of completed rounds."
    )

selected_model_summary = pd.DataFrame(
    [
        {
            "selected_depth":
                SELECTED_DEPTH,
            "selected_rounds":
                SELECTED_ROUNDS,
            "validation_accuracy":
                BEST_VALIDATION_ACCURACY,
            "train_accuracy":
                float(
                    selected_row[
                        "train_accuracy"
                    ]
                ),
            "training_time_seconds":
                float(
                    selected_row[
                        "training_time_seconds"
                    ]
                ),
            "selection_rule":
                (
                    "highest validation accuracy; "
                    "then fewer rounds; "
                    "then smaller depth"
                ),
            "random_guess_error_threshold":
                RANDOM_GUESS_ERROR,
            "stop_reason_for_selected_depth":
                stop_reasons_by_depth[
                    SELECTED_DEPTH
                ]
        }
    ]
)


# ---------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------

MODEL_SELECTION_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_selection_results.to_csv(
    MODEL_SELECTION_FILE,
    index=False
)

selected_round_log.to_csv(
    SELECTED_ROUND_LOG_FILE,
    index=False
)

selected_model_summary.to_csv(
    SELECTED_MODEL_FILE,
    index=False
)


# ---------------------------------------------------------
# Plot selected-model diagnostics
# ---------------------------------------------------------

DIAGNOSTIC_FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(11, 8)
)

axes[0, 0].plot(
    selected_round_log["round"],
    selected_round_log[
        "weighted_training_error"
    ],
    linewidth=1.8,
    color="tab:blue"
)

axes[0, 0].axhline(
    RANDOM_GUESS_ERROR,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label=(
        "Random threshold "
        f"({RANDOM_GUESS_ERROR:.3f})"
    )
)

axes[0, 0].set_title(
    "Weighted training error"
)
axes[0, 0].set_xlabel("Boosting round")
axes[0, 0].set_ylabel(
    r"Weighted error, $\epsilon_t$"
)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()


axes[0, 1].plot(
    selected_round_log["round"],
    selected_round_log["alpha"],
    linewidth=1.8,
    color="tab:orange"
)

axes[0, 1].set_title(
    "SAMME learner weight"
)
axes[0, 1].set_xlabel("Boosting round")
axes[0, 1].set_ylabel(
    r"Learner weight, $\alpha_t$"
)
axes[0, 1].grid(alpha=0.3)


axes[1, 0].plot(
    selected_round_log["round"],
    selected_round_log[
        "maximum_sample_weight"
    ],
    linewidth=1.8,
    color="tab:red"
)

axes[1, 0].set_title(
    "Maximum training-sample weight"
)
axes[1, 0].set_xlabel("Boosting round")
axes[1, 0].set_ylabel("Maximum sample weight")
axes[1, 0].grid(alpha=0.3)


axes[1, 1].plot(
    selected_round_log["round"],
    selected_round_log[
        "top_10_weight_total"
    ],
    linewidth=1.8,
    color="tab:green"
)

axes[1, 1].set_title(
    "Weight carried by 10 heaviest faces"
)
axes[1, 1].set_xlabel("Boosting round")
axes[1, 1].set_ylabel(
    "Total of 10 largest weights"
)
axes[1, 1].grid(alpha=0.3)


figure.suptitle(
    (
        "Manual SAMME diagnostics: "
        f"depth={SELECTED_DEPTH}, "
        f"selected rounds={SELECTED_ROUNDS}"
    ),
    fontsize=13
)

figure.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.95]
)

figure.savefig(
    DIAGNOSTIC_FIGURE_FILE,
    bbox_inches="tight"
)

plt.close(figure)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("QUESTION 2(b): MODEL-SELECTION RESULTS")
print()

print(
    model_selection_results.to_string(
        index=False
    )
)

print()
print("Selected SAMME model:")
print("Depth:", SELECTED_DEPTH)
print("Rounds:", SELECTED_ROUNDS)
print(
    "Validation accuracy:",
    BEST_VALIDATION_ACCURACY
)

print(
    "Tie rule: highest validation accuracy, "
    "then fewer rounds, then smaller depth."
)

print()
print("Training stop reasons:")

for depth in DEPTH_VALUES:
    print(
        f"Depth {depth}: "
        f"{stop_reasons_by_depth[depth]}"
    )

print()
print("Saved:", MODEL_SELECTION_FILE)
print("Saved:", SELECTED_ROUND_LOG_FILE)
print("Saved:", SELECTED_MODEL_FILE)
print("Saved:", DIAGNOSTIC_FIGURE_FILE)