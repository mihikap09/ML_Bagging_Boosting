"""
Question 2(c): Compare manual SAMME boosting and manual bagging.

Both methods are evaluated using 25, 50 and 100 learners.
The comparison reports:

1. Training accuracy.
2. Validation accuracy.
3. Clean-test accuracy.
4. Cumulative training time.

The script also identifies one training face that receives a
very large boosting weight and records its complete prediction
and weight history.

Neither sklearn AdaBoostClassifier nor BaggingClassifier is used.
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

Q2B_SELECTED_MODEL_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2b_selected_model.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2c_boosting_vs_bagging.csv"
)

FACE_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2c_heavy_face_summary.csv"
)

FACE_HISTORY_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2c_heavy_face_history.csv"
)

FACE_FIGURE_FILE = (
    PROJECT_ROOT
    / "figures"
    / "q2c_heavy_face_analysis.pdf"
)

SEED = 26004
N_CLASSES = 40

ENSEMBLE_SIZES = [25, 50, 100]
MAX_LEARNERS = max(ENSEMBLE_SIZES)

# Frozen bagging-tree configuration from Question 1(a).
BAGGING_MAX_DEPTH = None
BAGGING_MIN_SAMPLES_LEAF = 4

RANDOM_GUESS_ERROR = 1.0 - 1.0 / N_CLASSES
ERROR_EPSILON = 1e-12


# ---------------------------------------------------------
# Load the selected boosting depth from Question 2(b)
# ---------------------------------------------------------

q2b_selected_model = pd.read_csv(
    Q2B_SELECTED_MODEL_FILE
)

required_columns = {
    "selected_depth",
    "selected_rounds",
    "validation_accuracy"
}

assert required_columns.issubset(
    q2b_selected_model.columns
)

BOOSTING_DEPTH = int(
    q2b_selected_model.loc[
        0,
        "selected_depth"
    ]
)

Q2B_SELECTED_ROUNDS = int(
    q2b_selected_model.loc[
        0,
        "selected_rounds"
    ]
)

print("Boosting depth loaded from Q2(b):", BOOSTING_DEPTH)
print(
    "Q2(b) selected rounds:",
    Q2B_SELECTED_ROUNDS
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

images = np.asarray(
    faces.images,
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

X_test = X[test_idx]
y_test = y[test_idx]

train_images = images[train_idx]


# ---------------------------------------------------------
# Verify the common split
# ---------------------------------------------------------

assert X.shape == (400, 4096)
assert images.shape == (400, 64, 64)

assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

for class_label in range(N_CLASSES):
    assert np.sum(y_train == class_label) == 6
    assert np.sum(y_val == class_label) == 2
    assert np.sum(y_test == class_label) == 2


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def add_weighted_votes(
    score_matrix,
    predictions,
    alpha
):
    """
    Add alpha to each observation's predicted-class score.
    """
    observation_positions = np.arange(
        len(predictions)
    )

    score_matrix[
        observation_positions,
        predictions
    ] += alpha


def score_predictions(score_matrix):
    """
    Convert SAMME class scores to predicted classes.

    np.argmax breaks score ties in favour of the smaller
    class label.
    """
    return np.argmax(
        score_matrix,
        axis=1
    ).astype(np.int64)


def majority_vote(prediction_matrix):
    """
    Return manual multiclass majority-vote predictions.

    Input shape:
        number of trees x number of observations
    """
    number_of_observations = (
        prediction_matrix.shape[1]
    )

    ensemble_predictions = np.empty(
        number_of_observations,
        dtype=np.int64
    )

    for observation_index in range(
        number_of_observations
    ):
        vote_counts = np.bincount(
            prediction_matrix[
                :,
                observation_index
            ],
            minlength=N_CLASSES
        )

        ensemble_predictions[
            observation_index
        ] = np.argmax(vote_counts)

    return ensemble_predictions


# ---------------------------------------------------------
# Train manual SAMME boosting
# ---------------------------------------------------------

number_of_training_examples = len(X_train)

sample_weights = np.full(
    number_of_training_examples,
    fill_value=(
        1.0 / number_of_training_examples
    ),
    dtype=np.float64
)

boosting_train_scores = np.zeros(
    (len(X_train), N_CLASSES),
    dtype=np.float64
)

boosting_val_scores = np.zeros(
    (len(X_val), N_CLASSES),
    dtype=np.float64
)

boosting_test_scores = np.zeros(
    (len(X_test), N_CLASSES),
    dtype=np.float64
)

# Per-round diagnostic histories.
boosting_weight_before_history = np.empty(
    (
        MAX_LEARNERS,
        number_of_training_examples
    ),
    dtype=np.float64
)

boosting_weight_after_history = np.empty(
    (
        MAX_LEARNERS,
        number_of_training_examples
    ),
    dtype=np.float64
)

boosting_training_prediction_history = np.empty(
    (
        MAX_LEARNERS,
        number_of_training_examples
    ),
    dtype=np.int64
)

boosting_incorrect_history = np.empty(
    (
        MAX_LEARNERS,
        number_of_training_examples
    ),
    dtype=bool
)

boosting_errors = np.empty(
    MAX_LEARNERS,
    dtype=np.float64
)

boosting_alphas = np.empty(
    MAX_LEARNERS,
    dtype=np.float64
)

boosting_fit_times = np.empty(
    MAX_LEARNERS,
    dtype=np.float64
)

boosting_rows = []

for round_index in range(MAX_LEARNERS):
    round_number = round_index + 1

    boosting_weight_before_history[
        round_index
    ] = sample_weights

    weak_learner = DecisionTreeClassifier(
        max_depth=BOOSTING_DEPTH,
        random_state=(
            SEED
            + BOOSTING_DEPTH * 1000
            + round_index
        )
    )

    fit_start = time.perf_counter()

    weak_learner.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    boosting_fit_times[
        round_index
    ] = time.perf_counter() - fit_start

    train_predictions = weak_learner.predict(
        X_train
    ).astype(np.int64)

    val_predictions = weak_learner.predict(
        X_val
    ).astype(np.int64)

    test_predictions = weak_learner.predict(
        X_test
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

    # Explicit SAMME weak-learner validity check.
    if weighted_error >= RANDOM_GUESS_ERROR:
        raise RuntimeError(
            "Boosting stopped before the required "
            f"{MAX_LEARNERS} rounds. At round "
            f"{round_number}, error={weighted_error:.6f} "
            "was not better than the random threshold "
            f"{RANDOM_GUESS_ERROR:.6f}."
        )

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

    boosting_training_prediction_history[
        round_index
    ] = train_predictions

    boosting_incorrect_history[
        round_index
    ] = incorrect

    boosting_errors[
        round_index
    ] = weighted_error

    boosting_alphas[
        round_index
    ] = alpha

    add_weighted_votes(
        boosting_train_scores,
        train_predictions,
        alpha
    )

    add_weighted_votes(
        boosting_val_scores,
        val_predictions,
        alpha
    )

    add_weighted_votes(
        boosting_test_scores,
        test_predictions,
        alpha
    )

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
        not np.isfinite(normalization_constant)
        or normalization_constant <= 0.0
    ):
        raise RuntimeError(
            "Invalid boosting weight normalization at "
            f"round {round_number}."
        )

    sample_weights = (
        sample_weights
        / normalization_constant
    )

    assert np.isclose(
        np.sum(sample_weights),
        1.0
    )

    boosting_weight_after_history[
        round_index
    ] = sample_weights

    if round_number in ENSEMBLE_SIZES:
        boosting_train_predictions = (
            score_predictions(
                boosting_train_scores
            )
        )

        boosting_val_predictions = (
            score_predictions(
                boosting_val_scores
            )
        )

        boosting_test_predictions = (
            score_predictions(
                boosting_test_scores
            )
        )

        boosting_rows.append(
            {
                "method": "manual_boosting",
                "number_of_learners":
                    round_number,
                "tree_depth":
                    BOOSTING_DEPTH,
                "min_samples_leaf":
                    1,
                "train_accuracy":
                    accuracy_score(
                        y_train,
                        boosting_train_predictions
                    ),
                "validation_accuracy":
                    accuracy_score(
                        y_val,
                        boosting_val_predictions
                    ),
                "clean_test_accuracy":
                    accuracy_score(
                        y_test,
                        boosting_test_predictions
                    ),
                "training_time_seconds":
                    float(
                        np.sum(
                            boosting_fit_times[
                                :round_number
                            ]
                        )
                    )
            }
        )

    if (
        round_number == 1
        or round_number % 10 == 0
    ):
        print(
            f"Boosting: completed "
            f"{round_number}/{MAX_LEARNERS}"
        )


# ---------------------------------------------------------
# Train manual bagging
# ---------------------------------------------------------

rng = np.random.default_rng(SEED)

bagging_train_prediction_matrix = np.empty(
    (
        MAX_LEARNERS,
        len(X_train)
    ),
    dtype=np.int64
)

bagging_val_prediction_matrix = np.empty(
    (
        MAX_LEARNERS,
        len(X_val)
    ),
    dtype=np.int64
)

bagging_test_prediction_matrix = np.empty(
    (
        MAX_LEARNERS,
        len(X_test)
    ),
    dtype=np.int64
)

bagging_fit_times = np.empty(
    MAX_LEARNERS,
    dtype=np.float64
)

bagging_rows = []

for tree_index in range(MAX_LEARNERS):
    tree_number = tree_index + 1

    bootstrap_positions = rng.integers(
        low=0,
        high=number_of_training_examples,
        size=number_of_training_examples
    )

    bagging_tree = DecisionTreeClassifier(
        max_depth=BAGGING_MAX_DEPTH,
        min_samples_leaf=(
            BAGGING_MIN_SAMPLES_LEAF
        ),
        random_state=SEED + tree_index
    )

    fit_start = time.perf_counter()

    bagging_tree.fit(
        X_train[bootstrap_positions],
        y_train[bootstrap_positions]
    )

    bagging_fit_times[
        tree_index
    ] = time.perf_counter() - fit_start

    bagging_train_prediction_matrix[
        tree_index
    ] = bagging_tree.predict(
        X_train
    )

    bagging_val_prediction_matrix[
        tree_index
    ] = bagging_tree.predict(
        X_val
    )

    bagging_test_prediction_matrix[
        tree_index
    ] = bagging_tree.predict(
        X_test
    )

    if tree_number in ENSEMBLE_SIZES:
        bagging_train_predictions = majority_vote(
            bagging_train_prediction_matrix[
                :tree_number
            ]
        )

        bagging_val_predictions = majority_vote(
            bagging_val_prediction_matrix[
                :tree_number
            ]
        )

        bagging_test_predictions = majority_vote(
            bagging_test_prediction_matrix[
                :tree_number
            ]
        )

        bagging_rows.append(
            {
                "method": "manual_bagging",
                "number_of_learners":
                    tree_number,
                "tree_depth":
                    BAGGING_MAX_DEPTH,
                "min_samples_leaf":
                    BAGGING_MIN_SAMPLES_LEAF,
                "train_accuracy":
                    accuracy_score(
                        y_train,
                        bagging_train_predictions
                    ),
                "validation_accuracy":
                    accuracy_score(
                        y_val,
                        bagging_val_predictions
                    ),
                "clean_test_accuracy":
                    accuracy_score(
                        y_test,
                        bagging_test_predictions
                    ),
                "training_time_seconds":
                    float(
                        np.sum(
                            bagging_fit_times[
                                :tree_number
                            ]
                        )
                    )
            }
        )

    if (
        tree_number == 1
        or tree_number % 10 == 0
    ):
        print(
            f"Bagging: completed "
            f"{tree_number}/{MAX_LEARNERS}"
        )


# ---------------------------------------------------------
# Build and save the method-comparison table
# ---------------------------------------------------------

comparison_results = pd.DataFrame(
    boosting_rows + bagging_rows
)

comparison_results = (
    comparison_results
    .sort_values(
        by=[
            "number_of_learners",
            "method"
        ]
    )
    .reset_index(drop=True)
)

COMPARISON_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

comparison_results.to_csv(
    COMPARISON_FILE,
    index=False
)


# ---------------------------------------------------------
# Select one heavily weighted boosting face
# ---------------------------------------------------------

# Find the single largest post-update weight reached by any
# training image during the 100 boosting rounds.
largest_weight_flat_position = np.argmax(
    boosting_weight_after_history
)

(
    largest_weight_round_index,
    heavy_face_train_position
) = np.unravel_index(
    largest_weight_flat_position,
    boosting_weight_after_history.shape
)

largest_weight_round = (
    largest_weight_round_index + 1
)

heavy_face_train_position = int(
    heavy_face_train_position
)

heavy_face_original_index = int(
    train_idx[heavy_face_train_position]
)

heavy_face_true_label = int(
    y_train[heavy_face_train_position]
)

heavy_face_predictions = (
    boosting_training_prediction_history[
        :,
        heavy_face_train_position
    ]
)

heavy_face_incorrect = (
    boosting_incorrect_history[
        :,
        heavy_face_train_position
    ]
)

heavy_face_weight_before = (
    boosting_weight_before_history[
        :,
        heavy_face_train_position
    ]
)

heavy_face_weight_after = (
    boosting_weight_after_history[
        :,
        heavy_face_train_position
    ]
)

heavy_face_history = pd.DataFrame(
    {
        "round": np.arange(
            1,
            MAX_LEARNERS + 1
        ),
        "true_label":
            heavy_face_true_label,
        "weak_learner_prediction":
            heavy_face_predictions,
        "incorrect":
            heavy_face_incorrect.astype(
                np.int64
            ),
        "weight_before":
            heavy_face_weight_before,
        "weighted_training_error":
            boosting_errors,
        "alpha":
            boosting_alphas,
        "weight_after":
            heavy_face_weight_after
    }
)

number_of_misclassifications = int(
    np.sum(heavy_face_incorrect)
)

number_of_correct_predictions = int(
    MAX_LEARNERS
    - number_of_misclassifications
)

top_10_membership_by_round = (
    np.argsort(
        boosting_weight_after_history,
        axis=1
    )[:, -10:]
)

number_of_top_10_appearances = int(
    np.sum(
        top_10_membership_by_round
        == heavy_face_train_position
    )
)

heavy_face_summary = pd.DataFrame(
    [
        {
            "train_position":
                heavy_face_train_position,
            "original_dataset_index":
                heavy_face_original_index,
            "true_label":
                heavy_face_true_label,
            "largest_weight":
                float(
                    heavy_face_weight_after[
                        largest_weight_round_index
                    ]
                ),
            "round_of_largest_weight":
                largest_weight_round,
            "initial_weight":
                float(
                    1.0
                    / number_of_training_examples
                ),
            "final_weight":
                float(
                    heavy_face_weight_after[-1]
                ),
            "number_of_misclassifications":
                number_of_misclassifications,
            "number_of_correct_predictions":
                number_of_correct_predictions,
            "misclassification_fraction":
                (
                    number_of_misclassifications
                    / MAX_LEARNERS
                ),
            "number_of_top_10_weight_appearances":
                number_of_top_10_appearances
        }
    ]
)

heavy_face_history.to_csv(
    FACE_HISTORY_FILE,
    index=False
)

heavy_face_summary.to_csv(
    FACE_SUMMARY_FILE,
    index=False
)


# ---------------------------------------------------------
# Plot the face and its prediction/weight history
# ---------------------------------------------------------

FACE_FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 4.5)
)

axes[0].imshow(
    train_images[
        heavy_face_train_position
    ],
    cmap="gray"
)

axes[0].set_title(
    "Heavily weighted training face\n"
    f"True label={heavy_face_true_label}, "
    f"dataset index={heavy_face_original_index}"
)

axes[0].axis("off")


axes[1].plot(
    heavy_face_history["round"],
    heavy_face_history["weight_after"],
    linewidth=1.8,
    color="tab:red",
    label="Weight after update"
)

axes[1].axhline(
    1.0 / number_of_training_examples,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Initial uniform weight"
)

axes[1].axvline(
    largest_weight_round,
    color="tab:blue",
    linestyle=":",
    linewidth=1.3,
    label=(
        "Largest weight at round "
        f"{largest_weight_round}"
    )
)

axes[1].set_title(
    "Boosting sample-weight history"
)
axes[1].set_xlabel("Boosting round")
axes[1].set_ylabel("Sample weight")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)


incorrect_rounds = heavy_face_history.loc[
    heavy_face_history["incorrect"] == 1,
    "round"
]

correct_rounds = heavy_face_history.loc[
    heavy_face_history["incorrect"] == 0,
    "round"
]

axes[2].scatter(
    correct_rounds,
    np.zeros(len(correct_rounds)),
    color="tab:green",
    s=22,
    label="Correct"
)

axes[2].scatter(
    incorrect_rounds,
    np.ones(len(incorrect_rounds)),
    color="tab:red",
    s=22,
    label="Incorrect"
)

axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(
    ["Correct", "Incorrect"]
)

axes[2].set_ylim(-0.4, 1.4)
axes[2].set_xlabel("Boosting round")
axes[2].set_title(
    "Weak-learner prediction history"
)
axes[2].grid(
    axis="x",
    alpha=0.3
)
axes[2].legend(fontsize=8)


figure.suptitle(
    (
        "Why boosting repeatedly returns attention "
        "to one difficult training face"
    ),
    fontsize=13
)

figure.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.93]
)

figure.savefig(
    FACE_FIGURE_FILE,
    bbox_inches="tight"
)

plt.close(figure)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("QUESTION 2(c): BOOSTING VERSUS BAGGING")
print()

print(
    comparison_results.to_string(
        index=False
    )
)

print()
print("Heavily weighted training face:")
print(
    heavy_face_summary.to_string(
        index=False
    )
)

print()
print(
    "Interpretation: whenever this face is "
    "misclassified, SAMME multiplies its weight by "
    "exp(alpha). Correct predictions do not directly "
    "increase its weight. Repeated mistakes therefore "
    "make the face increasingly influential in later "
    "weak-learner training rounds."
)

print()
print("Saved:", COMPARISON_FILE)
print("Saved:", FACE_SUMMARY_FILE)
print("Saved:", FACE_HISTORY_FILE)
print("Saved:", FACE_FIGURE_FILE)