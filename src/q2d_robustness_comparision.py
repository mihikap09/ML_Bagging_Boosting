"""
Question 2(d): Robustness comparison on clean and masked faces.

The following frozen models are compared:

1. Single decision-tree baseline from Question 1(a).
2. Final manual bagging model selected in Question 1(c).
3. Final manual SAMME model selected in Question 2(b).

Models are trained only on clean training images and evaluated
on clean and centrally masked test images.

For each model and condition, the script reports accuracy and
mean winning probability or normalized weighted vote.

It also selects one face for which bagging and boosting respond
differently to the missing region and examines:

- bagging-tree disagreement;
- bagging vote distributions;
- boosting alpha-weighted votes;
- boosting training-sample weight concentration.
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
# Paths and constants
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

ROBUSTNESS_FILE = (
    PROJECT_ROOT / "results" / "q2d_robustness.csv"
)

SELECTED_FACE_FILE = (
    PROJECT_ROOT / "results" / "q2d_selected_face.csv"
)

BOOSTING_FACE_HISTORY_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2d_selected_face_boosting_history.csv"
)

SELECTED_FACE_FIGURE_FILE = (
    PROJECT_ROOT
    / "figures"
    / "q2d_selected_face_analysis.pdf"
)

SEED = 26004
N_CLASSES = 40

TREE_MAX_DEPTH = None
TREE_MIN_SAMPLES_LEAF = 4

RANDOM_GUESS_ERROR = 1.0 - 1.0 / N_CLASSES
ERROR_EPSILON = 1e-12


# ---------------------------------------------------------
# Load frozen model choices
# ---------------------------------------------------------

q1c_results = pd.read_csv(Q1C_RESULT_FILE)

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

print(
    "Final bagging size:",
    FINAL_BAGGING_SIZE
)

print(
    "Final boosting configuration:",
    f"depth={FINAL_BOOSTING_DEPTH},",
    f"rounds={FINAL_BOOSTING_ROUNDS}"
)


# ---------------------------------------------------------
# Load data and common split
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

X_test_clean = X[test_idx]
y_test = y[test_idx]

test_images_clean = images[test_idx].copy()


# ---------------------------------------------------------
# Verify split
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
    assert np.sum(y_test == class_label) == 2


# ---------------------------------------------------------
# Create centrally masked test set
# ---------------------------------------------------------

test_images_masked = test_images_clean.copy()

# 24:40 means pixel indices 24 through 39.
test_images_masked[
    :,
    24:40,
    24:40
] = 0.0

X_test_masked = test_images_masked.reshape(
    len(test_images_masked),
    -1
)

assert X_test_masked.shape == X_test_clean.shape
assert np.all(
    test_images_masked[:, 24:40, 24:40] == 0.0
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def vote_counts_from_predictions(
    prediction_matrix
):
    """
    Convert tree predictions into per-class vote counts.
    """
    number_of_observations = (
        prediction_matrix.shape[1]
    )

    vote_counts = np.zeros(
        (
            number_of_observations,
            N_CLASSES
        ),
        dtype=np.int64
    )

    for observation_index in range(
        number_of_observations
    ):
        vote_counts[
            observation_index
        ] = np.bincount(
            prediction_matrix[
                :,
                observation_index
            ],
            minlength=N_CLASSES
        )

    return vote_counts


def predictions_and_vote_confidence(
    vote_counts
):
    """
    Return majority prediction and winning vote fraction.
    """
    number_of_voters = np.sum(
        vote_counts,
        axis=1
    )

    predictions = np.argmax(
        vote_counts,
        axis=1
    ).astype(np.int64)

    confidence = (
        np.max(vote_counts, axis=1)
        / number_of_voters
    )

    return predictions, confidence


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


def predictions_and_weighted_confidence(
    score_matrix,
    total_alpha
):
    """
    Return SAMME predictions and normalized winning score.

        q_k(x) = S_k(x) / sum_t alpha_t
    """
    predictions = np.argmax(
        score_matrix,
        axis=1
    ).astype(np.int64)

    confidence = (
        np.max(score_matrix, axis=1)
        / total_alpha
    )

    return predictions, confidence


def pairwise_disagreement_for_face(
    tree_predictions
):
    """
    Fraction of tree pairs that disagree on one face.
    """
    number_of_trees = len(tree_predictions)

    if number_of_trees < 2:
        return np.nan

    class_counts = np.bincount(
        tree_predictions,
        minlength=N_CLASSES
    )

    total_pairs = (
        number_of_trees
        * (number_of_trees - 1)
        / 2
    )

    agreeing_pairs = np.sum(
        class_counts
        * (class_counts - 1)
        / 2
    )

    return float(
        (total_pairs - agreeing_pairs)
        / total_pairs
    )


def format_top_values(
    values,
    number_to_show=5
):
    """
    Format the largest class values as class:value pairs.
    """
    class_value_pairs = [
        (class_label, float(value))
        for class_label, value
        in enumerate(values)
        if value > 0
    ]

    class_value_pairs.sort(
        key=lambda item: (-item[1], item[0])
    )

    return ", ".join(
        f"{class_label}:{value:.4f}"
        for class_label, value
        in class_value_pairs[:number_to_show]
    )


def top_classes_and_values(
    values,
    number_to_show=8
):
    """
    Return class labels and values for a compact bar chart.
    """
    sorted_positions = np.argsort(
        values
    )[::-1]

    selected_positions = [
        int(position)
        for position in sorted_positions
        if values[position] > 0
    ][:number_to_show]

    selected_values = [
        float(values[position])
        for position in selected_positions
    ]

    return selected_positions, selected_values


# ---------------------------------------------------------
# Train frozen single-tree baseline
# ---------------------------------------------------------

single_tree = DecisionTreeClassifier(
    max_depth=TREE_MAX_DEPTH,
    min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
    random_state=SEED
)

single_start = time.perf_counter()

single_tree.fit(
    X_train,
    y_train
)

single_fit_time = (
    time.perf_counter() - single_start
)

single_clean_predictions = single_tree.predict(
    X_test_clean
).astype(np.int64)

single_masked_predictions = single_tree.predict(
    X_test_masked
).astype(np.int64)

single_clean_probabilities = single_tree.predict_proba(
    X_test_clean
)

single_masked_probabilities = single_tree.predict_proba(
    X_test_masked
)

single_clean_confidence = np.max(
    single_clean_probabilities,
    axis=1
)

single_masked_confidence = np.max(
    single_masked_probabilities,
    axis=1
)


# ---------------------------------------------------------
# Train frozen manual bagging model
# ---------------------------------------------------------

number_of_training_examples = len(X_train)

rng = np.random.default_rng(SEED)

bagging_clean_prediction_matrix = np.empty(
    (
        FINAL_BAGGING_SIZE,
        len(X_test_clean)
    ),
    dtype=np.int64
)

bagging_masked_prediction_matrix = np.empty(
    (
        FINAL_BAGGING_SIZE,
        len(X_test_masked)
    ),
    dtype=np.int64
)

bagging_start = time.perf_counter()

for tree_index in range(FINAL_BAGGING_SIZE):
    bootstrap_positions = rng.integers(
        low=0,
        high=number_of_training_examples,
        size=number_of_training_examples
    )

    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=SEED + tree_index
    )

    tree.fit(
        X_train[bootstrap_positions],
        y_train[bootstrap_positions]
    )

    bagging_clean_prediction_matrix[
        tree_index
    ] = tree.predict(
        X_test_clean
    )

    bagging_masked_prediction_matrix[
        tree_index
    ] = tree.predict(
        X_test_masked
    )

    if (
        tree_index == 0
        or (tree_index + 1) % 25 == 0
    ):
        print(
            f"Bagging: trained "
            f"{tree_index + 1}/"
            f"{FINAL_BAGGING_SIZE}"
        )

bagging_fit_time = (
    time.perf_counter() - bagging_start
)

bagging_clean_vote_counts = (
    vote_counts_from_predictions(
        bagging_clean_prediction_matrix
    )
)

bagging_masked_vote_counts = (
    vote_counts_from_predictions(
        bagging_masked_prediction_matrix
    )
)

(
    bagging_clean_predictions,
    bagging_clean_confidence
) = predictions_and_vote_confidence(
    bagging_clean_vote_counts
)

(
    bagging_masked_predictions,
    bagging_masked_confidence
) = predictions_and_vote_confidence(
    bagging_masked_vote_counts
)


# ---------------------------------------------------------
# Train frozen manual SAMME model
# ---------------------------------------------------------

sample_weights = np.full(
    number_of_training_examples,
    fill_value=(
        1.0 / number_of_training_examples
    ),
    dtype=np.float64
)

boosting_clean_scores = np.zeros(
    (
        len(X_test_clean),
        N_CLASSES
    ),
    dtype=np.float64
)

boosting_masked_scores = np.zeros(
    (
        len(X_test_masked),
        N_CLASSES
    ),
    dtype=np.float64
)

boosting_clean_prediction_matrix = np.empty(
    (
        FINAL_BOOSTING_ROUNDS,
        len(X_test_clean)
    ),
    dtype=np.int64
)

boosting_masked_prediction_matrix = np.empty(
    (
        FINAL_BOOSTING_ROUNDS,
        len(X_test_masked)
    ),
    dtype=np.int64
)

boosting_alphas = np.empty(
    FINAL_BOOSTING_ROUNDS,
    dtype=np.float64
)

maximum_training_weight_by_round = np.empty(
    FINAL_BOOSTING_ROUNDS,
    dtype=np.float64
)

top_10_training_weight_by_round = np.empty(
    FINAL_BOOSTING_ROUNDS,
    dtype=np.float64
)

boosting_start = time.perf_counter()

for round_index in range(
    FINAL_BOOSTING_ROUNDS
):
    round_number = round_index + 1

    weak_learner = DecisionTreeClassifier(
        max_depth=FINAL_BOOSTING_DEPTH,
        random_state=(
            SEED
            + FINAL_BOOSTING_DEPTH * 1000
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

    clean_predictions = weak_learner.predict(
        X_test_clean
    ).astype(np.int64)

    masked_predictions = weak_learner.predict(
        X_test_masked
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

    if weighted_error >= RANDOM_GUESS_ERROR:
        raise RuntimeError(
            "Frozen boosting model could not complete. "
            f"At round {round_number}, the weak learner "
            f"error {weighted_error:.6f} was not below "
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

    boosting_alphas[
        round_index
    ] = alpha

    boosting_clean_prediction_matrix[
        round_index
    ] = clean_predictions

    boosting_masked_prediction_matrix[
        round_index
    ] = masked_predictions

    add_weighted_votes(
        boosting_clean_scores,
        clean_predictions,
        alpha
    )

    add_weighted_votes(
        boosting_masked_scores,
        masked_predictions,
        alpha
    )

    sample_weights = (
        sample_weights
        * np.exp(
            alpha
            * incorrect.astype(np.float64)
        )
    )

    sample_weights = (
        sample_weights
        / np.sum(sample_weights)
    )

    sorted_weights = np.sort(
        sample_weights
    )

    maximum_training_weight_by_round[
        round_index
    ] = sorted_weights[-1]

    top_10_training_weight_by_round[
        round_index
    ] = np.sum(
        sorted_weights[-10:]
    )

    if (
        round_number == 1
        or round_number % 10 == 0
    ):
        print(
            f"Boosting: trained "
            f"{round_number}/"
            f"{FINAL_BOOSTING_ROUNDS}"
        )

boosting_fit_time = (
    time.perf_counter() - boosting_start
)

total_boosting_alpha = float(
    np.sum(boosting_alphas)
)

(
    boosting_clean_predictions,
    boosting_clean_confidence
) = predictions_and_weighted_confidence(
    boosting_clean_scores,
    total_boosting_alpha
)

(
    boosting_masked_predictions,
    boosting_masked_confidence
) = predictions_and_weighted_confidence(
    boosting_masked_scores,
    total_boosting_alpha
)


# ---------------------------------------------------------
# Build clean-versus-masked robustness table
# ---------------------------------------------------------

robustness_rows = [
    {
        "model": "single_tree",
        "test_condition": "clean",
        "accuracy": accuracy_score(
            y_test,
            single_clean_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    single_clean_confidence
                )
            ),
        "number_of_learners": 1,
        "fit_time_seconds": single_fit_time
    },
    {
        "model": "single_tree",
        "test_condition": "masked",
        "accuracy": accuracy_score(
            y_test,
            single_masked_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    single_masked_confidence
                )
            ),
        "number_of_learners": 1,
        "fit_time_seconds": single_fit_time
    },
    {
        "model": "manual_bagging",
        "test_condition": "clean",
        "accuracy": accuracy_score(
            y_test,
            bagging_clean_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    bagging_clean_confidence
                )
            ),
        "number_of_learners":
            FINAL_BAGGING_SIZE,
        "fit_time_seconds":
            bagging_fit_time
    },
    {
        "model": "manual_bagging",
        "test_condition": "masked",
        "accuracy": accuracy_score(
            y_test,
            bagging_masked_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    bagging_masked_confidence
                )
            ),
        "number_of_learners":
            FINAL_BAGGING_SIZE,
        "fit_time_seconds":
            bagging_fit_time
    },
    {
        "model": "manual_boosting",
        "test_condition": "clean",
        "accuracy": accuracy_score(
            y_test,
            boosting_clean_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    boosting_clean_confidence
                )
            ),
        "number_of_learners":
            FINAL_BOOSTING_ROUNDS,
        "fit_time_seconds":
            boosting_fit_time
    },
    {
        "model": "manual_boosting",
        "test_condition": "masked",
        "accuracy": accuracy_score(
            y_test,
            boosting_masked_predictions
        ),
        "mean_winning_probability_or_vote":
            float(
                np.mean(
                    boosting_masked_confidence
                )
            ),
        "number_of_learners":
            FINAL_BOOSTING_ROUNDS,
        "fit_time_seconds":
            boosting_fit_time
    }
]

robustness_results = pd.DataFrame(
    robustness_rows
)


# ---------------------------------------------------------
# Choose one face with a different response
# ---------------------------------------------------------

bagging_clean_correct = (
    bagging_clean_predictions == y_test
)

bagging_masked_correct = (
    bagging_masked_predictions == y_test
)

boosting_clean_correct = (
    boosting_clean_predictions == y_test
)

boosting_masked_correct = (
    boosting_masked_predictions == y_test
)

bagging_prediction_changed = (
    bagging_clean_predictions
    != bagging_masked_predictions
)

boosting_prediction_changed = (
    boosting_clean_predictions
    != boosting_masked_predictions
)

# Highest-priority case:
# both models are correct on the clean face, but after masking
# exactly one remains correct.
candidate_positions = np.flatnonzero(
    bagging_clean_correct
    & boosting_clean_correct
    & (
        bagging_masked_correct
        != boosting_masked_correct
    )
)

selection_reason = (
    "Both models were correct on the clean face, but "
    "only one remained correct after masking."
)

# Second priority:
# their correctness differs on the masked face.
if len(candidate_positions) == 0:
    candidate_positions = np.flatnonzero(
        bagging_masked_correct
        != boosting_masked_correct
    )

    selection_reason = (
        "Bagging and boosting had different correctness "
        "outcomes after masking."
    )

# Third priority:
# masking changes one model's prediction but not the other.
if len(candidate_positions) == 0:
    candidate_positions = np.flatnonzero(
        bagging_prediction_changed
        != boosting_prediction_changed
    )

    selection_reason = (
        "Masking changed one model's predicted class "
        "but did not change the other's."
    )

# Final deterministic fallback:
# choose the observation with the largest difference between
# the two models' confidence changes.
if len(candidate_positions) == 0:
    bagging_confidence_change = np.abs(
        bagging_masked_confidence
        - bagging_clean_confidence
    )

    boosting_confidence_change = np.abs(
        boosting_masked_confidence
        - boosting_clean_confidence
    )

    selected_test_position = int(
        np.argmax(
            np.abs(
                bagging_confidence_change
                - boosting_confidence_change
            )
        )
    )

    selection_reason = (
        "Selected by the largest difference between "
        "bagging and boosting confidence responses."
    )

else:
    selected_test_position = int(
        candidate_positions[0]
    )


# ---------------------------------------------------------
# Analyse the selected face
# ---------------------------------------------------------

selected_true_label = int(
    y_test[selected_test_position]
)

selected_original_index = int(
    test_idx[selected_test_position]
)

bagging_clean_disagreement = (
    pairwise_disagreement_for_face(
        bagging_clean_prediction_matrix[
            :,
            selected_test_position
        ]
    )
)

bagging_masked_disagreement = (
    pairwise_disagreement_for_face(
        bagging_masked_prediction_matrix[
            :,
            selected_test_position
        ]
    )
)

selected_face_summary = pd.DataFrame(
    [
        {
            "test_position":
                selected_test_position,
            "original_dataset_index":
                selected_original_index,
            "true_label":
                selected_true_label,
            "selection_reason":
                selection_reason,

            "single_clean_prediction":
                int(
                    single_clean_predictions[
                        selected_test_position
                    ]
                ),
            "single_masked_prediction":
                int(
                    single_masked_predictions[
                        selected_test_position
                    ]
                ),
            "single_clean_confidence":
                float(
                    single_clean_confidence[
                        selected_test_position
                    ]
                ),
            "single_masked_confidence":
                float(
                    single_masked_confidence[
                        selected_test_position
                    ]
                ),

            "bagging_clean_prediction":
                int(
                    bagging_clean_predictions[
                        selected_test_position
                    ]
                ),
            "bagging_masked_prediction":
                int(
                    bagging_masked_predictions[
                        selected_test_position
                    ]
                ),
            "bagging_clean_confidence":
                float(
                    bagging_clean_confidence[
                        selected_test_position
                    ]
                ),
            "bagging_masked_confidence":
                float(
                    bagging_masked_confidence[
                        selected_test_position
                    ]
                ),
            "bagging_clean_pairwise_disagreement":
                bagging_clean_disagreement,
            "bagging_masked_pairwise_disagreement":
                bagging_masked_disagreement,
            "bagging_clean_top_votes":
                format_top_values(
                    bagging_clean_vote_counts[
                        selected_test_position
                    ]
                ),
            "bagging_masked_top_votes":
                format_top_values(
                    bagging_masked_vote_counts[
                        selected_test_position
                    ]
                ),

            "boosting_clean_prediction":
                int(
                    boosting_clean_predictions[
                        selected_test_position
                    ]
                ),
            "boosting_masked_prediction":
                int(
                    boosting_masked_predictions[
                        selected_test_position
                    ]
                ),
            "boosting_clean_confidence":
                float(
                    boosting_clean_confidence[
                        selected_test_position
                    ]
                ),
            "boosting_masked_confidence":
                float(
                    boosting_masked_confidence[
                        selected_test_position
                    ]
                ),
            "boosting_clean_top_weighted_votes":
                format_top_values(
                    boosting_clean_scores[
                        selected_test_position
                    ]
                    / total_boosting_alpha
                ),
            "boosting_masked_top_weighted_votes":
                format_top_values(
                    boosting_masked_scores[
                        selected_test_position
                    ]
                    / total_boosting_alpha
                ),

            "boosting_final_max_training_weight":
                float(
                    maximum_training_weight_by_round[-1]
                ),
            "boosting_final_top_10_training_weight":
                float(
                    top_10_training_weight_by_round[-1]
                ),
            "boosting_peak_max_training_weight":
                float(
                    np.max(
                        maximum_training_weight_by_round
                    )
                ),
            "boosting_peak_top_10_training_weight":
                float(
                    np.max(
                        top_10_training_weight_by_round
                    )
                )
        }
    ]
)


# ---------------------------------------------------------
# Save selected-face boosting learner history
# ---------------------------------------------------------

selected_clean_weak_predictions = (
    boosting_clean_prediction_matrix[
        :,
        selected_test_position
    ]
)

selected_masked_weak_predictions = (
    boosting_masked_prediction_matrix[
        :,
        selected_test_position
    ]
)

boosting_face_history = pd.DataFrame(
    {
        "round": np.arange(
            1,
            FINAL_BOOSTING_ROUNDS + 1
        ),
        "alpha": boosting_alphas,
        "clean_weak_prediction":
            selected_clean_weak_predictions,
        "masked_weak_prediction":
            selected_masked_weak_predictions,
        "prediction_changed_by_mask":
            (
                selected_clean_weak_predictions
                != selected_masked_weak_predictions
            ).astype(np.int64),
        "maximum_training_sample_weight":
            maximum_training_weight_by_round,
        "top_10_training_weight_total":
            top_10_training_weight_by_round
    }
)


# ---------------------------------------------------------
# Save result tables
# ---------------------------------------------------------

ROBUSTNESS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

robustness_results.to_csv(
    ROBUSTNESS_FILE,
    index=False
)

selected_face_summary.to_csv(
    SELECTED_FACE_FILE,
    index=False
)

boosting_face_history.to_csv(
    BOOSTING_FACE_HISTORY_FILE,
    index=False
)


# ---------------------------------------------------------
# Create selected-face figure
# ---------------------------------------------------------

SELECTED_FACE_FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 9)
)

axes[0, 0].imshow(
    test_images_clean[
        selected_test_position
    ],
    cmap="gray"
)

axes[0, 0].set_title(
    "Clean test face\n"
    f"True label={selected_true_label}"
)

axes[0, 0].axis("off")


axes[1, 0].imshow(
    test_images_masked[
        selected_test_position
    ],
    cmap="gray"
)

axes[1, 0].set_title(
    "Masked test face\n"
    "Rows/columns 24–39 removed"
)

axes[1, 0].axis("off")


bag_clean_classes, bag_clean_values = (
    top_classes_and_values(
        bagging_clean_vote_counts[
            selected_test_position
        ]
    )
)

bag_masked_classes, bag_masked_values = (
    top_classes_and_values(
        bagging_masked_vote_counts[
            selected_test_position
        ]
    )
)

axes[0, 1].bar(
    [str(value) for value in bag_clean_classes],
    bag_clean_values,
    color="tab:blue"
)

axes[0, 1].set_title(
    "Bagging votes: clean\n"
    f"Disagreement={bagging_clean_disagreement:.3f}"
)
axes[0, 1].set_xlabel("Predicted class")
axes[0, 1].set_ylabel("Number of tree votes")
axes[0, 1].grid(axis="y", alpha=0.3)


axes[1, 1].bar(
    [str(value) for value in bag_masked_classes],
    bag_masked_values,
    color="tab:orange"
)

axes[1, 1].set_title(
    "Bagging votes: masked\n"
    f"Disagreement={bagging_masked_disagreement:.3f}"
)
axes[1, 1].set_xlabel("Predicted class")
axes[1, 1].set_ylabel("Number of tree votes")
axes[1, 1].grid(axis="y", alpha=0.3)


boost_clean_normalized_scores = (
    boosting_clean_scores[
        selected_test_position
    ]
    / total_boosting_alpha
)

boost_masked_normalized_scores = (
    boosting_masked_scores[
        selected_test_position
    ]
    / total_boosting_alpha
)

boost_clean_classes, boost_clean_values = (
    top_classes_and_values(
        boost_clean_normalized_scores
    )
)

boost_masked_classes, boost_masked_values = (
    top_classes_and_values(
        boost_masked_normalized_scores
    )
)

axes[0, 2].bar(
    [str(value) for value in boost_clean_classes],
    boost_clean_values,
    color="tab:green"
)

axes[0, 2].set_title(
    "Boosting weighted votes: clean"
)
axes[0, 2].set_xlabel("Predicted class")
axes[0, 2].set_ylabel("Normalized weighted vote")
axes[0, 2].grid(axis="y", alpha=0.3)


axes[1, 2].bar(
    [str(value) for value in boost_masked_classes],
    boost_masked_values,
    color="tab:red"
)

axes[1, 2].set_title(
    "Boosting weighted votes: masked"
)
axes[1, 2].set_xlabel("Predicted class")
axes[1, 2].set_ylabel("Normalized weighted vote")
axes[1, 2].grid(axis="y", alpha=0.3)


figure.suptitle(
    (
        "Different bagging and boosting responses "
        "to the missing facial region"
    ),
    fontsize=13
)

figure.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.95]
)

figure.savefig(
    SELECTED_FACE_FIGURE_FILE,
    bbox_inches="tight"
)

plt.close(figure)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("QUESTION 2(d): ROBUSTNESS RESULTS")
print()

print(
    robustness_results.to_string(
        index=False
    )
)

print()
print("Selected face analysis:")
print(
    selected_face_summary.to_string(
        index=False
    )
)

print()
print(
    "Interpretation guidance: bagging disagreement "
    "shows how consistently its independently "
    "bootstrapped trees respond to the mask. Boosting "
    "does not count all learners equally; its decision "
    "is dominated by alpha-weighted votes from learners "
    "trained after difficult training observations "
    "received increased sample weight."
)

print()
print("Saved:", ROBUSTNESS_FILE)
print("Saved:", SELECTED_FACE_FILE)
print("Saved:", BOOSTING_FACE_HISTORY_FILE)
print("Saved:", SELECTED_FACE_FIGURE_FILE)