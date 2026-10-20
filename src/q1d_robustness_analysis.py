"""
Question 1(d): Robustness analysis.

The frozen single decision tree and final manually
implemented bagging ensemble are trained only on clean
Olivetti faces.

They are evaluated on:

1. The original clean test set.
2. A masked test set in which the centered 16 x 16 block,
   rows 24--39 and columns 24--39, is set to zero.

The script also finds:
- two clean-test faces corrected by bagging;
- two clean-test faces misclassified by both models.

Bagging is implemented manually. sklearn's
BaggingClassifier is not used.
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

Q1C_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q1c_diagnostics.csv"
)

ROBUSTNESS_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q1d_robustness.csv"
)

EXAMPLE_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q1d_examples.csv"
)

EXAMPLE_FIGURE_FILE = (
    PROJECT_ROOT / "figures" / "q1d_examples.pdf"
)

SEED = 26004
N_CLASSES = 40

# Frozen Q1(a) tree configuration.
TREE_MAX_DEPTH = None
TREE_MIN_SAMPLES_LEAF = 4


# ---------------------------------------------------------
# Load the selected B from Question 1(c)
# ---------------------------------------------------------

q1c_results = pd.read_csv(Q1C_RESULT_FILE)

required_q1c_columns = {
    "B",
    "validation_accuracy"
}

assert required_q1c_columns.issubset(
    q1c_results.columns
)

# Select the greatest validation accuracy.
# If tied, select the smaller ensemble.
selected_row = (
    q1c_results
    .sort_values(
        by=["validation_accuracy", "B"],
        ascending=[False, True]
    )
    .iloc[0]
)

FINAL_B = int(selected_row["B"])
BEST_VALIDATION_ACCURACY = float(
    selected_row["validation_accuracy"]
)

print("Final B loaded from Q1(c):", FINAL_B)
print(
    "Selection validation accuracy:",
    BEST_VALIDATION_ACCURACY
)


# ---------------------------------------------------------
# Load Olivetti faces and the common saved split
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

X_test_clean = X[test_idx]
y_test = y[test_idx]

test_images_clean = images[test_idx].copy()


# ---------------------------------------------------------
# Verify the common assignment split
# ---------------------------------------------------------

assert X.shape == (400, 4096)
assert images.shape == (400, 64, 64)

assert len(train_idx) == 240
assert len(val_idx) == 80
assert len(test_idx) == 80

assert len(
    set(train_idx) & set(val_idx)
) == 0

assert len(
    set(train_idx) & set(test_idx)
) == 0

assert len(
    set(val_idx) & set(test_idx)
) == 0

for class_label in range(N_CLASSES):
    assert np.sum(y_train == class_label) == 6
    assert np.sum(y_val == class_label) == 2
    assert np.sum(y_test == class_label) == 2


# ---------------------------------------------------------
# Construct the masked test set
# ---------------------------------------------------------

test_images_masked = test_images_clean.copy()

# Python slicing excludes the upper endpoint. Therefore,
# 24:40 corresponds exactly to indices 24, ..., 39.
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

# Verify that exactly the required block was zeroed.
assert np.all(
    test_images_masked[:, 24:40, 24:40] == 0.0
)


# ---------------------------------------------------------
# Manual multiclass vote handling
# ---------------------------------------------------------

def vote_counts_from_predictions(
    prediction_matrix,
    n_classes
):
    """
    Convert individual tree predictions into vote counts.

    Input shape:
        number of trees x number of observations

    Output shape:
        number of observations x number of classes
    """
    number_of_observations = (
        prediction_matrix.shape[1]
    )

    vote_counts = np.zeros(
        (number_of_observations, n_classes),
        dtype=np.int64
    )

    for observation_index in range(
        number_of_observations
    ):
        vote_counts[observation_index] = np.bincount(
            prediction_matrix[:, observation_index],
            minlength=n_classes
        )

    return vote_counts


def predictions_and_confidence(vote_counts):
    """
    Return the majority-vote prediction and winning vote
    fraction for every observation.

    np.argmax deterministically selects the smaller class
    label when classes receive equal vote counts.
    """
    number_of_trees = np.sum(
        vote_counts,
        axis=1
    )

    predictions = np.argmax(
        vote_counts,
        axis=1
    ).astype(np.int64)

    confidence = (
        np.max(vote_counts, axis=1)
        / number_of_trees
    )

    return predictions, confidence


def format_nonzero_votes(vote_count_vector):
    """
    Produce a readable list of all classes receiving at
    least one vote, sorted by decreasing vote count and
    then by increasing class label.
    """
    vote_items = []

    for class_label, count in enumerate(
        vote_count_vector
    ):
        if count > 0:
            vote_items.append(
                (class_label, int(count))
            )

    vote_items.sort(
        key=lambda item: (-item[1], item[0])
    )

    return ", ".join(
        f"{class_label}:{count}"
        for class_label, count in vote_items
    )


def format_top_votes(
    vote_count_vector,
    number_to_show=3
):
    """
    Short vote summary for figure titles.
    """
    vote_items = [
        (class_label, int(count))
        for class_label, count
        in enumerate(vote_count_vector)
        if count > 0
    ]

    vote_items.sort(
        key=lambda item: (-item[1], item[0])
    )

    return ", ".join(
        f"{class_label}:{count}"
        for class_label, count
        in vote_items[:number_to_show]
    )


# ---------------------------------------------------------
# Train the frozen single decision tree
# ---------------------------------------------------------

single_tree = DecisionTreeClassifier(
    max_depth=TREE_MAX_DEPTH,
    min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
    random_state=SEED
)

single_tree_start = time.perf_counter()

single_tree.fit(
    X_train,
    y_train
)

single_tree_fit_time = (
    time.perf_counter() - single_tree_start
)

single_clean_predictions = single_tree.predict(
    X_test_clean
).astype(np.int64)

single_masked_predictions = single_tree.predict(
    X_test_masked
).astype(np.int64)

# Decision-tree confidence is the maximum estimated class
# probability in the reached leaf.
single_clean_probabilities = (
    single_tree.predict_proba(X_test_clean)
)

single_masked_probabilities = (
    single_tree.predict_proba(X_test_masked)
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
# Train the final manual bagging ensemble
# ---------------------------------------------------------

number_of_training_examples = len(X_train)
rng = np.random.default_rng(SEED)

bagging_clean_prediction_matrix = np.empty(
    (FINAL_B, len(X_test_clean)),
    dtype=np.int64
)

bagging_masked_prediction_matrix = np.empty(
    (FINAL_B, len(X_test_masked)),
    dtype=np.int64
)

bagging_start = time.perf_counter()

for tree_index in range(FINAL_B):
    # Draw n training positions with replacement.
    bootstrap_positions = rng.integers(
        low=0,
        high=number_of_training_examples,
        size=number_of_training_examples
    )

    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=(
            TREE_MIN_SAMPLES_LEAF
        ),
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
            f"Trained {tree_index + 1}/"
            f"{FINAL_B} bagging trees"
        )

bagging_fit_time = (
    time.perf_counter() - bagging_start
)


# ---------------------------------------------------------
# Obtain ensemble predictions and confidence
# ---------------------------------------------------------

bagging_clean_vote_counts = (
    vote_counts_from_predictions(
        bagging_clean_prediction_matrix,
        N_CLASSES
    )
)

bagging_masked_vote_counts = (
    vote_counts_from_predictions(
        bagging_masked_prediction_matrix,
        N_CLASSES
    )
)

(
    bagging_clean_predictions,
    bagging_clean_confidence
) = predictions_and_confidence(
    bagging_clean_vote_counts
)

(
    bagging_masked_predictions,
    bagging_masked_confidence
) = predictions_and_confidence(
    bagging_masked_vote_counts
)


# ---------------------------------------------------------
# Compute clean and masked robustness results
# ---------------------------------------------------------

robustness_rows = [
    {
        "model": "single_tree",
        "test_condition": "clean",
        "accuracy": accuracy_score(
            y_test,
            single_clean_predictions
        ),
        "mean_prediction_confidence": float(
            np.mean(single_clean_confidence)
        ),
        "number_of_estimators": 1,
        "fit_time_seconds":
            single_tree_fit_time
    },
    {
        "model": "single_tree",
        "test_condition": "masked",
        "accuracy": accuracy_score(
            y_test,
            single_masked_predictions
        ),
        "mean_prediction_confidence": float(
            np.mean(single_masked_confidence)
        ),
        "number_of_estimators": 1,
        "fit_time_seconds":
            single_tree_fit_time
    },
    {
        "model": "manual_bagging",
        "test_condition": "clean",
        "accuracy": accuracy_score(
            y_test,
            bagging_clean_predictions
        ),
        "mean_prediction_confidence": float(
            np.mean(bagging_clean_confidence)
        ),
        "number_of_estimators": FINAL_B,
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
        "mean_prediction_confidence": float(
            np.mean(bagging_masked_confidence)
        ),
        "number_of_estimators": FINAL_B,
        "fit_time_seconds":
            bagging_fit_time
    }
]

robustness_results = pd.DataFrame(
    robustness_rows
)


# ---------------------------------------------------------
# Find the requested clean-test examples
# ---------------------------------------------------------

tree_clean_correct = (
    single_clean_predictions == y_test
)

bagging_clean_correct = (
    bagging_clean_predictions == y_test
)

# Single tree is wrong, but bagging is correct.
bagging_corrected_positions = np.flatnonzero(
    (~tree_clean_correct)
    & bagging_clean_correct
)

# Both single tree and bagging are wrong.
both_wrong_positions = np.flatnonzero(
    (~tree_clean_correct)
    & (~bagging_clean_correct)
)

if len(bagging_corrected_positions) < 2:
    raise RuntimeError(
        "Fewer than two clean-test examples were "
        "corrected by bagging."
    )

if len(both_wrong_positions) < 2:
    raise RuntimeError(
        "Fewer than two clean-test examples were "
        "misclassified by both models."
    )

selected_corrected_positions = (
    bagging_corrected_positions[:2]
)

selected_both_wrong_positions = (
    both_wrong_positions[:2]
)

selected_examples = [
    (
        "bagging_corrected",
        int(selected_corrected_positions[0])
    ),
    (
        "bagging_corrected",
        int(selected_corrected_positions[1])
    ),
    (
        "both_wrong",
        int(selected_both_wrong_positions[0])
    ),
    (
        "both_wrong",
        int(selected_both_wrong_positions[1])
    )
]


# ---------------------------------------------------------
# Build the example table
# ---------------------------------------------------------

example_rows = []

for category, test_position in selected_examples:
    example_rows.append(
        {
            "category": category,
            "test_position": test_position,
            "original_dataset_index": int(
                test_idx[test_position]
            ),
            "true_label": int(
                y_test[test_position]
            ),
            "clean_tree_prediction": int(
                single_clean_predictions[
                    test_position
                ]
            ),
            "clean_bagging_prediction": int(
                bagging_clean_predictions[
                    test_position
                ]
            ),
            "clean_bagging_confidence": float(
                bagging_clean_confidence[
                    test_position
                ]
            ),
            "clean_bagging_votes":
                format_nonzero_votes(
                    bagging_clean_vote_counts[
                        test_position
                    ]
                ),
            "masked_tree_prediction": int(
                single_masked_predictions[
                    test_position
                ]
            ),
            "masked_bagging_prediction": int(
                bagging_masked_predictions[
                    test_position
                ]
            ),
            "masked_bagging_confidence": float(
                bagging_masked_confidence[
                    test_position
                ]
            ),
            "masked_bagging_votes":
                format_nonzero_votes(
                    bagging_masked_vote_counts[
                        test_position
                    ]
                )
        }
    )

example_results = pd.DataFrame(example_rows)


# ---------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------

ROBUSTNESS_RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

robustness_results.to_csv(
    ROBUSTNESS_RESULT_FILE,
    index=False
)

example_results.to_csv(
    EXAMPLE_RESULT_FILE,
    index=False
)


# ---------------------------------------------------------
# Create the requested example figure
# ---------------------------------------------------------

EXAMPLE_FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(15, 7)
)

for column_index, (
    category,
    test_position
) in enumerate(selected_examples):
    true_label = int(y_test[test_position])

    clean_tree_prediction = int(
        single_clean_predictions[test_position]
    )

    clean_bagging_prediction = int(
        bagging_clean_predictions[test_position]
    )

    masked_tree_prediction = int(
        single_masked_predictions[test_position]
    )

    masked_bagging_prediction = int(
        bagging_masked_predictions[test_position]
    )

    clean_top_votes = format_top_votes(
        bagging_clean_vote_counts[test_position]
    )

    masked_top_votes = format_top_votes(
        bagging_masked_vote_counts[test_position]
    )

    # Top row: clean image and clean predictions.
    axes[0, column_index].imshow(
        test_images_clean[test_position],
        cmap="gray"
    )

    axes[0, column_index].set_title(
        f"{category.replace('_', ' ')}\n"
        f"True={true_label}, "
        f"Tree={clean_tree_prediction}, "
        f"Bag={clean_bagging_prediction}\n"
        f"Top votes: {clean_top_votes}",
        fontsize=8
    )

    axes[0, column_index].axis("off")

    # Bottom row: masked image and masked predictions.
    axes[1, column_index].imshow(
        test_images_masked[test_position],
        cmap="gray"
    )

    axes[1, column_index].set_title(
        f"Masked\n"
        f"True={true_label}, "
        f"Tree={masked_tree_prediction}, "
        f"Bag={masked_bagging_prediction}\n"
        f"Top votes: {masked_top_votes}",
        fontsize=8
    )

    axes[1, column_index].axis("off")

figure.suptitle(
    "Single-tree and bagging behaviour on selected faces",
    fontsize=12
)

figure.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.95]
)

figure.savefig(
    EXAMPLE_FIGURE_FILE,
    bbox_inches="tight"
)

plt.close(figure)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("Question 1(d) robustness results:")
print(
    robustness_results.to_string(
        index=False
    )
)

print()
print("Selected example details:")
print(
    example_results.to_string(
        index=False
    )
)

print()
print(
    "Number of clean-test faces corrected by bagging:",
    len(bagging_corrected_positions)
)

print(
    "Number of clean-test faces wrong for both models:",
    len(both_wrong_positions)
)

print()
print("Saved:", ROBUSTNESS_RESULT_FILE)
print("Saved:", EXAMPLE_RESULT_FILE)
print("Saved:", EXAMPLE_FIGURE_FILE)