"""
Question 2(a): One complete SAMME boosting round.

This script uses a tiny three-class dataset and records:

1. Initial sample weights.
2. Weak-learner predictions.
3. Incorrect-prediction indicators.
4. Weighted classification error.
5. SAMME learner weight alpha.
6. Unnormalized updated sample weights.
7. Normalized updated sample weights.

The same update rule will later be used for the Olivetti
faces dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# Project paths and constants
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUND_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q2a_samme_round.csv"
)

SUMMARY_RESULT_FILE = (
    PROJECT_ROOT / "results" / "q2a_samme_summary.csv"
)

SEED = 26004


# ---------------------------------------------------------
# Tiny three-class dataset
# ---------------------------------------------------------

# There are nine samples and three classes.
#
# A depth-1 decision tree can isolate one class but cannot
# perfectly separate all three classes in a single split.
#
# This guarantees a nonzero and nontrivial weighted error.

X_tiny = np.array(
    [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0]
    ],
    dtype=np.float64
)

y_tiny = np.array(
    [
        0, 0, 0,
        1, 1, 1,
        2, 2, 2
    ],
    dtype=np.int64
)

classes = np.unique(y_tiny)
number_of_classes = len(classes)
number_of_samples = len(y_tiny)

assert number_of_classes == 3
assert number_of_samples <= 10


# ---------------------------------------------------------
# Initial SAMME sample weights
# ---------------------------------------------------------

# Every observation initially receives equal weight.

weights_before = np.full(
    number_of_samples,
    fill_value=1.0 / number_of_samples,
    dtype=np.float64
)

assert np.isclose(
    np.sum(weights_before),
    1.0
)


# ---------------------------------------------------------
# Train one weak learner
# ---------------------------------------------------------

weak_learner = DecisionTreeClassifier(
    max_depth=1,
    random_state=SEED
)

weak_learner.fit(
    X_tiny,
    y_tiny,
    sample_weight=weights_before
)

predictions = weak_learner.predict(
    X_tiny
).astype(np.int64)

incorrect = (
    predictions != y_tiny
)

incorrect_as_float = incorrect.astype(
    np.float64
)


# ---------------------------------------------------------
# Compute the weighted error
# ---------------------------------------------------------

# Since weights sum to one, the weighted error is simply
# the sum of the weights of misclassified observations.

weighted_error = float(
    np.sum(
        weights_before * incorrect_as_float
    )
)

random_guess_error = (
    1.0 - 1.0 / number_of_classes
)

# A SAMME weak learner must perform strictly better than
# random multiclass guessing.
if weighted_error >= random_guess_error:
    raise RuntimeError(
        "Weak learner is not better than random "
        "multiclass guessing."
    )

if weighted_error <= 0.0:
    raise RuntimeError(
        "This demonstration requires a non-perfect "
        "weak learner so that alpha remains finite."
    )


# ---------------------------------------------------------
# Calculate the SAMME learner weight
# ---------------------------------------------------------

alpha = float(
    np.log(
        (1.0 - weighted_error)
        / weighted_error
    )
    + np.log(number_of_classes - 1.0)
)


# ---------------------------------------------------------
# Update and normalize the sample weights
# ---------------------------------------------------------

# Correctly classified observations are multiplied by 1.
# Incorrectly classified observations are multiplied by
# exp(alpha).

weight_multiplier = np.exp(
    alpha * incorrect_as_float
)

weights_unnormalized = (
    weights_before * weight_multiplier
)

normalization_constant = float(
    np.sum(weights_unnormalized)
)

weights_after = (
    weights_unnormalized
    / normalization_constant
)

assert np.isclose(
    np.sum(weights_after),
    1.0
)


# ---------------------------------------------------------
# Store the per-observation round log
# ---------------------------------------------------------

round_log = pd.DataFrame(
    {
        "sample_index": np.arange(
            number_of_samples
        ),
        "feature_value": X_tiny[:, 0],
        "true_class": y_tiny,
        "weight_before": weights_before,
        "prediction": predictions,
        "incorrect": incorrect.astype(
            np.int64
        ),
        "weight_multiplier": weight_multiplier,
        "weight_unnormalized":
            weights_unnormalized,
        "weight_after": weights_after
    }
)


# ---------------------------------------------------------
# Store the round summary
# ---------------------------------------------------------

# A decision stump contains one internal split. The root
# threshold is stored at position zero.
root_threshold = float(
    weak_learner.tree_.threshold[0]
)

root_feature = int(
    weak_learner.tree_.feature[0]
)

summary = pd.DataFrame(
    [
        {
            "number_of_samples":
                number_of_samples,
            "number_of_classes":
                number_of_classes,
            "weak_learner_depth":
                weak_learner.get_depth(),
            "root_feature":
                root_feature,
            "root_threshold":
                root_threshold,
            "weighted_error":
                weighted_error,
            "random_guess_error":
                random_guess_error,
            "alpha":
                alpha,
            "normalization_constant":
                normalization_constant,
            "weights_before_sum":
                float(np.sum(weights_before)),
            "weights_after_sum":
                float(np.sum(weights_after))
        }
    ]
)


# ---------------------------------------------------------
# Save reproducible results
# ---------------------------------------------------------

ROUND_RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

round_log.to_csv(
    ROUND_RESULT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_RESULT_FILE,
    index=False
)


# ---------------------------------------------------------
# Print the complete SAMME round
# ---------------------------------------------------------

print("QUESTION 2(a): ONE COMPLETE SAMME ROUND")
print()

print("Tiny feature matrix:")
print(X_tiny)

print()
print("True labels:")
print(y_tiny)

print()
print("Initial weights:")
print(weights_before)

print()
print("Weak-learner predictions:")
print(predictions)

print()
print("Incorrect indicators:")
print(incorrect.astype(np.int64))

print()
print("Weighted error:")
print(weighted_error)

print()
print("Random multiclass error threshold:")
print(random_guess_error)

print()
print("SAMME alpha:")
print(alpha)

print()
print("Unnormalized updated weights:")
print(weights_unnormalized)

print()
print("Normalization constant:")
print(normalization_constant)

print()
print("Normalized updated weights:")
print(weights_after)

print()
print("Sum of updated weights:")
print(np.sum(weights_after))

print()
print("Complete per-sample log:")
print(round_log.to_string(index=False))

print()
print("Round summary:")
print(summary.to_string(index=False))

print()
print("Saved:", ROUND_RESULT_FILE)
print("Saved:", SUMMARY_RESULT_FILE)