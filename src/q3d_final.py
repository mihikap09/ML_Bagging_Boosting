"""
Question 3(d): Create the final model-comparison table.

This script combines:
- clean and masked test accuracy from Q2(d);
- measured bias squared and variance from Q3(a).

No models are retrained.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBUSTNESS_FILE = (
    PROJECT_ROOT
    / "results"
    / "q2d_robustness.csv"
)

BIAS_VARIANCE_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3a_bias_variance_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "q3d_final_comparison.csv"
)


robustness = pd.read_csv(
    ROBUSTNESS_FILE
)

bias_variance = pd.read_csv(
    BIAS_VARIANCE_FILE
)


model_order = [
    "single_tree",
    "manual_bagging",
    "manual_boosting"
]

rows = []

for model in model_order:
    clean_accuracy = float(
        robustness.loc[
            (
                robustness["model"] == model
            )
            & (
                robustness["test_condition"]
                == "clean"
            ),
            "accuracy"
        ].iloc[0]
    )

    masked_accuracy = float(
        robustness.loc[
            (
                robustness["model"] == model
            )
            & (
                robustness["test_condition"]
                == "masked"
            ),
            "accuracy"
        ].iloc[0]
    )

    bias_squared = float(
        bias_variance.loc[
            bias_variance["model"] == model,
            "bias_squared"
        ].iloc[0]
    )

    variance = float(
        bias_variance.loc[
            bias_variance["model"] == model,
            "variance"
        ].iloc[0]
    )

    rows.append(
        {
            "model": model,
            "clean_test_accuracy":
                clean_accuracy,
            "masked_test_accuracy":
                masked_accuracy,
            "accuracy_drop_due_to_mask":
                clean_accuracy
                - masked_accuracy,
            "bias_squared":
                bias_squared,
            "variance":
                variance
        }
    )


final_table = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_table.to_csv(
    OUTPUT_FILE,
    index=False
)


print("QUESTION 3(d): FINAL COMPARISON")
print()

print(
    final_table.to_string(
        index=False
    )
)

print()
print("Saved:", OUTPUT_FILE)