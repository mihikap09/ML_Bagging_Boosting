# Machine Learning Assignment 2: Bagging, Boosting and Bias–Variance on Olivetti Faces

## Project Overview

This repository contains the implementation, experimental results, diagnostic figures and report for Machine Learning Assignment 2.

The project investigates decision trees, manually implemented bootstrap aggregation (bagging), multiclass SAMME boosting, robustness to masked facial images, and the bias–variance trade-off.

## Dataset

The experiments use the Olivetti Faces dataset, containing 400 grayscale facial images of 40 identities, with 10 images per identity. Each image has dimensions of 64 × 64 pixels.

The assignment requires raw flattened pixel features without PCA or other dimensionality reduction.

A fixed, identity-stratified split is stored in `data/split_indices.npz`:

- Training set: 240 images (6 per identity)
- Validation set: 80 images (2 per identity)
- Test set: 80 images (2 per identity)

The robustness experiment uses a masked test set created by setting the central 16 × 16 image block (rows and columns 24–39) to zero. Models are trained using clean training images only.

The saved split must be reused across all experiments.

## Repository Structure

```text
ML-assignment-2/
├── README.md
├── requirements.txt
├── data/
│   └── split_indices.npz
├── src/
│   ├── q1b_manual_bagging.py
│   ├── q1c_bagging_diagnostics.py
│   ├── q1d_robustness_analysis.py
│   ├── q2a_samme_demo.py
│   ├── q2b_samme_faces.py
│   ├── q2c_bagging_vs_boosting.py
│   ├── q2d_robustness_comparision.py
│   ├── q3a_bias_variance.py
│   ├── q3b_bias_variance_depth.py
│   └── q3d_final.py
├── results/
├── figures/
├── logs/
└── reports/
```

The directories `results/`, `figures/`, `logs/` and `reports/` contain experimental outputs, diagnostic visualizations, execution logs and the final report, respectively. Their contents should be checked against the actual submitted files.

## Installation

Python 3.10 or a compatible version supported by the project's dependencies is recommended.

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Experiments

### Q1: Decision Tree and Bagging

- Establish a decision-tree baseline using validation accuracy for model selection.
- Implement bootstrap aggregation manually.
- Evaluate out-of-bag (OOB) accuracy and coverage, pairwise disagreement, and vote margin.
- Compare a single tree and the selected bagging ensemble on clean and masked test images.

### Q2: Multiclass SAMME Boosting

- Demonstrate the SAMME boosting algorithm.
- Select the boosting model using validation performance.
- Compare boosting with bagging for different ensemble sizes.
- Evaluate robustness to centrally masked facial images.

### Q3: Bias–Variance Analysis

- Estimate bias squared and variance over 30 resampled training sets.
- Compare the selected single tree, bagging and boosting.
- Investigate the effect of decision-tree depth on bagging performance.
- Summarize the measured bias–variance trade-off.

## Reproducibility

Use the saved split indices for all experiments. Do not use the test set for hyperparameter selection.

Run the experiment scripts from the repository root, following their dependencies and the intended execution order. For example:

```bash
python src/q1b_manual_bagging.py
python src/q1c_bagging_diagnostics.py
python src/q1d_robustness_analysis.py
```

Continue with the Q2 and Q3 scripts after checking their required inputs and outputs. The commands above are entry-point examples; the full execution sequence must be verified against the actual scripts before claiming that the entire project can be reproduced.

Save the final outputs and execution logs. Confirm that generated results correspond to the committed code and saved split.

## Results and Report

The `results/` directory stores numerical outputs and experiment summaries. The `figures/` directory stores diagnostic visualizations. The final CVPR-format report should be placed in `reports/`.

Only include results that have been generated and verified by running the submitted implementation.

## Submission

The repository should contain:

- Complete source code for Q1, Q2 and Q3.
- The saved data split and any required reproducibility artifacts.
- Dependency information.
- Final result files, diagnostic figures and execution logs.
- The final CVPR two-column PDF report, within the assignment's page limit.

The first page of the report must include the student's name, complete roll number, repository URL and final Git commit hash.
