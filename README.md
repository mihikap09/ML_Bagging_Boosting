# Manual Bagging and SAMME Boosting on Olivetti Faces

This repository contains the implementation and experimental
results for Machine Learning Assignment 2.

The project compares:

- a selected decision-tree baseline;
- manually implemented bootstrap aggregation;
- manually implemented multiclass SAMME boosting;
- clean and centrally masked face recognition;
- measured bias and variance across 30 resampled training sets.

## Dataset

The project uses the Olivetti Faces dataset:

- 400 grayscale 64 × 64 images
- 40 identities
- 10 images per identity

A fixed identity-stratified split is stored in:

`data/split_indices.npz`

The split contains:

- 240 training images: 6 per identity
- 80 validation images: 2 per identity
- 80 test images: 2 per identity

## Installation

```bash
python -m venv .venv