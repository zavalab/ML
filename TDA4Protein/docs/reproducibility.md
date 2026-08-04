# Reproducibility

## Randomness

The scripts expose `--seed`, defaulting to `0`, for PCA sampling and neural-network training. Training uses balanced per-protein sampling at each epoch:

```text
selected indices = all interface vertices + an equal number of sampled non-interface vertices
```

## Feature Definitions

For each surface vertex, the extraction script builds a radius-neighborhood point cloud and computes an alpha complex using GUDHI. Finite persistence intervals for dimensions 0, 1, and 2 are converted into Betti curves. The Euler characteristic curve is:

```text
EC = Betti0 - Betti1 + Betti2
```

Chemical descriptors are local averages of charge, hydrogen-bond potential, and hydrophobicity over the requested nested radii.

## PCA

PCA is fit on the training split only. The trained scaler and PCA model are then applied to both the training and testing splits. By default, the retained number of principal components is the smallest number reaching 95 percent cumulative explained variance.

## Training Outputs

Each training run writes:

- `best_model.pt`
- `last_model.pt`
- `training_history.csv`
- `summary.json`

The history CSV includes mean, median, and all-point ROC AUC for train and test sets at every epoch, plus per-epoch runtime.
