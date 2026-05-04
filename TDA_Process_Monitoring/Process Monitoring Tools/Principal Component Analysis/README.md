# PCA-Based Event Detection

This folder contains the PCA-based reconstruction monitoring workflow for the synthetic three-tank system (TTS) example dataset.

## Method Summary

Principal component analysis (PCA) is used here as a reconstruction-based process monitoring method. The multivariate time-series data are first divided into overlapping moving windows. For the $k$-th window, the local data matrix is represented as:

$$
\mathbf{X}_k \in \mathbb{R}^{M_k \times n}
$$

where $M_k$ is the number of time samples in the window and $n$ is the number of measured variables.

PCA maps each window into a lower-dimensional principal subspace:

$$
\mathbf{T}_k = \mathbf{X}_k\mathbf{P}
$$

where $\mathbf{P}$ is the PCA loading matrix. The reconstructed window is then obtained as:

$$
\widehat{\mathbf{X}}_k = \mathbf{T}_k\mathbf{P}^{T}
= \mathbf{X}_k\mathbf{P}\mathbf{P}^{T}
$$

The event-detection metric is the window-wise reconstruction error:

$$
\varepsilon_k = \lVert \mathbf{X}_k - \widehat{\mathbf{X}}_k \rVert
$$

Large values of $\varepsilon_k$ indicate that the corresponding window departs from the nominal PCA subspace and may therefore correspond to an event or abnormal operating condition.

## Implementation Details

The notebook in this folder uses:

- measured TTS liquid levels: `h1_meas`, `h2_meas`, and `h3_meas`,
- min-max normalization of each measured variable,
- moving windows of 60 samples,
- 10% overlap between consecutive windows,
- the first six windows as nominal training data.

The resulting window-wise reconstruction-error profile is plotted against the original physical time axis over the full 4000-minute monitoring horizon.

## Folder Contents

| File | Description |
|---|---|
| `PCA_Monitoring.ipynb` | Main notebook for PCA-based moving-window event detection. |
| `README.md` | This description file. |

## Input Data

The notebook expects the synthetic TTS dataset file:

```text
TTS_multifault_short_events.csv
```

This folder provides a baseline reconstruction-based event detection workflow. It is intended for comparison with other monitoring approaches in the repository, including autoencoders, Koopman autoencoders, and TDA-based trajectory monitoring.
