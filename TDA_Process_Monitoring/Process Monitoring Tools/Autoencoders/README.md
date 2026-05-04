# AE-Based Event Detection

This folder contains the autoencoder (AE)-based reconstruction monitoring workflow for the synthetic three-tank system (TTS) example dataset.

## Method Summary

Autoencoders (AE) are used here as a reconstruction-based process monitoring method. The multivariate time-series data are first divided into overlapping moving windows. For the $k$-th window, the local data matrix is represented as:

$$
\mathbf{X}_k \in \mathbb{R}^{M_k \times n}
$$

where $M_k$ is the number of time samples in the window and $n$ is the number of measured variables.

The AE learns nonlinear encoder and decoder mappings:

$$
\mathbf{z}_k = E_{\theta}(\mathbf{X}_k)
$$

$$
\widehat{\mathbf{X}}_k = D_{\phi}(\mathbf{z}_k)
$$

where $\mathbf{z}_k$ is the latent representation of the $k$-th moving window, and $\widehat{\mathbf{X}}_k$ is the reconstructed window.

The event-detection metric is the window-wise reconstruction error:

$$
\varepsilon_k = \lVert \mathbf{X}_k - \widehat{\mathbf{X}}_k \rVert
$$

Large values of $\varepsilon_k$ indicate that the corresponding window departs from the nonlinear latent representation learned during nominal operation and may therefore correspond to an event or abnormal operating condition.

## Implementation Details

The notebook in this folder uses measured TTS liquid levels (`h1_meas`, `h2_meas`, and `h3_meas`) as the multivariate monitoring variables. Each measured variable is min-max normalized before constructing moving windows.

The monitoring workflow uses moving windows of 60 samples with 10% overlap between consecutive windows. The first six windows are treated as nominal operation data and are used to train the AE model. 

A dense autoencoder is used with a low-dimensional latent space. The model is trained only on nominal windows.

The final reconstruction-error profile is plotted against the original physical time axis over the full 4000-minute monitoring horizon.

## Folder Contents

| File | Description |
|---|---|
| `AE_Monitoring.ipynb` | Main notebook for AE-based moving-window event detection. |
| `README.md` | This description file. |

## Input Data

The notebook expects the synthetic TTS dataset file:

```text
TTS_multifault_short_events.csv
```

This folder provides a baseline reconstruction-based event detection workflow. It is intended for comparison with other monitoring approaches in the repository, including principal component analysis, Koopman autoencoders, and TDA-based trajectory monitoring.
