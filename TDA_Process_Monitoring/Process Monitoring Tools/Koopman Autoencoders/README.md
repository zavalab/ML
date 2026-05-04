# Koopman Autoencoder-Based Event Detection

This folder contains the Koopman autoencoder (KAE)-based trajectory monitoring workflow for the synthetic three-tank system (TTS) example dataset.

## Method Summary

Koopman autoencoders are used here as a trajectory-based process monitoring method. In contrast to reconstruction-based methods such as PCA and AE, which detect events using reconstruction mismatch relative to nominal operation, KAE detects events through changes in the temporal evolution of a learned latent representation.

The multivariate process measurement associated with the $k$-th moving window is represented as:

$$
\mathbf{X}_k \in \mathbb{R}^{M_k \times n}
$$

where $M_k$ is the number of time samples in the window and $n$ is the number of measured variables.

The encoder maps each moving window into a latent state:

$$
z_k = E_{\theta}(\mathbf{X}_k)
$$

The decoder reconstructs the window from the latent state:

$$
\widehat{\mathbf{X}}_k = D_{\phi}(z_k)
$$

The Koopman assumption is that the latent state evolves approximately linearly across consecutive windows:

$$
z_{k+1} \approx A z_k
$$

where $A$ is a learned linear Koopman operator. The corresponding latent dynamic increment is:

$$
\Delta z_k \approx {\bf (A-I)z}_k
$$

The event-detection metric is based on the norm:

$$
\bf \lVert (A-I)z_k \rVert
$$

Large values of this quantity indicate abrupt changes in the learned latent-space trajectory and may therefore correspond to transient events or abnormal operating conditions.

## Implementation Details

The notebook in this folder uses measured TTS liquid levels (`h1_meas`, `h2_meas`, and `h3_meas`) as the multivariate monitoring variables. Each measured variable is min-max normalized before constructing moving windows.

The monitoring workflow uses moving windows of 60 samples with 10% overlap between consecutive windows. Unlike reconstruction-based PCA and AE workflows, the KAE workflow does not use a separate nominal training region. Instead, all consecutive moving-window pairs are used to learn the latent trajectory dynamics:

$$
\mathbf{X}_k \rightarrow \mathbf{X}_{k+1}
$$

Each window matrix is flattened and passed through the encoder to obtain a latent state $z_k$. The Koopman matrix $A$ is trained to approximate the one-window-ahead latent evolution:

$$
z_{k+1} \approx A z_k
$$

The KAE model contains:

- an encoder network,
- a decoder network,
- a trainable Koopman matrix $A$,
- a reconstruction loss,
- a latent dynamics loss.

The final event-detection signal is the window-wise latent dynamics metric:

$$
\bf \left\|(A-I)z_k\right\|
$$

This signal is plotted against the original physical time axis over the full 4000-minute monitoring horizon.

## Folder Contents

| File | Description |
|---|---|
| `KAE_Monitoring.ipynb` | Main notebook for KAE-based moving-window event detection. |
| `README.md` | This description file. |

## Input Data

The notebook expects the synthetic TTS dataset file:

```text
TTS_multifault_short_events.csv
```

This folder provides a baseline trajectory-based event detection workflow. It is intended for comparison with other monitoring approaches in the repository, including principal component analysis, autoencoders, and TDA-based trajectory monitoring.
