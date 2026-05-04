# TDA-Based Event Detection

This folder contains the topological data analysis (TDA)-based trajectory monitoring workflow for the synthetic three-tank system (TTS) example dataset.

## Method Summary

The TDA workflow is a trajectory-based process monitoring approach. Instead of detecting events through reconstruction error, the method represents multivariate time-series data as evolving topological descriptors and identifies events through changes in the temporal evolution of those descriptors.

For the $k$-th moving window, the local data matrix is represented as:

$$
\mathbf{X}_k \in \mathbb{R}^{M_k \times n}
$$

where $M_k$ is the number of time samples in the window and $n$ is the number of measured variables.

The normalized and transposed time-series matrix defines a 2D manifold representation:

$$
\mathbf{Y} = \mathcal{N}(\mathbf{X}^{T}) \in [0,1]^{n \times M}
$$

where rows correspond to process variables, columns correspond to time samples, and intensity denotes the normalized measured value.

For each moving window, the local manifold is interpreted as a scalar field:

$$
h_k(i,j) = (\mathbf{Y}_k)_{i,j}
$$

A sublevel-set filtration is constructed over thresholds $\ell \in [0,1]$:

$$
\mathcal{S}_k(\ell) = \{(i,j) : h_k(i,j) \leq \ell\}
$$

Using a cubical complex, the Betti numbers $\beta_0^{(k)}(\ell)$ and $\beta_1^{(k)}(\ell)$ are computed. The Euler characteristic is then:

$$
\chi_k(\ell) = \beta_0^{(k)}(\ell) - \beta_1^{(k)}(\ell)
$$

The EC curve $\chi_k(\ell)$ provides a compact topological descriptor of the $k$-th moving window.

## Learning Topological Dynamics

The sequence of EC curves is treated as a trajectory of topological descriptors:

$$
\{\chi_k\}_{k=1}^{K}
$$

The temporal derivative of the EC curves is approximated using finite differences:

$$
\frac{d\chi(\tau_k)}{d\tau} \approx \frac{\chi_{k+1} - \chi_k}{\Delta \tau}
$$

A dense neural network is then trained to learn the mapping:

$$
\chi_k \mapsto \frac{d\chi(\tau_k)}{d\tau}
$$

The final event-detection signal is the norm of the learned topological velocity:

$$
\lVert \dot{\chi}_k \rVert
$$

Large values indicate abrupt changes in the topological trajectory and may therefore correspond to transient events or abnormal operating conditions.

## Note on NODE vs Dense NN

In the manuscript, the temporal evolution of TDA descriptors is formulated using a neural ordinary differential equation (NODE). In this synthetic TTS example, however, only three measured variables are used. Because the resulting EC curves can be highly piecewise constant, the descriptor trajectories may not be sufficiently smooth for stable NODE learning.

For this reason, the notebook uses a dense neural network to learn the EC-derivative map. For higher-dimensional multivariate process datasets with smoother topological descriptor trajectories, the NODE formulation can be used directly.

## Implementation Details

The notebook in this folder uses measured TTS liquid levels (`h1_meas`, `h2_meas`, and `h3_meas`) as the multivariate monitoring variables. Each measured variable is min-max normalized before constructing the manifold representation.

The monitoring workflow uses:

- moving windows of 60 samples,
- 10% overlap between consecutive windows,
- cubical-complex filtrations,
- 201 filtration thresholds over $[0,1]$,
- Euler characteristic curves as topological descriptors,
- dense NN learning of EC-curve derivatives,
- $\|\dot\chi|$ as the final monitoring signal.

## Folder Contents

| File | Description |
|---|---|
| `TDA_NN_Monitoring.ipynb` | Main notebook for TDA-based moving-window event detection. |
| `README.md` | This description file. |

## Input Data

The notebook expects the synthetic TTS dataset file:

```text
TTS_multifault_short_events.csv
```
This folder provides a baseline trajectory-based event detection workflow. It is intended for comparison with other monitoring approaches in the repository, including principal component analysis, autoencoders, and Koopman autoencoders.
