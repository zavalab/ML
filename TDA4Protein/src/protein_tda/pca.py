"""PCA fitting and projection utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from .features import descriptor_matrix
from .io import load_npz, save_npz


def load_feature_batches(feature_dir: str | Path, protein_ids: list[str], feature_set: str):
    """Yield descriptor matrices from feature NPZ files."""

    feature_dir = Path(feature_dir)
    for protein_id in protein_ids:
        path = feature_dir / f"{protein_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")
        yield protein_id, descriptor_matrix(load_npz(path), feature_set=feature_set)


def fit_pca(
    feature_dir: str | Path,
    protein_ids: list[str],
    feature_set: str = "combined",
    variance_threshold: float = 0.95,
    max_fit_rows: int | None = None,
    random_seed: int = 0,
):
    """Fit StandardScaler + PCA and choose the number of PCs reaching a variance threshold.

    The fit is streamed over per-protein feature files so the full manuscript split does not
    need to be concatenated into one large matrix.
    """

    rng = np.random.default_rng(random_seed)
    total_rows = 0
    input_dim = None
    scaler = StandardScaler()

    for _, matrix in load_feature_batches(feature_dir, protein_ids, feature_set):
        if max_fit_rows and matrix.shape[0] > max_fit_rows:
            idx = rng.choice(matrix.shape[0], size=max_fit_rows, replace=False)
            matrix = matrix[idx]
        scaler.partial_fit(matrix)
        total_rows += matrix.shape[0]
        input_dim = matrix.shape[1]
    if total_rows == 0 or input_dim is None:
        raise ValueError("No feature matrices were loaded.")

    pca_model = IncrementalPCA(n_components=input_dim)
    for _, matrix in load_feature_batches(feature_dir, protein_ids, feature_set):
        if max_fit_rows and matrix.shape[0] > max_fit_rows:
            idx = rng.choice(matrix.shape[0], size=max_fit_rows, replace=False)
            matrix = matrix[idx]
        pca_model.partial_fit(scaler.transform(matrix))

    cumulative = np.cumsum(pca_model.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    pca_model.retained_n_components_ = n_components
    metadata = {
        "feature_set": feature_set,
        "variance_threshold": variance_threshold,
        "n_components": n_components,
        "fit_rows": int(total_rows),
        "input_dim": int(input_dim),
        "explained_variance_ratio": pca_model.explained_variance_ratio_,
        "cumulative_variance": cumulative,
    }
    return scaler, pca_model, metadata


def save_pca_artifacts(out_dir: str | Path, scaler, pca_model, metadata: dict) -> None:
    """Write PCA artifacts and variance CSV."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "pca": pca_model, "metadata": metadata}, out_dir / "pca_model.joblib")
    rows = np.column_stack(
        [
            np.arange(1, len(metadata["explained_variance_ratio"]) + 1),
            metadata["explained_variance_ratio"],
            metadata["cumulative_variance"],
        ]
    )
    np.savetxt(
        out_dir / "pca_variance.csv",
        rows,
        delimiter=",",
        header="component,explained_variance_ratio,cumulative_variance",
        comments="",
    )


def project_features(
    feature_dir: str | Path,
    descriptor_dir: str | Path,
    protein_ids: list[str],
    feature_set: str,
    scaler,
    pca_model,
) -> None:
    """Transform per-protein feature files into PCA descriptor NPZ files."""

    descriptor_dir = Path(descriptor_dir)
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    for protein_id, matrix in load_feature_batches(feature_dir, protein_ids, feature_set):
        source = load_npz(Path(feature_dir) / f"{protein_id}.npz")
        n_components = int(getattr(pca_model, "retained_n_components_", pca_model.n_components))
        descriptors = pca_model.transform(scaler.transform(matrix))[:, :n_components].astype(np.float32)
        save_npz(
            descriptor_dir / f"{protein_id}.npz",
            descriptors=descriptors,
            labels=source["labels"].astype(np.int64),
            coords=source["coords"].astype(np.float32),
        )
