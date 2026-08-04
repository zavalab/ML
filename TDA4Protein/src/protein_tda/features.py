"""TDA and chemical descriptor extraction from protein surface CSV files."""

from __future__ import annotations

from dataclasses import dataclass

import gudhi as gd
import gudhi.representations as gdr
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class ProteinFeatures:
    """Per-surface-vertex descriptors for one protein chain."""

    coords: np.ndarray
    labels: np.ndarray
    betti: np.ndarray
    ec: np.ndarray
    chemical: np.ndarray
    neighbor_counts: np.ndarray
    filtration_grid: np.ndarray


def neighbor_indices(coords: np.ndarray, radius: float) -> list[list[int]]:
    """Return radius-neighborhood indices for every surface vertex."""

    tree = cKDTree(coords)
    return tree.query_ball_point(coords, r=radius, workers=-1)


def betti_and_ec_for_patch(points: np.ndarray, resolution: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute alpha-complex Betti-0/1/2 curves and Euler characteristic for one patch."""

    if points.shape[0] < 2:
        betti = np.zeros((3, resolution), dtype=np.float32)
        return betti, betti[0].copy()

    try:
        alpha = gd.AlphaComplex(points=points)
        simplex_tree = alpha.create_simplex_tree()
        simplex_tree.compute_persistence()
        diagrams = [simplex_tree.persistence_intervals_in_dimension(dim) for dim in range(3)]
        diagrams = gdr.DiagramSelector(use=True, limit=np.inf, point_type="finite").fit_transform(diagrams)
        if all(len(diagram) == 0 for diagram in diagrams):
            betti = np.zeros((3, resolution), dtype=np.float32)
        else:
            diagrams = gdr.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())]).fit_transform(diagrams)
            betti = gdr.BettiCurve(resolution=resolution).fit_transform(diagrams).astype(np.float32)
    except Exception:
        betti = np.zeros((3, resolution), dtype=np.float32)

    ec = betti[0] - betti[1] + betti[2]
    return betti, ec.astype(np.float32)


def chemical_averages(
    coords: np.ndarray,
    chemical_values: np.ndarray,
    radii: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0),
) -> np.ndarray:
    """Average charge, hbond, and hydrophobicity over nested radius neighborhoods."""

    tree = cKDTree(coords)
    blocks = []
    for radius in radii:
        idxs = tree.query_ball_point(coords, r=radius, workers=-1)
        block = np.zeros((coords.shape[0], chemical_values.shape[1]), dtype=np.float32)
        for i, idx in enumerate(idxs):
            block[i] = chemical_values[idx].mean(axis=0)
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def extract_features_from_dataframe(
    df,
    patch_radius: float = 9.0,
    chemical_radii: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0),
    betti_resolution: int = 100,
) -> ProteinFeatures:
    """Compute alpha-complex TDA curves and chemical descriptors for one protein."""

    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    chemical_values = df[["charge", "hbond", "hphob"]].to_numpy(dtype=np.float32)
    labels = df["iface"].to_numpy(dtype=np.int64)
    idxs = neighbor_indices(coords, patch_radius)

    betti = np.zeros((coords.shape[0], 3, betti_resolution), dtype=np.float32)
    ec = np.zeros((coords.shape[0], betti_resolution), dtype=np.float32)
    for i, idx in enumerate(idxs):
        betti[i], ec[i] = betti_and_ec_for_patch(coords[idx], resolution=betti_resolution)

    chem = chemical_averages(coords, chemical_values, radii=chemical_radii)
    filtration_grid = np.linspace(0.0, 1.0, betti_resolution, dtype=np.float32)
    neighbor_counts = np.array([len(idx) for idx in idxs], dtype=np.int32)
    return ProteinFeatures(
        coords=coords.astype(np.float32),
        labels=labels,
        betti=betti,
        ec=ec,
        chemical=chem,
        neighbor_counts=neighbor_counts,
        filtration_grid=filtration_grid,
    )


def descriptor_matrix(feature_npz: dict[str, np.ndarray], feature_set: str = "combined") -> np.ndarray:
    """Build a 2D matrix from saved features for TDA, chemical, or combined input."""

    feature_set = feature_set.lower()
    tda = np.concatenate(
        [
            feature_npz["betti"].reshape(feature_npz["betti"].shape[0], -1),
            feature_npz["ec"],
        ],
        axis=1,
    )
    chemical = feature_npz["chemical"]
    if feature_set == "tda":
        return tda.astype(np.float32)
    if feature_set == "chemical":
        return chemical.astype(np.float32)
    if feature_set == "combined":
        return np.concatenate([tda, chemical], axis=1).astype(np.float32)
    raise ValueError("feature_set must be one of: tda, chemical, combined")
