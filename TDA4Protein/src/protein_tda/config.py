"""Shared defaults for the protein TDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    """Default parameters used in the manuscript pipeline."""

    train_list: Path = REPO_ROOT / "data" / "splits" / "training.txt"
    test_list: Path = REPO_ROOT / "data" / "splits" / "testing.txt"
    csv_dir: Path = REPO_ROOT / "data" / "protein_csvs"
    feature_dir: Path = REPO_ROOT / "data" / "features"
    descriptor_dir: Path = REPO_ROOT / "data" / "descriptors"
    output_dir: Path = REPO_ROOT / "results"
    patch_radius: float = 9.0
    chemical_radii: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
    betti_resolution: int = 100
    pca_variance_threshold: float = 0.95
    mlp_hidden_dims: tuple[int, ...] = (40, 20, 5, 4)
    random_seed: int = 0


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
