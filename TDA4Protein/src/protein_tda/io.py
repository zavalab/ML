"""Input/output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CSV_COLUMNS = ("x", "y", "z", "charge", "hbond", "hphob", "iface")


def read_split(path: str | Path) -> list[str]:
    """Read a protein-tag split file, ignoring blanks and comments."""

    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def read_protein_csv(path: str | Path) -> pd.DataFrame:
    """Load a protein surface CSV and validate the required columns."""

    df = pd.read_csv(path)
    missing = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.loc[:, CSV_COLUMNS].copy()


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save arrays as compressed NPZ, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ file into a plain dictionary."""

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
