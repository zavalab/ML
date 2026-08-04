#!/usr/bin/env python3
"""Generate the two-panel PCA projection and cumulative-variance figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.io import load_npz, read_split
from protein_tda.plotting import pca_projection_and_variance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-dir", type=Path, default=Path("data/descriptors/combined_alpha_9a"))
    parser.add_argument("--protein-list", type=Path, default=Path("data/splits/training.txt"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/pca_combined_alpha_9a/pca_figure"))
    parser.add_argument("--sample-per-class", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptors = []
    labels = []
    for protein_id in read_split(args.protein_list):
        arrays = load_npz(args.descriptor_dir / f"{protein_id}.npz")
        descriptors.append(arrays["descriptors"])
        labels.append(arrays["labels"])
    pca_projection_and_variance(
        np.concatenate(descriptors, axis=0),
        np.concatenate(labels, axis=0),
        args.output_prefix,
        sample_per_class=args.sample_per_class,
        random_seed=args.seed,
    )
    print(f"Wrote PCA figure and CSVs with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
