#!/usr/bin/env python3
"""Fit PCA on training features and save per-protein descriptors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.io import read_split
from protein_tda.pca import fit_pca, project_features, save_pca_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("data/splits/training.txt"))
    parser.add_argument("--test-list", type=Path, default=Path("data/splits/testing.txt"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/alpha_9a"))
    parser.add_argument("--descriptor-dir", type=Path, default=Path("data/descriptors/combined_alpha_9a"))
    parser.add_argument("--pca-dir", type=Path, default=Path("results/pca_combined_alpha_9a"))
    parser.add_argument("--feature-set", choices=["tda", "chemical", "combined"], default="combined")
    parser.add_argument("--variance-threshold", type=float, default=0.95)
    parser.add_argument("--max-fit-rows-per-protein", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ids = read_split(args.train_list)
    test_ids = read_split(args.test_list)
    scaler, pca_model, metadata = fit_pca(
        args.feature_dir,
        train_ids,
        feature_set=args.feature_set,
        variance_threshold=args.variance_threshold,
        max_fit_rows=args.max_fit_rows_per_protein,
        random_seed=args.seed,
    )
    save_pca_artifacts(args.pca_dir, scaler, pca_model, metadata)
    project_features(args.feature_dir, args.descriptor_dir, train_ids + test_ids, args.feature_set, scaler, pca_model)
    print(f"Feature set: {args.feature_set}")
    print(f"Input dimension: {metadata['input_dim']}")
    print(f"Retained PCs at {args.variance_threshold:.2f} variance: {metadata['n_components']}")
    print(f"Wrote PCA artifacts: {args.pca_dir}")
    print(f"Wrote descriptors: {args.descriptor_dir}")


if __name__ == "__main__":
    main()
