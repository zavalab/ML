#!/usr/bin/env python3
"""Train the TDA/chemical interface neural network."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.io import read_split
from protein_tda.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("data/splits/training.txt"))
    parser.add_argument("--test-list", type=Path, default=Path("data/splits/testing.txt"))
    parser.add_argument("--descriptor-dir", type=Path, default=Path("data/descriptors/combined_alpha_9a"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nn_combined_alpha_9a"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[40, 20, 5, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps, or omitted for automatic selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_model(
        descriptor_dir=args.descriptor_dir,
        train_ids=read_split(args.train_list),
        test_ids=read_split(args.test_list),
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        random_seed=args.seed,
        device_name=args.device,
    )
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
