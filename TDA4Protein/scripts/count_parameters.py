#!/usr/bin/env python3
"""Count trainable parameters for the manuscript MLP architecture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.model import InterfaceMLP, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[40, 20, 5, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = InterfaceMLP(input_dim=args.input_dim, hidden_dims=tuple(args.hidden_dims))
    print(f"Input dimension: {args.input_dim}")
    print(f"Hidden dimensions: {args.hidden_dims}")
    print(f"Trainable parameters: {count_parameters(model)}")


if __name__ == "__main__":
    main()
