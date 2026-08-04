#!/usr/bin/env python3
"""Plot ROC AUC training curves from a training_history.csv file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.plotting import plot_training_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_auc(args.history_csv, args.output_png)
    print(f"Wrote plot: {args.output_png}")


if __name__ == "__main__":
    main()
