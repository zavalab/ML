#!/usr/bin/env python3
"""Validate split files and local data availability."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.io import read_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("data/splits/training.txt"))
    parser.add_argument("--test-list", type=Path, default=Path("data/splits/testing.txt"))
    parser.add_argument("--csv-dir", type=Path, default=Path("data/protein_csvs"))
    parser.add_argument("--precompute-dir", type=Path, default=Path("data/precomputation_tda"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/dataset_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_split(args.train_list)
    test = read_split(args.test_list)
    rows = []
    for split, protein_ids in [("train", train), ("test", test)]:
        for protein_id in protein_ids:
            rows.append(
                {
                    "protein_id": protein_id,
                    "split": split,
                    "csv_exists": (args.csv_dir / f"{protein_id}.csv").exists(),
                    "precompute_exists": (args.precompute_dir / protein_id).exists(),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_csv = sum(row["csv_exists"] for row in rows)
    n_pre = sum(row["precompute_exists"] for row in rows)
    print(f"Training proteins: {len(train)}")
    print(f"Testing proteins:  {len(test)}")
    print(f"Total split entries: {len(rows)}")
    print(f"CSV files found: {n_csv}/{len(rows)}")
    print(f"Precomputation folders found: {n_pre}/{len(rows)}")
    print(f"Wrote manifest: {args.output_csv}")


if __name__ == "__main__":
    main()
