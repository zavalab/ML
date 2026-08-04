#!/usr/bin/env python3
"""Extract alpha-complex TDA curves and chemical descriptors from protein CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protein_tda.features import extract_features_from_dataframe
from protein_tda.io import read_protein_csv, read_split, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, default=Path("data/splits/training.txt"))
    parser.add_argument("--test-list", type=Path, default=Path("data/splits/testing.txt"))
    parser.add_argument("--csv-dir", type=Path, default=Path("data/protein_csvs"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features/alpha_9a"))
    parser.add_argument("--patch-radius", type=float, default=9.0)
    parser.add_argument("--chemical-radii", type=float, nargs="+", default=[0.0, 3.0, 6.0, 9.0, 12.0])
    parser.add_argument("--betti-resolution", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional small-run limit for smoke tests.")
    parser.add_argument("--timing-csv", type=Path, default=Path("results/feature_extraction_timing.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protein_ids = read_split(args.train_list) + read_split(args.test_list)
    if args.limit:
        protein_ids = protein_ids[: args.limit]
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.timing_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for protein_id in tqdm(protein_ids, desc="Extracting features"):
        input_csv = args.csv_dir / f"{protein_id}.csv"
        output_npz = args.feature_dir / f"{protein_id}.npz"
        if args.skip_existing and output_npz.exists():
            rows.append({"protein_id": protein_id, "status": "skipped_existing", "seconds": 0.0, "n_vertices": ""})
            continue
        if not input_csv.exists():
            rows.append({"protein_id": protein_id, "status": "missing_csv", "seconds": 0.0, "n_vertices": ""})
            continue
        tic = time.time()
        features = extract_features_from_dataframe(
            read_protein_csv(input_csv),
            patch_radius=args.patch_radius,
            chemical_radii=tuple(args.chemical_radii),
            betti_resolution=args.betti_resolution,
        )
        save_npz(
            output_npz,
            coords=features.coords,
            labels=features.labels,
            betti=features.betti,
            ec=features.ec,
            chemical=features.chemical,
            neighbor_counts=features.neighbor_counts,
            filtration_grid=features.filtration_grid,
        )
        rows.append(
            {
                "protein_id": protein_id,
                "status": "processed",
                "seconds": time.time() - tic,
                "n_vertices": int(features.labels.shape[0]),
            }
        )

    with args.timing_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "status", "seconds", "n_vertices"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote features to: {args.feature_dir}")
    print(f"Wrote timing log to: {args.timing_csv}")


if __name__ == "__main__":
    main()
