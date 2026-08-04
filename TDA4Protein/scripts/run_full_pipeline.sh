#!/usr/bin/env bash
set -euo pipefail

CSV_DIR="${1:-data/protein_csvs}"
RADIUS="${2:-9}"

python scripts/validate_dataset.py --csv-dir "$CSV_DIR"
python scripts/extract_tda_features.py \
  --csv-dir "$CSV_DIR" \
  --patch-radius "$RADIUS" \
  --feature-dir "data/features/alpha_${RADIUS}a" \
  --timing-csv "results/feature_extraction_alpha_${RADIUS}a.csv" \
  --skip-existing
python scripts/fit_pca_descriptors.py \
  --feature-dir "data/features/alpha_${RADIUS}a" \
  --descriptor-dir "data/descriptors/combined_alpha_${RADIUS}a" \
  --pca-dir "results/pca_combined_alpha_${RADIUS}a" \
  --feature-set combined
python scripts/train_nn.py \
  --descriptor-dir "data/descriptors/combined_alpha_${RADIUS}a" \
  --output-dir "results/nn_combined_alpha_${RADIUS}a" \
  --epochs 50
python scripts/plot_training_curves.py \
  --history-csv "results/nn_combined_alpha_${RADIUS}a/training_history.csv" \
  --output-png "results/nn_combined_alpha_${RADIUS}a/training_auc_curves.png"
