# Ablation Commands

These commands reproduce TDA-only, chemical-only, and combined feature-space comparisons. Use separate output folders so each run keeps its own PCA model, descriptors, model checkpoints, and training history.

## 9 Angstrom Radius

```bash
python scripts/extract_tda_features.py \
  --csv-dir data/protein_csvs \
  --patch-radius 9 \
  --feature-dir data/features/alpha_9a \
  --skip-existing
```

```bash
python scripts/fit_pca_descriptors.py \
  --feature-dir data/features/alpha_9a \
  --descriptor-dir data/descriptors/tda_alpha_9a \
  --pca-dir results/pca_tda_alpha_9a \
  --feature-set tda

python scripts/train_nn.py \
  --descriptor-dir data/descriptors/tda_alpha_9a \
  --output-dir results/nn_tda_alpha_9a \
  --epochs 50
```

```bash
python scripts/fit_pca_descriptors.py \
  --feature-dir data/features/alpha_9a \
  --descriptor-dir data/descriptors/chemical_alpha_9a \
  --pca-dir results/pca_chemical_alpha_9a \
  --feature-set chemical

python scripts/train_nn.py \
  --descriptor-dir data/descriptors/chemical_alpha_9a \
  --output-dir results/nn_chemical_alpha_9a \
  --epochs 50
```

```bash
python scripts/fit_pca_descriptors.py \
  --feature-dir data/features/alpha_9a \
  --descriptor-dir data/descriptors/combined_alpha_9a \
  --pca-dir results/pca_combined_alpha_9a \
  --feature-set combined

python scripts/train_nn.py \
  --descriptor-dir data/descriptors/combined_alpha_9a \
  --output-dir results/nn_combined_alpha_9a \
  --epochs 50
```

## 12 Angstrom Radius

Repeat the same commands with:

```text
--patch-radius 12
--feature-dir data/features/alpha_12a
```

and use output folders ending in `_12a`.
