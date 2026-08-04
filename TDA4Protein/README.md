# Analyzing Protein-Protein Interactions using Topology

This repository contains all codes and data necessary for the implementation of a topology-based workflow used for protein interface prediction in the accompanying manuscript.

## Publication

This repository consists of scripts corresponding to the TDA-NN framework for predicting protein interface patches as discussed in the research paper:

**Mukherjee, A.**, Park, B., Malmstrom, A., Cisewski-Kehe, J., Van Lehn, R. C., and Zavala, V. M. "Scalable Extraction of Information on Protein-Protein Interactions using Topological Data Analysis" (*manuscript under review*)

The canonical manuscript split files are included in `data/splits/`:

- `training.txt`: 3003 protein-chain tags
- `testing.txt`: 359 protein-chain tags
- Total: 3362 entries

The protein surface CSV files are provided in `data/protein_csvs/`. Each CSV is named `<protein_id>.csv` and includes:

```text
x,y,z,charge,hbond,hphob,iface,nx,ny,nz
```

Only `x`, `y`, `z`, `charge`, `hbond`, `hphob`, and `iface` are required by these scripts.

## Environment

The repository is script-based. Run all commands from the repository root.

The scripts require a Python environment with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `joblib`, `gudhi`, and `torch` available.

## Recommended Data Layout

```text
data/
  splits/
    training.txt
    testing.txt
  protein_csvs/
    1A0G_B.csv
    ...
```

The `features/`, `descriptors/`, and `results/` folders are generated when the scripts are run. They are not included in the repository because they are reproducible from the CSV inputs.

## Full Pipeline

### Data Download

The protein surface CSV files are provided separately as `protein_csvs.zip`.

Download `protein_csvs.zip` from [Google Drive](https://drive.google.com/file/d/1mBtB3PWEUaauj9k1Re0iAkrMZ5Ugb0wC/view?usp=sharing).

Then unzip it into the repository `data/` folder:

```bash
unzip protein_csvs.zip -d data/
```

From the repository root:

```bash
bash scripts/run_full_pipeline.sh data/protein_csvs 9
```

This runs:

1. split/data validation
2. alpha-complex Betti and Euler characteristic feature extraction
3. chemical feature averaging over nested radii
4. PCA fitting on the training set and projection of train/test proteins
5. feedforward neural network (NN) training
6. ROC AUC training curve plotting

For a 12 Angstrom patch radius:

```bash
bash scripts/run_full_pipeline.sh data/protein_csvs 12
```

## Individual Commands

Validate local data availability:

```bash
python scripts/validate_dataset.py --csv-dir data/protein_csvs
```

Extract alpha-complex TDA and chemical features:

```bash
python scripts/extract_tda_features.py \
  --csv-dir data/protein_csvs \
  --patch-radius 9 \
  --feature-dir data/features/alpha_9a \
  --skip-existing
```

Fit PCA and write PCA-reduced descriptors:

```bash
python scripts/fit_pca_descriptors.py \
  --feature-dir data/features/alpha_9a \
  --descriptor-dir data/descriptors/combined_alpha_9a \
  --pca-dir results/pca_combined_alpha_9a \
  --feature-set combined
```

Run ablations by changing `--feature-set` to `tda` or `chemical` and writing to separate descriptor folders.

Train the neural network:

```bash
python scripts/train_nn.py \
  --descriptor-dir data/descriptors/combined_alpha_9a \
  --output-dir results/nn_combined_alpha_9a \
  --epochs 50
```

Generate training curves:

```bash
python scripts/plot_training_curves.py \
  --history-csv results/nn_combined_alpha_9a/training_history.csv \
  --output-png results/nn_combined_alpha_9a/training_auc_curves.png
```

