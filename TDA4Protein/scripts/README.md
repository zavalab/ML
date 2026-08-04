# Scripts

Run all commands from the repository root.

## Main Workflow

The complete 9 Angstrom workflow can be run with:

```bash
bash scripts/run_full_pipeline.sh data/protein_csvs 9
```

For 12 Angstrom patches:

```bash
bash scripts/run_full_pipeline.sh data/protein_csvs 12
```

## Script Contents

- `validate_dataset.py`: checks the train/test split files against `data/protein_csvs/` and writes a dataset manifest.
- `extract_tda_features.py`: computes alpha-complex Betti curves, Euler characteristic curves, and local chemical descriptors from the protein CSV files.
- `fit_pca_descriptors.py`: fits PCA on the training split and writes PCA-reduced descriptor files for train/test proteins.
- `train_nn.py`: trains the interface-prediction neural network and writes training history plus model checkpoints.
- `plot_training_curves.py`: plots train/test ROC AUC curves from `training_history.csv`.
- `plot_pca_projection.py`: generates the two-panel PCA projection and cumulative-variance figure.
- `count_parameters.py`: counts trainable parameters for the MLP architecture.
- `run_full_pipeline.sh`: runs validation, feature extraction, PCA, NN training, and training-curve plotting.
- `ablation_commands.md`: command examples for TDA-only, chemical-only, and combined descriptor ablations.

## Generated Outputs

The scripts create output folders as needed:

```text
data/features/
data/descriptors/
results/
```

These generated folders are reproducible from `data/protein_csvs/` and are intentionally ignored by Git.
