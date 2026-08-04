# Data Preparation

## Required Input

The full dataset is represented by the split lists in `data/splits/`:

- `training.txt`: 3003 protein IDs
- `testing.txt`: 359 protein IDs

For every ID, place the corresponding surface CSV in:

```text
data/protein_csvs/<protein_id>.csv
```

The scripts require the columns:

```text
x,y,z,charge,hbond,hphob,iface
```

Additional columns, such as surface normals, are allowed and ignored by this pipeline.

## Generated Files

`scripts/extract_tda_features.py` creates one compressed file per protein:

```text
data/features/alpha_9a/<protein_id>.npz
```

Each file contains:

- `coords`: surface coordinates
- `labels`: interface labels
- `betti`: Betti-0, Betti-1, and Betti-2 curves
- `ec`: Euler characteristic curve, computed as `Betti0 - Betti1 + Betti2`
- `chemical`: averaged charge, hbond, and hydrophobicity descriptors over nested radii
- `neighbor_counts`: number of vertices in each patch
- `filtration_grid`: normalized filtration grid used for Betti/EC curves

`scripts/fit_pca_descriptors.py` creates:

```text
data/descriptors/<descriptor_name>/<protein_id>.npz
results/<pca_name>/pca_model.joblib
results/<pca_name>/pca_variance.csv
```

The descriptor NPZ files contain PCA-reduced descriptors, labels, and coordinates.

