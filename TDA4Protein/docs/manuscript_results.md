# Manuscript Pipeline Mapping

This code is organized around the manuscript workflow:

1. Surface vertex CSV files provide local point clouds and chemical features.
2. `scripts/extract_tda_features.py` computes alpha-complex Betti curves and Euler characteristic curves for radius-defined patches.
3. The same script computes nested-radius chemical averages for charge, hydrogen-bond potential, and hydrophobicity.
4. `scripts/fit_pca_descriptors.py` concatenates the selected TDA and/or chemical descriptors, fits PCA on the training split, saves the 95 percent cumulative-variance threshold, and writes PCA-reduced descriptors.
5. `scripts/train_nn.py` trains the MLP on balanced interface/non-interface patch samples and reports train/test ROC AUC values at every epoch.
6. `scripts/plot_training_curves.py` and `scripts/plot_pca_projection.py` generate SI figures and their source CSV files.

The included split files contain 3362 total entries, consisting of 3003 training proteins and 359 testing proteins.
