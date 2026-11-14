# DES-ML

Graph neural networks and other machine-learning models for predicting CO₂ solubility in deep eutectic solvents (DES), with generalization tests across temperature, pressure, and solvent composition.

## Contents
- `00_images.ipynb`: figures
- `01_ANN_sigma_reproduction.ipynb`: artificial neural network based on σ-profiles (reproduction of earlier work)
- `02_RF_fingerprints.ipynb`: random forest on molecular fingerprints
- `03_GNN_model1.ipynb`, `04_GNN_modelo2.ipynb`: graph neural network models
- `05_GNN_generalization_T_P.ipynb`: generalization to new temperatures and pressures
- `06_GNN_generalization_DES.ipynb`: generalization to new solvents
- `07_GNN_generalization_xHBD.ipynb`, `07_GNN_generalization/`: generalization to new hydrogen-bond-donor fractions
- `raw_data/`, `generated_files/`: data and model outputs

## Reference
This code accompanies the paper:

Gabriel Hernández Morales, Edgar Ivan Sanchez Medina, Arturo Jiménez-Gutiérrez, and Victor M. Zavala. "Graph neural networks for CO₂ solubility predictions in Deep Eutectic Solvents." *Computers & Chemical Engineering* 187 (2024): 108750. https://doi.org/10.1016/j.compchemeng.2024.108750

## BibTeX
```bibtex
@article{hernandezmorales2024graph,
  title   = {{Graph neural networks for CO$_2$ solubility predictions in Deep Eutectic Solvents}},
  author  = {Hernández Morales, Gabriel and Sanchez Medina, Edgar Ivan and Jiménez-Gutiérrez, Arturo and Zavala, Victor M.},
  journal = {Computers \& Chemical Engineering},
  volume  = {187},
  pages   = {108750},
  year    = {2024},
  doi     = {10.1016/j.compchemeng.2024.108750}
}
```

## Authors
Gabriel Hernández Morales, Edgar Ivan Sanchez Medina, Arturo Jiménez-Gutiérrez, and Victor M. Zavala
