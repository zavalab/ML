# CMC_GCN

Graph convolutional neural networks that predict critical micelle concentrations of surfactants from molecular structure.

## Overview
This folder contains: 
- data (csv files containing the SMILES strings with the logCMC values)
- code (python script for dataset construction, models, and training)
- notebooks (jupyter notebooks to visualize data and results)
- saved_models (containing trained models)

## Model Training
```
run GNN_workflow.py
```
- specify input parameters as needed. For example, use the following to specify #epoch, batch size, and learning rate.
```
run GNN_workflow.py -e 100 -b 5 -l 0.005
```

## Reference
This code accompanies the paper:

Shiyi Qin, Tianyi Jin, Reid C. Van Lehn, and Victor M. Zavala. "Predicting Critical Micelle Concentrations for Surfactants Using Graph Convolutional Neural Networks." *The Journal of Physical Chemistry B* 125, no. 37 (2021): 10610-10620. https://doi.org/10.1021/acs.jpcb.1c05264

Preprint: https://doi.org/10.26434/chemrxiv.14384693

## BibTeX
```bibtex
@article{qin2021predicting,
  title   = {{Predicting Critical Micelle Concentrations for Surfactants Using Graph Convolutional Neural Networks}},
  author  = {Qin, Shiyi and Jin, Tianyi and Van Lehn, Reid C. and Zavala, Victor M.},
  journal = {The Journal of Physical Chemistry B},
  volume  = {125},
  number  = {37},
  pages   = {10610--10620},
  year    = {2021},
  doi     = {10.1021/acs.jpcb.1c05264}
}
```

## Authors
Shiyi Qin, Tianyi Jin, Reid C. Van Lehn, and Victor M. Zavala
