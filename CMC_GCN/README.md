# CMC GCN

## Overview
This repository contains 4 directories: 
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
