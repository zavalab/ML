# asthma
## Introduction
This repository contains code for combining 3D convolutional neural networks (CNNs) and support vector machines (SVMs) to predict the longitudinal progression of lung decline.

## Structure
- `data.ipynb` contains statistics on clinical biomarker data and CT scan data. It also includes code for segmenting lung volumes from volumetric CT scans and adjusting the resolution of CT scans.
- `result.ipynb` contains codes for generating all results, including SVM and CNN results with varying resolutions, losses, and hyperparameters, as well as hybrid model and importance map results.
- `figure.ipynb` contains the code necessary to reproduce all figures presented in the paper.

All Python files are located inside the `src` directory.
- `train_cnn.py` is the primary code for training a 3D CNN to predict lung function decline.
- `segmentation.py` is used to extract only the lung volumes from volumetric CT scans.
- `saliency.py` calculates the saliency map, which is the Integrated Gradients of the 3D CNN prediction.
- `data_utils.py` contains functions to process and load data.
- `result_utils.py` includes functions for calculating the results of each model.
- `hybrid_utils.py` includes functions for hybrid modeling, such as cascade, and-, or-, and avg-voting.
- `sal_utils.py` contains a function to visualize 3D saliency maps.
- `aux_utils.py` contains some helping functions.

## Training
To train a 3D CNN, you can

```bash
./train_cnn_gpu.sh
```

## Data
The dataset utilized in this study can be accessed by submitting a formal request to the SARP Data Coordinating Center.
