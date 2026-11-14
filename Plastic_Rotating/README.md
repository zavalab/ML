# Plastic_Rotating

CNN analysis of mid-infrared spectra collected on a rotating platform for online characterization of mixed plastic waste, including prediction of unseen blend compositions.

## Overview
This repository contains 3 notebooks: 
- data (preprocess data and visulization)
- train (CNN training and unseen plastic blend composition prediction)
- result (result visulization)

You can find all the data, results and weights [here](https://drive.google.com/drive/folders/12weaMLGIwFKFGAOsPZI3p7HkYt67dsYs?usp=sharing)
- data (folder containing raw and preprocessed data)
- h5 (training weights of CNNs)
- pickle (result of training)
- unseen_pickle (result of unseen prediction)

## Requirements
Python 3.8, Tensorflow 2.3

## Reference
This code accompanies the paper:

Fei Long, Shengli Jiang, Adeyinka Gbenga Adekunle, Victor M Zavala, and Ezra Bar-Ziv. "Online Characterization of Mixed Plastic Waste Using Machine Learning and Mid-Infrared Spectroscopy." *ACS Sustainable Chemistry & Engineering* 10, no. 48 (2022): 16064-16069. https://doi.org/10.1021/acssuschemeng.2c06052

## BibTeX
```bibtex
@article{long2022online,
  title   = {{Online Characterization of Mixed Plastic Waste Using Machine Learning and Mid-Infrared Spectroscopy}},
  author  = {Long, Fei and Jiang, Shengli and Adekunle, Adeyinka Gbenga and M Zavala, Victor and Bar-Ziv, Ezra},
  journal = {ACS Sustainable Chemistry \& Engineering},
  volume  = {10},
  number  = {48},
  pages   = {16064--16069},
  year    = {2022},
  doi     = {10.1021/acssuschemeng.2c06052}
}
```

## Authors
Fei Long, Shengli Jiang, Adeyinka Gbenga Adekunle, Victor M Zavala, and Ezra Bar-Ziv
