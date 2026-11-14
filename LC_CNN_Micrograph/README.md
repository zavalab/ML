# LC_CNN_Micrograph

Convolutional neural networks that classify optical micrographs of contact-printed liquid crystal droplet arrays to detect and identify aqueous surfactants (SDS, DTAB, SDS+DTAB, rhamnolipid).

## Contents
- `code/`: image segmentation, dataset generation, CNN models, and training (`run_model.py`)
- `models/`: trained models (`level1/`, `level2/`)
- `data_upload.zip`: micrographs used for training and validation; the naming of the files is explained in `data_upload_readme.pdf`

## Reference
This code accompanies the paper:

Fengrui Wang, Shiyi Qin, Claribel Acevedo-Vélez, Reid C. Van Lehn, Victor M. Zavala, and David M. Lynn. "Decoding Optical Responses of Contact-Printed Arrays of Thermotropic Liquid Crystals Using Machine Learning: Detection and Reporting of Aqueous Amphiphiles with Enhanced Sensitivity and Selectivity." *ACS Applied Materials & Interfaces* 15, no. 43 (2023): 50532-50545. https://doi.org/10.1021/acsami.3c12905

Preprint: https://doi.org/10.26434/chemrxiv-2023-ns10h

## BibTeX
```bibtex
@article{wang2023decoding,
  title   = {{Decoding Optical Responses of Contact-Printed Arrays of Thermotropic Liquid Crystals Using Machine Learning: Detection and Reporting of Aqueous Amphiphiles with Enhanced Sensitivity and Selectivity}},
  author  = {Wang, Fengrui and Qin, Shiyi and Acevedo-Vélez, Claribel and Van Lehn, Reid C. and Zavala, Victor M. and Lynn, David M.},
  journal = {ACS Applied Materials \& Interfaces},
  volume  = {15},
  number  = {43},
  pages   = {50532--50545},
  year    = {2023},
  doi     = {10.1021/acsami.3c12905}
}
```

## Authors
Fengrui Wang, Shiyi Qin, Claribel Acevedo-Vélez, Reid C. Van Lehn, Victor M. Zavala, and David M. Lynn
