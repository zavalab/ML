# FastTopology

Fast, parallel C++/Python computation of the Euler characteristic, with timing studies on random fields, hyperspectral images, liquid crystals, and molecular dynamics data.

## Contents
- `fast_EC_functions.cpp`, `fast_EC_functions.hpp`: C++ implementation; `py_calls_to_cpp.py`: Python interface
- `EC_computation_funcs.py`: Python functions for the Euler characteristic
- `timing_for_*.py`: timing studies (random 2D/3D fields, low-memory 2D, hyperspectral images, liquid crystals, molecular dynamics)
- `data/`: data; `requirements.txt`: Python requirements

## Reference
This code accompanies the paper:

Daniel J. Laky and Victor M. Zavala. "A fast and scalable computational topology framework for the Euler characteristic." *Digital Discovery* 3, no. 2 (2024): 392-409. https://doi.org/10.1039/d3dd00226h

Preprint: https://arxiv.org/abs/2311.11740

## BibTeX
```bibtex
@article{laky2024fast,
  title   = {{A fast and scalable computational topology framework for the Euler characteristic}},
  author  = {Laky, Daniel J. and Zavala, Victor M.},
  journal = {Digital Discovery},
  volume  = {3},
  number  = {2},
  pages   = {392--409},
  year    = {2024},
  doi     = {10.1039/d3dd00226h}
}
```

## Authors
Daniel J. Laky and Victor M. Zavala
