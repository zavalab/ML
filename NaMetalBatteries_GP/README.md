# Data-Driven Electrolyte Design for Anode-Free Sodium Metal Batteries

This repository comprises scripts used in the research paper titled 
'Data-Driven Electrolyte Design for Anode-Free Sodium Metal Batteries'.


## Introduction

This repository contains the scripts used to perform a closed-loop Bayesian optimization of ether electrolyte formulations for anode-free sodium-ion batteries, plot the results using t-SNE, and test the robustness of this algorithm with a ground-truth model. The optimization model utilizes the volume fraction of ethers of a formulation (no chemical information) as Gaussian process model inputs to provide broad generalizability and accessibility to our methods. Experimental measurements were done in batches of 5 and follow the workflow shown below.

The data collected in this study is available in the `Experimental Data` folder to provide all tested formulation results to the broader scientific community. This consists of 45 formulations of varying mixtures of the electrolytes shown below.

<p align="center">
<img src="./Readme Figures/Active Learning TOC.jpg" alt="drawing" width="300"/> 
<img src="./Readme Figures/Electrolytes.jpg" alt="drawing" width="300"/> 
</p>


> Publication

- [In Progress](Link)

<br />

## Sample Output

<p align="center">
<img src="./Readme Figures/t-SNE by Iter.png" alt="drawing" width="600"/> 
</p>

## Tutorials

For detailed usage, navigate to the `Jupyter Notebooks` directory.

## Software Versions

- scikit-learn v1.6.1
- numpy v1.26.4
- pandas v1.4.4
- scipy v1.7.3
- matplotlib v3.4.3

## Links

- [Scalable Systems Lab](https://zavalab.engr.wisc.edu/)
- [Liu Lab](https://liulab.mse.wisc.edu/about/)
- [Van Lehn Group](https://vanlehngroup.che.wisc.edu/)

<br />