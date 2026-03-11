# Identifying the Impact of Chemical Functional Groups on Ionic Liquid Conductivity

## Introduction

Here we provide the code and data used for the manuscript "Identifying the Impact of Chemical Functional Groups on Ionic Liquid Conductivity". This work defines a molecular fragment representation that reflects charge-carrier resonance using SMARTS fragments to capture the electrostatic contributions to conductivity of ionic liquid compounds. We find this representation simplifies structure-conductivity relationships and improves predictive performance in low-data regimes. 

We additionally map the ionic liquid chemical design space using principal covariates regression to visualize the underlying structure-property relationships. From this analysis, we observe that ionic liquid anion core identity and cation functionalization have more significant roles in determining ionic liquid conductivity than ionic liquid cation core identity.

Overall, this work aims to provide an interpretable survey of structure-property relationships for currently available ionic liquid structures and provides new perspectives on how to design better ionic liquids for desirable transport behavior.

<br />

> Features

- `Shallow learning data science framework` for analyzing ionic liquid structure-property trends
- `Custom SMARTS representations` for describing ionic liquid ion transport
- `Principal covariates regression` (PCovR) latent space mapping for visualizing ionic liquid property trends

> Publication

- In Progress.

<br />

## Sample Outputs

> Ionic Liquid SMARTS Fragments

<br />
<img src="./ReadMe Figures/Ionic Liquid Fragments.png" />
<br />

> PCovR Latent Space

<br />
<img src="./ReadMe Figures/SMARTS PCovR.png" /> 
<br />


## Tutorials

For detailed usage, navigate to the `Jupyter Notebooks` directory.

## Software Versions

- SciKit-Learn v1.6.1
- SKMatter v0.2.0
- PubChemPy v1.0.4
- RDKit v2024.3.5
- ILThermo v2.0
- pyILT2 v0.9.8

## Links

- [Cersonsky Lab] (https://cersonsky.com/)
- [Gebbie Lab](https://interfaces.che.wisc.edu/)
- [Zavalab](https://zavalab.engr.wisc.edu/)
- [SciKit Matter] (https://scikit-matter.readthedocs.io/en/latest/)
- [RDKit](https://github.com/rdkit/rdkit)
- [IL Thermo](https://ilthermo.boulder.nist.gov/)

<br />