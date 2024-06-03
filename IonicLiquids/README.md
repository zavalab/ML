# A Data Science Framework for the Analysis of Ion Transport Mechanisms in Ionic Liquids

## Introduction

Here, we provide the scripts used to model and analyze ionic liquid conductivity using the Nernst-Einstein hydrodynamic transport model, the modified Arrhenius kinetic transport model, t-stochastic neighbor embedding (t-SNE) dimensionality reduction, and machine learning modeling with various forms of chemical information inputs (molecular connectivity, 3D moleculear descriptors, and bulk properties). We provide a framework for analyzing poorly-understood ionic liquid proprties using easily accessible chemical information. In this work, we first contrast the accuracies of the Nernst-Einstein model and modified Arrhenius model predictions to understand which mechanistic model best describes ion transport in ionic liquids. We then use t-SNE to project ionic liquid molecular structure into a visualizable 2D latent space and analyze structure-conductivity relationships. Finally, we use machine learning models to test the our ability to predict ionic liquid conductivity using common chemical descriptors, and we analyze our models to identify which molecular descriptors and bulk properties most influence ion transport.

The databases we created for this analysis can be found in the `Databases` folder and combines **RDKit** descriptors for simulated single ions, reported **PubChem** molecular descriptors, and experimentally measured bulk properties from **ILThermo**. Multiple datasests are available for the various ionic liquid properties modeled; however our primary database of interest for conductivity contains *2,371* temperature dependent data points for *218* ionic liquids. Tutorials for our framework and results can be found in the `Jupyter Notebooks` folder.

<br />

> Features

- `Data science framework` for analyzing ionic liquid properties
- `Databases` from RDKit, PubChem, and ILThermo containing properties and molecular descriptors for **218** ionic liquids and **2,371** temperature dependent data points
- `Bulk property  predictions` for ionic liquids given their **SMILES strings** and **RDKit Simulations**
- `Descriptor analysis` to evaluate predictive capabilities and identify underlying ion transport mechanisms
- `Classical model analyses` to contrast ion transport mechanisms in ionic liquids

> Publication

- [In Progress](Link)

<br />

## Sample Outputs

<br />

> Classical and Machine Learning Modeling

<br />
<img src="./Figures/Readme_Model Fits.png" /> 
<br />
<br />

> t-SNE Property Mapping
<br />

<img src="./Figures/Readme_Six t-SNE Properties White.png" />

<br />

## Tutorials

For detailed usage, navigate to the `Jupyter Notebooks` directory.


## Links

- [Zavalab](https://zavalab.engr.wisc.edu/)
- [Gebbie Lab](https://interfaces.che.wisc.edu/)
- [RDKit](https://github.com/rdkit/rdkit)
- [PubChem 3D](https://pubchem.ncbi.nlm.nih.gov/docs/pubchem3d)
- [IL Thermo](https://ilthermo.boulder.nist.gov/)

<br />