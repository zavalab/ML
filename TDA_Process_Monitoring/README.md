# Topological Data Analysis (TDA) for High-Dimensional Dynamic Process Monitoring

## Event Detection in Multivariate Time-Series Data using TDA

This repository contains only source code and a synthetic three-tank system example dataset for demonstrating detection of events in multivariate time-series using topological data analysis (TDA).

Author: Angan Mukherjee (amukherjee43@wisc.edu)

## Publication

This repository consists of scripts based on all monitoring approaches discussed in the research paper:

**Mukherjee, A.**, Soderstrom, T. A., Kurtz, M. J., and Zavala, V. M. "Topological Data Analysis for High-Dimensional Dynamic Process Monitoring" (*manuscript under review*)

## Repository Overview

This repository provides Python implementations of the monitoring methods discussed in the manuscript using a synthetic three-tank system dataset. The repository is designed to demonstrate how multivariate time-series data can be transformed into geometric / topological representations for detecting transient events during dynamic process operation.

The main focus is on comparing reconstruction-based (e.g., principal component analysis (PCA), autoencoders (AE), etc.) and trajectory-based (e.g., Koopman autoencoders (KAE), TDA-neural ordinary differential equation (TDA-NODE) approach, etc.) monitoring strategies. The reconstruction-based methods use learned low-dimensional representations and reconstruction errors to identify deviations from nominal process behavior. The trajectory-based methods analyze the evolution of process behavior directly in time, including approaches based on topological descriptors and latent-space dynamics.

The provided examples use only synthetic data generated from the three-tank system case study.

The repository also includes baseline monitoring workflows to support comparison with conventional reconstruction-based and trajectory-based approaches.

