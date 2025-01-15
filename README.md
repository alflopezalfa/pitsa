# PITSA - Python Interpretable Time Series Analysis

This repository contains the Python implementation of the **BAPC (Before and After Correction Parameter Comparison)** method for explainable time series analysis, as formulated in our paper:  
[Surrogate Modeling for Explainable Predictive Time Series Corrections](http://arxiv.org/abs/2412.19897).

## Overview

The BAPC is alocal surrogate approach for explainable time-series analysis. An initially non-interpretable predictive model to improve the forecast of a classical time-series 'base model' is used. 'Explainability' of the correction is provided by fitting the base model again to the data from which the error prediction is removed (subtracted), yielding a difference in the model parameters which can be interpreted.

### Comparison with LIME

For comparison purposes, we utilize the formulation of **LIME** as proposed in the paper:  
["LIME for Time Series"](https://arxiv.org/abs/2109.08438).  

The implementation of this formulation is available in the repository [ts-mule](https://github.com/dbvis-ukon/ts-mule). To simplify usage, we have included the relevant files from `ts-mule` in this repository under the folder `1_tsmule`.

## Repository Structure

```bash
├── 1_tsmule/          # Contains files from the ts-mule repository for convenience
├── 2_data/            # Datasets used in the paper
├── 3_code/            # Source code
├── 4_results/         # Resulting figures
├── README.md          # Project documentation (this file)
```

## Instalation

1. bar
2. foo
3. baz




