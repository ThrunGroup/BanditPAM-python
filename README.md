# BanditPAM

This repository contains an implementation of BanditPAM, and provides the code
used to generate the results for our NeurIPS 2020 submission.

# Requirements

This code requires `python >= 3.6.3` and `pip >= 20.1.1`. All other packages are in `requirements.txt`
and can be installed via `pip install -r requirements.txt`.

## Datasets

A number of datasets are required to recreate the results from the paper. See
below for instructions on how to acquire the datasets.

### MNIST

`pip install mnist` will install the `mnist` python package from
https://pypi.org/project/mnist/ and will enable loading of the MNIST dataset

### SCRNA

The single-cell RNA (SCRNA) sequence dataset used in the paper is available from
10x genomics at https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
-- "Gene / cell matrix (filtered)".

The SCRNA-PCA dataset referred to in the appendix contains the SCRNA dataset,
but projected onto the top 10 principal components, which is available from the
same webpage and is labeled "Clustering analysis".

These datasets have been preprocessed with `scrna_preprocessing.py`

### HOC4

The Code.org Hour of Code 4 (HOC4) dataset is available at https://code.org/research.
The HOC4 solution abstract syntax trees (ASTs) are available in zip file from
the link above, and the `.tree` files are also included in this zip archive.

#### Preprocessing the HOC4 dataset

The HOC4 ASTs were preprocessed into zss trees using `tree_utils.py`. For the
convenience of the reviewers, we have included the `.tree` files so they can be
used directly.

The HOC4 ASTs also require precomputation of the pairwise distance matrix
between trees; this matrix is included in `tree-3630.dist`.

To compute this matrix from scratch, run `python tree_utils.py -d HOC4`. Note
that this computation takes a while.

## ELKI Implementation of FastPAM

For comparison to FastPAM, an existing baseline, we use the ELKI implementation
available from https://elki-project.github.io/tutorial/. The GUI allows for the
use of FastPAM; to use FastPAM; open up the GUI, set the `dcb.in` field to be
the input data (CSV format), set `algorithm` to be clustering -> kmeans ->
KMedoidsFastPAM, and set `kmeans.k` to be the desired number of medoids.

# Introduction

# Quick Start

# Running Tests

# Running Experiments


# Creating Figures

# Information About Each File
