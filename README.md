# BanditPAM

This repository contains an implementation of BanditPAM, and provides the code
used to generate the results for our NeurIPS 2020 submission. This python code
is used for all experiments and to generate all figures in the paper. The C++
implementation is used for the wall clock time comparisons.

# Introduction

Throughout this repo, an "experiment" refers to a specific $k$-medoids problem
we wish to solve. Each experiment in this repository is specified by a number of
parameters:
- algorithm (ucb, also known as BanditPAM, or naive_v1, which refers to PAM)
- a string specifying whether to run the BUILD and/or SWAP steps
- verbosity
- num_medoids (k, e.g. 10)
- dataset subsample size (N, e.g. 30,000)
- dataset (e.g. MNIST)
- random seed (e.g. 42)
- metric (e.g. L2)
- intial medoids (usually blank)

When an experiment is run, a profiler is run and tallies the number of times the
distance function is called, as well as other statistics such as wall clock
time. These profiles are stored in files prefixed with `p-`.

Additionally, logfiles are written for every experiment. These logfiles contain
information like the medoids assigned after the BUILD step, the final medoids
chosen after the SWAP step, the final loss, as well as further details such as
the number of swaps performed. The logfiles are prefixed with `L-`.

All profiles and logfiles necessary to recreate the plots in the paper are
provided in the `/profiles/` directory. We also include instructions on how to
create these profiles and logfiles from scratch (see below).

# Requirements

This code requires `python >= 3.6.3` and `pip >= 20.1.1`. All other packages are in `requirements.txt`
and can be installed via `pip install -r requirements.txt`.

## Required Datasets

A number of datasets are required to recreate the results from the paper. See
below for instructions on how to acquire the datasets.

### MNIST

`pip install mnist` will install the `mnist` python package from
https://pypi.org/project/mnist/ and will enable loading of the MNIST dataset.

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

## Required: ELKI Implementation of FastPAM

For comparison to FastPAM, an existing baseline, we use the ELKI implementation
available from https://elki-project.github.io/tutorial/. The GUI allows for the
use of FastPAM; to use FastPAM; open up the GUI, set the `dcb.in` field to be
the input data (CSV format), set `algorithm` to be clustering -> kmeans ->
KMedoidsFastPAM, and set `kmeans.k` to be the desired number of medoids.

# Quick Start



# Running Tests

A few test cases have been implemented in `tests.py`. To run them, run:
`py.test tests.py`

Expected output:
```
======================================== test session starts ===================
platform darwin -- Python 3.6.3, pytest-4.3.0, py-1.8.0, pluggy-0.9.0
rootdir: /Users/motiwari/Desktop/FastPAM, inifile:
plugins: celery-4.2.1
collected 1 item

tests.py .                                                                [100%]

===================================== 1 passed in 30.71 seconds ================
```

# Running Experiments


# Creating Figures

# Information About Each File
