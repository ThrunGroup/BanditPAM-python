# WARNING

This is almost certainly not the repo you're looking for. You probably want
https://github.com/ThrunGroup/BanditPAM.

This repo only exists as a reference Python implementation, and for the
reproducibility of some of the figures in the paper.

If you use this software, please cite:

Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time *k*-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.

```python
@inproceedings{BanditPAM,
  title={Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits},
  author={Tiwari, Mo and Zhang, Martin J and Mayclin, James and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

# BanditPAM

This repository contains a reference python implementation of BanditPAM, and provides the code
used to generate the results for our NeurIPS 2020 submission. This python code
is used for to generate some of the figures in the paper. The C++
implementation at https://github.com/ThrunGroup/BanditPAM is used for the wall clock time comparisons,
and is almost certainly what you're looking for (it is also callable from Python).

In this repo, the file `ucb_pam.py` implements BanditPAM.

# Introduction

Throughout this repo, an "experiment" refers to a specific k-medoids problem
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

When an experiment is conducted, a profiler is run and tallies the number of times the
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

This code requires `python >= 3.6.3` and `pip >= 20.1.1`. All other packages are
in `requirements.txt` and can be installed via `pip install -r requirements.txt`.

If you are using a virtualenv, please ensure that you are able to run
`python --version` and `pip --version` from your virtualenv. There are known
issues with virtualenvs on Mac OSX, unrelated to this work (for example, openssl
errors that manifest with errors like `cannot import name md5`). See
https://stackoverflow.com/questions/59123154/importerror-cannot-import-name-md5

## Required Datasets

A number of datasets are required to recreate the results from the paper. See
below for instructions on how to acquire the datasets.

### MNIST

`pip install mnist` will install the `mnist` python package from
https://pypi.org/project/mnist/ and will enable loading of the MNIST dataset.

### SCRNA

The single-cell RNA (SCRNA) sequence dataset used in the paper is available from
10x genomics at https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
-- the dataset is "Gene / cell matrix (filtered)".

The SCRNA-PCA dataset referred to in the appendix contains the SCRNA dataset,
but projected onto the top 10 principal components, which is available from the
same webpage and is labeled "Clustering analysis".

These datasets have been preprocessed with `preprocess_scrna.py`, which expects
the files to be saved in specific locations (see the constants at top of file).

### HOC4

The Code.org Hour of Code 4 (HOC4) dataset is available at https://code.org/research.
The HOC4 solution abstract syntax trees (ASTs) are available in zip file from
the link above, and the `.tree` files are also included in this zip archive.

#### Preprocessing the HOC4 dataset

The HOC4 ASTs were preprocessed into zss trees using `tree_utils.py`. For the
convenience of the reviewers, we have included the `.tree` files so they can be
used directly.

The HOC4 ASTs also require precomputation of the pairwise distance matrix
between trees. To compute this matrix from scratch, run `python tree_utils.py -d HOC4`. Note
that this computation takes several hours.

## Required: ELKI Implementation of FastPAM

For comparison to FastPAM, an existing baseline, we use the ELKI implementation
available from https://elki-project.github.io/releases/. A tutorial is available
at https://elki-project.github.io/tutorial/. The GUI allows for the
use of FastPAM; to use FastPAM; open up the GUI, set the `dcb.in` field to be
the input data (CSV format), set `algorithm` to be clustering -> kmeans ->
KMedoidsFastPAM, and set `kmeans.k` to be the desired number of medoids.

# Quick Start

It is possible to run the algorithms directly from their scripts. For example,

`python ucb_pam.py -k 5 -N 1000 -s 42 -d MNIST -m L2 -p`

will run BanditPAM (implemented in ucb_pam) to find 3 medoids in a subset of
size 1000 chosen from MNIST with random seed 42 and using L2 distance, while
using the FastPAM1 optimization (`-p`).

Expected output:

```
>>> python ucb_pam.py -k 5 -N 1000 -s 42 -d MNIST -m L2 -p
Computed exactly for: 143.0
Computed exactly for: 332.0
Computed exactly for: 337.0
Computed exactly for: 330.0
Computed exactly for: 263.0
Built medoids [285, 41, 390, 424, 306]
Computed exactly for: 1422.0
Computed exactly for: 1431.0
Computed exactly for: 954.0
Computed exactly for: 1131.0
Final medoids [41, 306, 324, 369, 492]
```

# Running Multiple Experiments

It is possible to run multiple experiments at once to obtain their profiles and
logfiles, including in parallel. `run_profiles.py` takes in an experiment config,
which contains a list of experiments to run, and will run each of them.

To generate an experiment config programmatically, use `python generate_config.py`.
The auto-generated config will be stored in `auto_exp_config.py`. Different sets
of experiments (e.g. for MNIST, SCRNA, etc.) can be generated by (un)commenting
the relevant lines of `generate_config.py`. All of the experiments can then be
run via:

`python run_profiles.py -e auto_exp_config.py -p`

where `-p` indicates that we should use the FastPAM1 optimization (which is
strongly suggested).

To run the experiments in parallel instead of sequentially, please see the
comments in `run_profiles.py`.

# Creating The Figures

All of the necessary profiles to recreate the figures from the paper are already
included in this zip archive. Instructions on recreating each plot are below:

To remake Figure 1(a), run `python make_loss_plots.py`. This retrieves the final
losses for each algorithm from the corresponding logfiles. In the case of
FastPAM, however, the losses were manually computed by inputting the different
subsets into the ELKI GUI. See `make_manual_MNIST_subsets.py` for instructions
on how to programatically create the different random subsets of MNIST for use
in the ELKI GUI version of FastPAM.

To remake Figures 1(b) and 1(c), run `python parse_profiles.py`, (un)commenting
the code for the necessary specific set of experiments in `main()` and changing
the filename in the call to `savefig` in `plot_slice_sns()`. Figures 2(a), 2(b),
and 2(c) are generated in the same way.

To remake Appendix Figure 2, 3, 4, and 5, run `python make_sigma_plots.py`,
modifying the dataset and metric specified in `main()` as necessary.

Appendix Figure 6 is remade using `python parse_profiles.py` as specified above.

# Information About Each File

## Algorithms
- `clarans.py` : implementation of CLARANS algorithm
- `em_style.py` : implementation of EM-style k-medoids algorithm (Voronoi iteration)
- `naive_pam_v1.py` : implementation of PAM
- `ucb_pam.py` : implementation of BanditPAM

## Running experiments
- `generate_config.py` : used to programatically generate a list of experiments
  to run
- `auto_exp_config.py`: experiment configs automatically generated by
  `generate_config.py`, for use with `run_profiles.py`
- `run_profiles.py` :

## Helpers
- `data_utils.py` : A number of common functions used by several algorithms
- `tree_utils.py` : Helper methods to parse the HOC4 ASTs
- `make_manual_MNIST_subsets.py` : Helper functions to create the subsets of
  MNIST to be used with the ELKI GUI implementation of FastPAM

## Parsing Results and Plotting
- `snakevizcode.py` : Methods to parse the profiles created with cProfile
- `parse_profiles.py` : Used for plotting scaling with N
- `make_loss_plots.py` : Used for comparing relative losses of algorithms
  (Figure 1(a))
- `make_sigma_plots.py` : Used for Appendix Figures (described above)

## Miscellaneous
- `requirements.txt` : Specification of packages to install via
  `pip install -r requirements.txt`
- `manual_fastpam_losses.txt` : the results (final losses) on the subsets of
  MNIST obtained by FastPAM, determined by using the ELKI GUI
