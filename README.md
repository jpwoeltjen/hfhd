
![logo](https://github.com/jpwoeltjen/hfhd/blob/master/docs/img/logo_full.png)
[![Documentation Status](https://readthedocs.org/projects/hfhd/badge/?version=latest)](https://hfhd.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/hfhd.svg)](https://badge.fury.io/py/hfhd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


hfhd is an accelerated Python library for estimating (integrated) covariance matrices with high frequency data in high dimensions. Its main focus lies on the estimation of covariance matrices of financial returns. This is a challenging task since high frequency data are observed irregularly (and non-synchronously across assets) and are contaminated with microstructure noise. The hfhd.hf module provides a collection of tools for synchronization and noise reduction.

When many assets are considered relative to the sample size, we say that the covariance matrix has high dimension. Then, sample eigenvalues are overdispersed relative to the population eigenvalues due to the curse of dimensionality. This overdispersion leads to an ill-conditioned covariance matrix estimate. The hfhd.hd module provides tools to improve the condition of the matrix. 

[Read the documentation.](https://hfhd.readthedocs.io/en/latest/)

# Install hfhd
```bash
$ pip install hfhd
```

# Developing hfhd
To install hfhd, along with the tools to develop and run tests, run the following command in your virtualenv after cloning the repo:

```bash
$ pip install -e .[dev]
```

