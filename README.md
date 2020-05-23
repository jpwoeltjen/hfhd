
![logo](https://github.com/jpwoeltjen/hfhd/blob/master/docs/img/logo_full.png)
hfhd is an accelerated Python library for estimating (integrated) covariance matrices with high frequency data in high dimensions. Its main focus lies on the estimation of covariance matrices of financial returns. This is a challenging task since high frequency data are observed irregularly (and non-synchronously across assets) and are contaminated with microstructure noise. The hfhd.hf module provides a collection of tools for synchronization and noise reduction.

When many assets are considered relative to the sample size, we say that the covariance matrix has high dimension. Then, sample eigenvalues are overdispersed relative to the population eigenvalues due to the curse of dimensionality. This overdispersion leads to an ill-conditioned covariance matrix estimate. The hfhd.hd module provides tools to improve the condition of the matrix. 

# Install hfhd
```bash
$ pip install hfhd
```

# Developing hfhd
To install hfhd, along with the tools to develop and run tests, run the following command in your virtualenv:

```bash
$ pip install -e .[dev]
```

[Read the documentation.](https://hfhd.readthedocs.io/en/latest/)