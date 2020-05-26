Welcome to hfhd's documentation!
================================

hfhd is an accelerated Python library for estimating (integrated) covariance matrices with high frequency data in high dimensions. Its main focus lies on the estimation of covariance matrices of financial returns. This is a challenging task since high frequency data are observed irregularly (and non-synchronously across assets) and are contaminated with microstructure noise. The :mod:`hf` module provides a collection of tools for synchronization and noise reduction.

When many assets are considered relative to the sample size, we say that the covariance matrix has high dimension. Then, sample eigenvalues are overdispersed relative to the population eigenvalues due to the curse of dimensionality. This overdispersion leads to an ill-conditioned covariance matrix estimate. The :mod:`hd` module provides tools to improve the condition of the matrix. 

Loss functions in the :mod:`loss` module help to judge the performance of covariance estimators. The :mod:`sim` module provides a Universe class with which heavy tailed, noisily and irregularly observed, high dimensional asset returns with an underlying factor model can be simulated. 

To install hfhd run

.. code-block:: console

   $ pip install hfhd

If you experience problems with the parallelism, you may need to install/upgrade the Threading Building Blocks (TBB). The easiest way is to

.. code-block:: console

   $ conda install tbb


The Price Process
-----------------

The library assumes the following standard model. Let $\left(\Omega, \mathcal{F},\left\{\mathcal{F}_{t}\right\}_{0 \leq t \leq T}, \mathbb{P}\right)$ be a filtered probability space on which the log-price process of the $p$ assets under study, $\left\{\mathbf{X}_{t}\right\}_{0 \leq t \leq T},$ is adapted, where $\mathbf{X}_{t}=\left(X_{t}^{(1)}, \ldots, X_{t}^{(p)}\right)^{\prime} .$ It is assumed that $\mathbf{X}_{t}$ follows a diffusion process satisfying
$$
d \mathbf{X}_{t}=\boldsymbol{\mu}_{t} d t+\boldsymbol{\sigma}_{t} d \mathbf{W}_{t}, \quad t \in[0,T].
$$

The process $\left\{\mathbf{W}_{t}\right\}$ is a $p$-dimensional standard Brownian motion. The drift $\boldsymbol{\mu}_{t} \in \mathbb{R}^{p}$ and the volatility $\boldsymbol{\sigma}_{t} \in \mathbb{R}^{p \times p}$ are càdlàg. Analogously to the covariance matrix in the low frequency setting, the integrated covariance matrix over the interval $[a, b] \subset[0,T],$ is defined as
$$
\boldsymbol{\Sigma}(a, b)=\int_{a}^{b} \boldsymbol{\sigma}_{t} \boldsymbol{\sigma}_{t}^{\prime} dt
$$
The log-price of each asset is observed at discrete times and with market microstructure noise. The set of observation times of asset $j$ is denoted as $t^{(j)} = \{t_{0}^{(j)} \leq t_{1}^{(j)} \leq \ldots \leq t_{n^{(j)}}^{(j)} \}$. Importantly, observation times are non-synchronous across assets, i.e., $t^{(j)} \neq t^{(k)}$ for $j \neq k$ in general. Furthermore, the concentration ratio $c_n = p/n$ is not a small number. The observed log-price process of asset $j$ is
\begin{equation}
\label{eqn:obs}
{Y_t^{(j)}}={X_t^{(j)}}+{\epsilon_t^{(j)}},  \quad t \in t^{(j)},
\end{equation}
where $\epsilon^{(j)}_t$ is a noise process independent of $\mathbf{X}_t$ with mean 0. $\left\{\epsilon_{t}\right\}_{0 \leq t \leq T}$ is assumed to be adapted to $\left\{\mathcal{F}_{t}\right\}_{0 \leq t \leq T}$. Hence, the observed price process $\left\{{Y}^{(j)}_{t}\right\}_{0 \leq t \leq T}$ is also adapted. The microstructure noise is the resulting process of an interplay of many effects. Among them are for example price discreteness and the bid–ask bounce. When the interest lies on the integrated covariance matrix of the underlying return process, it is important to account for the variance due to the noise process. When the observation frequency is high, the variance due to the the noise process dominates the variance of the underlying process and estimators that do not cancel the noise are severely biased. 

.. note::
    Accelerated functions are compiled Just In Time (JIT). This may cause the first call to be several orders of magnitude slower than the following calls.

Design Philosophy
-----------------

hfhd is written with the following priciples in mind:

 - Thorough and notationally consistent documentation, explaining not only the code but also the theory behind it.
 - The code needs to run sufficiently fast to be usable. Since many functions are inherently iterative, JIT compilation with Numba is used throughout to speed things up.
 - The API should be user friendly with consistent inputs and outputs.


.. toctree::
   :maxdepth: 2

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
