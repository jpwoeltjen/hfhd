"""
The hf module provides functions for synchronization of asynchronously
observed multivariate time series and observation noise cancelling. When
possible, functions are parallelized and accelerated via JIT compilation
with Numba. By default all cores of your machine are used. If your pipeline
allows for parallelization on a higher level, it is preferable to do so. You
may manually set the number of cores used by setting ``numba.set_num_threads(n)``.
Every estimator takes ``tick_series_list`` as the first argument.
This is a list of pd.Series (one for each asset) containing
tick log-prices with pandas.DatetimeIndex. If you want to comute the covariance
of residuals after predictions are subtracted from log-returnsjust cumsum
the residuals. The output is the integrated covariance matrix
estimate as a 2d numpy.ndarray.
"""

import numpy as np
import pandas as pd
from hfhd import hd
import numba
from numba import prange
import warnings


def refresh_time(tick_series_list):
    r"""
    The all-refresh time scheme of Barndorff-Nielsen et al. (2011).
    If this function is applied to two assets at a time, it becomes the
    pairwise-refresh time. The function is accelerated via JIT compilation
    with Numba.

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick prices of one asset with datetime index.

    Returns
    -------
    out : pd.DataFrame
        Synchronized previous ticks according to the refresh-time scheme.

    Notes
    -----
    Multivariate estimators require synchronization of the time series.
    This can be achieved via a grid. A grid is a subset of $[0, T]$ and it is
    defined as
    \begin{equation}
    \mathcal{V}=\left\{v_{0}, v_{1}, \ldots, v_{\tilde{n}}\right\}\subset[0, T]
    \end{equation}
    with $v_{0}=0$ and $v_{\tilde{n}}=T,$ where $\tilde{n}$ is the sampling
    frequency, i.e., the number of grid intervals. Two prominent ways to
    specify the grid are (i) a regular grid, where  $v_{m}-v_{m-1}=\Delta v,
    \text{ for } m=1, \ldots, \tilde{n}$, and (ii) a grid based on
    'refresh times' of Barndorff et al. (2011), where the grid spacing is
    dependent on the observation times. If more than two assets are considered,
    refresh times can be further classified into 'all-refresh-times' and
    'pairwise-refresh times'. Estimators based on pairwise-refresh times use
    the data more efficiently but the integrated covariance matrix estimate
    might not be positive definite. The pairwise-refresh time
    $\mathcal{V}_p=\left\{v_{0}, v_{1}, \ldots, v_{\tilde{n}}\right\}$ can be
    obtained by setting $v_{0}=0,$ and
    \begin{equation}
    v_{m}=\max \left\{\min \left\{t^{(k)}_i
    \in t^{(k)}: t^{(k)}_i > v_{m-1}\right\},\min \left\{t^{(l)}_i
    \in t^{(l)}: t^{(l)}_i > v_{m-1}\right\}\right\}
    \end{equation}
    where $\tilde{n}$ is the total number of refresh times in the interval
    $(0,1].$ This scheme is illustrated in the figure. The
    procedure has to be repeated for every asset pair. In contrast, the
    all-refresh time scheme uses a single grid for all assets, which is
    determined based on the trade time of the slowest asset of each grid
    interval. Hence, the spacing of grid elements can be much wider. This
    implies that estimators based on the latter scheme may discard a large
    proportion of the data, especially if there is a very slowly trading asset.
    In any case,
    there has to be at least one observation time of each asset between any two
    grid elements. With that condition in mind, the 'previous tick time' of
    asset $j$ is defined as
    \begin{equation}
    \tau^{(j)}_m=\max \left\{ t^{(j)}_i \in t^{(j)}:
    t^{(j)}_i \leq v_{m}\right\}
    \end{equation}
    The following diagram illustrates the scheme for two assets, $k$ and $l$.

    .. tikz::

        \draw
        (0,1.75) -- (11,1.75)
        (0,-0.75) -- (11,-0.75)
        (0,1.5) -- (0,2)
        (1.9,1.5) -- (1.9,2)
        (3.5,1.5) -- (3.5,2)
        (5,1.5) -- (5,2)
        (6.5,1.5) -- (6.5,2)
        (8,1.5) -- (8,2)
        (10.8,1.5) -- (10.8,2)
        (0,-0.5) -- (0,-1)
        (1.9,-0.5) -- (1.9,-1)
        (5.7,-0.5) -- (5.7,-1)
        (10.3,-0.5) -- (10.3,-1);
        \draw[dashed,gray]
        (0,3.75) -- (0,-2.75) node[below] {$\nu_0=0$}
        (1.9,3.75) -- (1.9,-2.75) node[below] {$\nu_1$}
        (5.7,3.75) -- (5.7,-2.75) node[below] {$\nu_2$}
        (9.5,3.75) -- (9.5,-2.75) node[below] {$t_{3}^{(l)}=
        \tau_{3}^{(l)}=\nu_3 = T$};
        \draw[dashed] (11,1.75) -- (12,1.75)
              (11,-0.75) -- (12,-0.75);
        \draw[very thick] (9.5,-1.4) -- (9.5,0.25)
              (9.5,0.8) -- (9.5,2.4);
        \draw
        (0,1) node{$t_{0}^{(k)} = \tau_{0}^{(k)}$}
        (1.9,1) node{$t_{1}^{(k)} = \tau_{1}^{(k)}$}
        (3.5,1) node{$t_{2}^{(k)} $}
        (5,1) node{$t_{3}^{(k)} = \tau_{2}^{(k)}$}
        (6.5,1) node{$t_{4}^{(k)}$}
        (8,1) node{$t_{5}^{(k)}= \tau_{3}^{(k)}$}
        (11,1) node{$t_{6}^{(k)}$}
        (9.5,0.5) node{\textbf{$T$}}
        (0,0) node{$t_{0}^{(l)} = \tau_{0}^{(l)}$}
        (1.9,0) node{$t_{1}^{(l)} = \tau_{1}^{(l)}$}
        (5.7,0) node{$t_{2}^{(l)}= \tau_{2}^{(l)}$}
        (10.3,0) node{$t_{4}^{(l)}$};
        \draw
        (0,1.75) node[left,xshift=-0pt]{$X^{(k)}$}
        (0,-0.75) node[left,xshift=-0pt]{$X^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (0,2)--(1.9,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{\tau^{(k)}_1}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (1.9,2)--(5,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{\tau^{(k)}_2}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (5,2)--(8,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{\tau^{(k)}_3}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (9.5,-1)--(5.7,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{\tau^{(l)}_3}^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (5.7,-1)--(1.9,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{\tau^{(l)}_2}^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (1.9,-1)--(0,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{\tau^{(l)}_1}^{(l)}$};

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite
    estimators of the covariation of equity prices with noise and
    non-synchronous trading, Journal of Econometrics 162(2): 149–169.

    Examples
    --------
    >>> np.random.seed(0)
    >>> n = 20
    >>> returns = np.random.multivariate_normal([0, 0], [[1,0.5],[0.5,1]], n)/n**0.5
    >>> prices = np.exp(returns.cumsum(axis=0))
    >>> # sample n/2 (non-synchronous) observations of each tick series
    >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index().rename('a')
    >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index().rename('b')
    >>> previous_ticks = refresh_time([series_a, series_b])
    >>> np.round(previous_ticks.values,4)
    array([[0.34  , 0.4309],
           [0.2317, 0.4313],
           [0.1744, 0.4109],
           [0.1336, 0.3007],
           [0.1383, 0.4537],
           [0.1292, 0.1665],
           [0.0936, 0.162 ]])
    """

    if (len(tick_series_list) < 2):
        raise ValueError(
         'tick_series_list should be a list containing at least two pd.Series.')
    indeces = tuple([np.array(x.dropna().index, dtype='uint64') for x in tick_series_list])
    values = tuple([x.dropna().to_numpy(dtype='float64') for x in tick_series_list])
    rt_data, index = _refresh_time(indeces, values)
    index = pd.to_datetime(index)
    return pd.DataFrame(rt_data, index=index).dropna()


@numba.njit
def _refresh_time(indeces, values):
    """
    The computationally expensive iteration of :func:`~refresh_time`
    is accelerated with Numba.

    Parameters
    ----------
    indeces : a tuple or list of numpy.ndarrays, int64
        The length is equal to the number of assets. Each numpy.ndarray contains
        the unix time of ticks of one asset.
    values : a tuple or list of numpy.ndarrays, float64
        Each numpy.ndarray contains the prices of ticks of one asset.

    Returns
    -------
    merged_values : numpy.ndarray
        Synchronized previous ticks according to the refresh-time scheme.
    merged_index
        The refresh times.

    """
    # get a sorted main index with all unique trade times
    merged_index = indeces[0]
    for index in indeces[1:]:
        merged_index = np.append(merged_index, index)
    merged_index = np.sort(np.unique(merged_index))

    # Initialize the merged_values array.
    merged_values = np.empty((merged_index.shape[0], len(values)))
    merged_values[:, :] = np.nan

    # Initialize the values array. These are the previous ticks.
    last_values = np.empty(merged_values.shape[1])
    last_values[:] = np.nan

    for i in range(merged_values.shape[0]):
        for j in range(merged_values.shape[1]):
            index = indeces[j]
            loc = np.searchsorted(index, merged_index[i])

            # if there was a trade of asset j update the last_value
            # make sure that loc < values[j].shape[0] since numba
            # will not raise an out-of-bounds error but will put some
            # random value currently in memory.
            if index[loc] == merged_index[i] and loc < values[j].shape[0]:
                last_values[j] = values[j][loc]

        # if all assets traded at least once since the last refresh
        # time, a new grid point is formed and the clock starts anew.
        if not np.isnan(last_values).any():
            merged_values[i, :] = last_values
            last_values[:] = np.full_like(last_values, np.nan)

    return merged_values, merged_index


def preaverage(data, K=None, g=None, return_K=False):
    r"""
    The preaveraging scheme of Podolskij and Vetter (2009). It uses the fact
    that if the noise is i.i.d with zero mean, then averaging a rolling window
    of (weighted) returns diminishes the effect of microstructure noise on the
    variance estimate.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        A time series of log-returns. If multivariate, the time series
        has to be synchronized (e.g. with :func:`~refresh_time`).
    K : int, default = ``None``
        The preaveraging window length. ``None``
        implies :math:`K=0.4 n^{1/2}` is chosen as recommended in
        Hautsch & Podolskij (2013).
    g : function, default = ``None``
        A weighting function. ``None`` implies
        :math:`g(x) = min(x, 1-x)` is chosen.

    Returns
    -------
    data_pa : pd.Series
        The preaveraged log-returns.

    Notes
    -----
    The preaveraged log-returns using the window-length :math:`K` are given by

    .. math::
        \begin{equation}
        \begin{aligned}
        \bar{\mathbf{Y}}_{i}=\sum_{j=1}^{K-1} g\left(\frac{j}{K}\right)
        \Delta_{i-j+1}\mathbf{Y}, \quad \text { for } i=K, \ldots, n,
        \end{aligned}
        \end{equation}

    where :math:`\mathbf{Y}_i` have been synchronized beforehand, for example
    with :func:`~refresh_time`. Note that the direction of the moving window
    has been reversed compared to the definition in Podolskij and Vetter (2009)
    to stay consistent within the package. :math:`g` is a weighting function.
    A popular choice is

    .. math::
        \begin{equation}
        g(x)=\min (x, 1-x).
        \end{equation}

    References
    ----------
    Podolskij, M., Vetter, M., 2009.
    Estimation of volatility functionals in the simultaneous presence of
    microstructure noise and jumps.
    Bernoulli 15 (3), 634–658.
    """

    index = data.index

    # if univariate add axis
    if len(data.shape) == 1:
        data = data.to_numpy()[:, None]
    else:
        data = data.to_numpy()

    n, p = data.shape

    if K is None:
        K = int(np.sqrt(n)*0.4)

    if g is None:
        g = _numba_minimum

    weight = g(np.arange(1, K)/K)
    data_pa = _preaverage(data, weight)

    if p == 1:
        data_pa = pd.Series(data_pa.flatten(), index=index)
    else:
        data_pa = pd.DataFrame(data_pa, index=index)

    if return_K:
        return data_pa, K
    else:
        return data_pa


@numba.njit(cache=False, parallel=False, fastmath=False)
def _preaverage(data, weight):
    """
    Preaverage an observation matrix with shape = (n, p) given a weight vector
    with shape = (K-1, p).

    Parameters
    ----------
    data : numpy.ndarray, shape = (n, p)
        The observation matrix of synchronized log-returns.
    weight : numpy.ndarray, shape = (K-1, )
        The weight vector, looking back K -2 time steps.

    Returns
    -------
    data_pa : numpy.ndarray, shape = (n, p)
        The preaveraged returns.


    """

    n, p = data.shape
    K = weight.shape[0] + int(1)
    data_pa = np.full_like(data, np.nan)
    for i in prange(K-1, n):
        for j in range(p):
            data_pa[i, j] = np.dot(weight, data[i-K+2:i+1, j])
    return data_pa


@numba.njit
def _upper_triangular_indeces(p):
    """Get the upper triangular indeces of a square matrix. int16 should
    suffice for even the largest ``p`` encountered in practice.

    Parameters
    ----------
    p : int
        The dimension of the square matrix.

    Returns
    -------
    idx : numpy.ndarray, shape(int((p*(p+1)/2), 2)
        The array of indeces. i in zeroth column, j in first column.
    """
    s = 0
    idx = np.zeros((int((p*(p+1)/2)), 2), dtype=np.int16)
    for i in range(p):
        for j in range(i, p):
            idx[s] = i, j
            s += 1
    if idx[-1, 0] <= 0:
        raise ValueError("Got negative index, ``p`` probably too large for int16")
    return idx


def _get_indeces_and_values(tick_series_list):
    """
    Get indeces and values each as 2d numpy.ndarray from a list of
    pd.Series.

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains ticks of one asset with datetime index.

    Returns
    -------
    indeces :  numpy.ndarray, dtype='uint64', shape = (p, n_max)
        where p is the number of assets and n_max is the length of the
        longest pd.Series.
    values : numpy.ndarray, dtype='float64', shape = (p, n_max)
        where p is the number of assets and n_max is the length of the
        longest pd.Series.

    """
    n_max = np.max([len(x) for x in tick_series_list])
    indeces = np.empty((len(tick_series_list), n_max), dtype='uint64')
    indeces[:, :] = np.nan
    values = np.empty((len(tick_series_list), n_max), dtype='float64')
    values[:, :] = np.nan
    for i, x in enumerate(tick_series_list):
        idx = np.array(x.dropna().index, dtype='uint64')
        v = np.array(x.dropna().to_numpy(), dtype='float64')
        indeces[i, :idx.shape[0]] = idx[:]
        values[i, :idx.shape[0]] = v[:]
    return indeces, values


def get_cumu_demeaned_resid(price, y_hat=None):
    r"""
    From a pd.Series of tick prices and predictions get a pd.Series of
    tick log-prices with zero-mean returns, i.e. the reconstructed
    log-prices from de-meaned log-return residuals. These log-prices are inputs
    to the integrated covariance matrix estimators.

    Parameters
    ----------
    series : pd.Series
        Tick prices of one asset with datetime index.
    y_hat : pd.Series
        The predictions.

    Returns
    -------
    out : pd.Series
        Log-prices corresponding to zero-mean returns.
    """
    y = np.log(price.dropna()).diff()
    resid = y - y.mean()

    if y_hat is not None:
        resid -= y_hat - y_hat.mean()

    return resid.cumsum()


def msrc(tick_series_list, M=None, N=None, pairwise=True):
    r"""
    The multi-scale realized volatility (MSRV) estimator of Zhang (2006).
    It is extended to multiple dimensions following Zhang (2011).
    If ``pairwise=True`` estimate correlations with pairwise-refresh time
    previous ticks and variances with all available ticks for each asset.

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick-log-prices of one asset
        with datetime index. Must not contain nans.
    M : int, >=1, default=None
        The number of scales
        If ``M=None`` all scales :math:`i = 1, ..., M` are used, where M is
        chosen :math:`M = n^{1/2}` acccording to Eqn (34) of Zhang (2006).
    N : int, >=0, default=None
        The constant $N$ of Tao et al. (2013)
        If ``N=None`` :math:`N = n^{1/2}`. Lam and Qian (2019) need
        :math:`N = n^{2/3}` for non-sparse integrated covariance matrices,
        in which case the rate of convergence reduces to $n^{1/6}$.
    pairwise : bool, default=True
        If ``True`` the estimator is applied to each pair individually. This
        increases the data efficiency but may result in an estimate that is
        not p.s.d.

    Returns
    -------
    out : numpy.ndarray
        The mrc estimate of the integrated covariance matrix.

    Examples
    --------
    >>> np.random.seed(0)
    >>> n = 200000
    >>> returns = np.random.multivariate_normal([0, 0], [[1,0.5],[0.5,1]], n)/n**0.5
    >>> prices = 100*np.exp(returns.cumsum(axis=0))
    >>> # add Gaussian microstructure noise
    >>> noise = 10*np.random.normal(0, 1, n*2).reshape(-1, 2)*np.sqrt(1/n**0.5)
    >>> prices +=noise
    >>> # sample n/2 (non-synchronous) observations of each tick series
    >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index()
    >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index()
    >>> # get log prices
    >>> series_a = np.log(series_a)
    >>> series_b = np.log(series_b)
    >>> icov = msrc([series_a, series_b], M=1, pairwise=False)
    >>> icov_c = msrc([series_a, series_b])
    >>> # This is the biased, uncorrected integrated covariance matrix estimate.
    >>> np.round(icov, 3)
    array([[11.553,  0.453],
           [ 0.453,  2.173]])
    >>> # This is the unbiased,  corrected integrated covariance matrix estimate.
    >>> np.round(icov_c, 3)
    array([[0.985, 0.392],
           [0.392, 1.112]])

    Notes
    -----
    Realized variance estimators based on multiple scales exploit the
    fact that the proportion of the observed realized variance over a
    specified interval due to microstructure noise increases with the sampling
    frequency, while the realized variance of the true underlying process stays
    constant. The bias can thus be corrected by subtracting a high frequency
    estimate, scaled by an optimal weight, from a medium frequency estimate.
    The weight is chosen such that the large bias in the high frequency
    estimate, when scaled by the weight, is exactly equal to the medium bias,
    and they cancel each other out as a result.

    By considering $M$ time scales, instead of just two as in :func:`~tsrc`,
    Zhang2006 improves the rate of convergence to $n^{-1 / 4}$.
    This is the best attainable rate of convergence in this setting.
    The proposed multi-scale realized volatility (MSRV) estimator is defined as
    \begin{equation}
    \langle\widehat{X^{(j)}, X^{(j)}}\rangle^{(MSRV)}_T=\sum_{i=1}^{M}
    \alpha_{i}[Y^{(j)}, Y^{(j)}]^{\left(K_{i}\right)}_T
    \end{equation}
    where $\alpha_{i}$ are weights satisfying
    \begin{equation}
    \begin{aligned}
    &\sum \alpha_{i}=1\\
    &\sum_{i=1}^{M}\left(\alpha_{i} / K_{i}\right)=0
    \end{aligned}
    \end{equation}
    The optimal weights for the chosen number of scales $M$, i.e.,
    the weights that minimize the noise variance contribution, are given by
    \begin{equation}
    a_{i}=\frac{K_{i}\left(K_{i}-\bar{K}\right)}
    {M \operatorname{var}\left(K\right)},
    \end{equation}
    where
    %$\bar{K}$ denotes the mean of $K$.
    $$\bar{K}=\frac{1}{M} \sum_{i=1}^{M} K_{i} \quad \text { and } \quad
    \operatorname{var}\left(K\right)=\frac{1}{M}
    \sum_{i=1}^{M} K_{i}^{2}-\bar{K}^{2}.
    $$
    If all scales are chosen, i.e., $K_{i}=i$, for $i=1, \ldots, M$, then
    $\bar{K}=\left(M+1\right) / 2$ and $\operatorname{var}\left(K\right)=
    \left(M^{2}-1\right) / 12$, and hence

    \begin{equation}
    a_{i}=12 \frac{i}{M^{2}} \frac{i / M-1 / 2-1 /
    \left(2 M\right)}{1-1 / M^{2}}.
    \end{equation}
    In this case, as shown by the author in Theorem 4, when $M$ is chosen
    optimally on the order of $M=\mathcal{O}(n^{1/2})$, the estimator is
    consistent at rate $n^{-1/4}$.

    References
    ----------
    Zhang, L. (2006).
    Efficient estimation of stochastic volatility using
    noisy observations: A multi-scale approach,
    Bernoulli 12(6): 1019–1043.

    Zhang, L. (2011).
    Estimating covariation: Epps effect, microstructure noise,
    Journal of Econometrics 160.

    """
    if pairwise:
        indeces, values = _get_indeces_and_values(tick_series_list)
        cov = _msrc_pairwise(indeces, values, M, N)

    else:
        data = refresh_time(tick_series_list)
        data = data.to_numpy().T
        if data.ndim == 1:
            data = data.reshape(1, -1)
        cov = _msrc(data, M, N)

    return cov


@numba.njit(fastmath=False, parallel=False)
def _get_YY_m(Y, N, m):
    Km = N + m
    log_rets = Y[:, Km:] - Y[:, :-Km]

    return log_rets @ log_rets.T / Km


@numba.njit(fastmath=False, parallel=False)
def _msrc(data, M, N):
    r"""
    The inner function of :func:`~msrc`, not pairwise. The multi-scale realized
    volatility (MSRV) estimator of Zhang (2006). It is extended to multiple
    dimensions following Zhang (2011).

    Parameters
    ----------
    data : numpy.ndarray, >0, shape = (p, n)
        previous tick prices with dimensions p by n, where
        p = #assets, n = #number of refresh times, most recent tick on the
        right, must be synchronized (e.g. with :func:`~refresh_time`).
    M : int, >=1
        The number of scales.
        If ``M=None`` all scales :math:`i = 1, ..., M` are used, where M is
        chosen :math:`M = n^{1/2}` acccording to Eqn (34) of Zhang (2006).
    N : int, >=0
        The constant $N$ of Tao et al. (2013)
        If ``N=None`` :math:`N = n^{1/2}`. Lam and Qian (2019) need
        :math:`N = n^{2/3}` for non-sparse integrated covariance matrices,
        in which case the rate of convergence reduces to $n^{1/6}$.

    Returns
    -------
    out : numpy.ndarray
        The mrc estimate of the integrated covariance matrix.

    Examples
    --------

    # >>> np.random.seed(0)
    # >>> n = 200000
    # >>> returns = np.random.multivariate_normal([0, 0], [[1,0.5],[0.5,1]], n)/n**0.5
    # >>> prices = 100*np.exp(returns.cumsum(axis=0))
    # >>> # add Gaussian microstructure noise
    # >>> noise = 10*np.random.normal(0, 1, n*2).reshape(-1, 2)*np.sqrt(1/n**0.5)
    # >>> prices +=noise
    # >>> # sample n/2 (non-synchronous) observations of each tick series
    # >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index()
    # >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index()
    # >>> pt = refresh_time([series_a, series_b])
    # >>> icov = _msrc(pt.values.T, K=np.array([1]))
    # >>> icov_c = _msrc(pt.values.T)
    # >>> # This is the biased uncorrected integrated covariance matrix estimate.
    # >>> icov
    # array([[11.55288112,  0.45281646],
    #        [ 0.45281646,  2.17269871]])
    # >>> # This is the unbiased corrected integrated covariance matrix estimate.
    # >>> icov_c
    # array([[0.89731589, 0.48705002],
    #        [0.48705002, 0.9801241 ]])
    # >>> # In the univariate case we add an axis
    # >>> univariate_ticks = series_a.values[:, None]
    # >>> ivar_c = _msrc(univariate_ticks.T)
    # >>> ivar_c
    # array([[0.90361064]])
    """

    p, n = data.shape

    if M is None:
        # Opt M according to Eqn (34) of Zhang (2006)
        M = int(np.ceil(n**(1/2)))

    if N is None:
        # N according to Fan and Wang (2007)
        N = int(np.ceil(n**(1/2)))
        # N according to Lam and Wang (2019)
        # N = int(np.ceil(n**(2/3)))

    s = np.zeros((p, p))

    if M > 1:
        for m in range(1, M+1):
            # optimal weights according to Eqn (18)
            a = 12*(m + N)*(m - M/2 - 1/2) / (M*(M**2 - 1))
            YY_m = _get_YY_m(data, N, m)
            s += a * YY_m

        zeta = (M + N)*(N + 1)/((n + 1)*(M - 1))
        YY_K1 = _get_YY_m(data, N, 1)
        YY_KM = _get_YY_m(data, N, M)
        s += zeta * (YY_K1 - YY_KM)
    else:
        s += _get_YY_m(data, 0, 1)

    return s


@numba.njit(cache=False, parallel=True)
def _msrc_pairwise(indeces, values, M=None, N=None):
    """
    Accelerated inner function of pairwise :func:`msrc`.

    Parameters
    ----------
    indeces : numpy.ndarray, shape(p, n_max), dtype='uint64'
        The length is equal to the number of assets. Each 'row' contains
        the unix time of ticks of one asset.
    values : numpy.ndarray, shape(p, n_max), dtype='float64'>0
        Each 'row' contains the log-prices of ticks of one asset.
    K : numpy.ndarray
        An array of sclales.

    Returns
    -------
    cov : numpy.ndarray, 2d
        The integrated ovariance matrix using the pairwise synchronized data.

    """
    p = indeces.shape[0]
    cov = np.ones((p, p))

    # don't loop over ranges but get all indeces in advance
    # to improve parallelization.
    idx = _upper_triangular_indeces(p)

    for t in prange(len(idx)):
        i, j = idx[t, :]

        # get the number of no nan values for asset i and j.
        # This is needed since nan is not defined
        # for int64, which are in the indeces. Hence, I use the fact that
        # values and indeces have the same shape and nans are only at the
        # end of an array.
        n_not_nans_i = values[i][~np.isnan(values[i])].shape[0]
        n_not_nans_j = values[i][~np.isnan(values[j])].shape[0]

        if i == j:
            cov[i, i] = _msrc(values[i, :n_not_nans_i].reshape(1, -1), M, N)[0, 0]
        else:
            merged_values, _ = _refresh_time(
                (indeces[i, :n_not_nans_i],
                 indeces[j, :n_not_nans_j]),
                (values[i, :n_not_nans_i],
                 values[j, :n_not_nans_j]))

            # numba doesn't support boolean indexing of 2d array
            merged_values = merged_values.flatten()
            merged_values = merged_values[~np.isnan(merged_values)]
            merged_values = merged_values.reshape(-1, 2)

            cov[i, j] = _msrc(merged_values.T, M, N)[0, 1]
            cov[j, i] = cov[i, j]

    return cov


def tsrc(tick_series_list, J=1, K=None):
    r"""
    The two-scales realized volatility (TSRV) of
    Zhang et al. (2005). It is extentended to handle multiple dimension
    according to Zhang (2011). :func:`~msrc` has better convergence
    rate and is thus prefrerred.

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick-log-prices of one asset
        with datetime index.
    K : int, default = ``int(n**(2/3))``
        long scale, default = ``int(n**(2/3))`` as per Zhang (2005)
    J : int, default = 1
        short scale

    Returns
    -------
    out : numpy.ndarray
        The TSRV estimate.

    Examples
    --------
    >>> np.random.seed(0)
    >>> n = 200000
    >>> returns = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], n)/n**0.5
    >>> prices = 100*np.exp(returns.cumsum(axis=0))
    >>> # add Gaussian microstructure noise
    >>> noise = 10*np.random.normal(0, 1, n*2).reshape(-1, 2)*np.sqrt(1/n**0.5)
    >>> prices += noise
    >>> # sample n/2 (non-synchronous) observations of each tick series
    >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index()
    >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index()
    >>> # take logs
    >>> series_a = np.log(series_a)
    >>> series_b = np.log(series_b)
    >>> icov_c = tsrc([series_a, series_b])
    >>> # This is the unbiased,  corrected integrated covariance matrix estimate.
    >>> np.round(icov_c, 3)
    array([[0.995, 0.361],
           [0.361, 0.977]])

    Notes
    -----
    The two-scales realized volatility (TSRV) estimator is defined as

    \begin{equation}
    \widehat{\langle X^{(j)}, X^{(j)}\rangle}^{(\mathrm{TSRV})}_{T}=
    \left[Y^{(j)}, Y^{(j)}\right]_{T}^{(K)}-\frac{\bar{n}_{K}}{\bar{n}_{J}}
    \left[Y^{(j)}, Y^{(j)}\right]_{T}^{(J)},
    \end{equation}
    where
    \begin{equation}
    \left[Y^{(j)}, Y^{(j)}\right]_{T}^{(K)}=\frac{1}{K}
    \sum_{i=K}^{n}\left(Y_{\tau_{i}^{(j)}}^{(j)}-
    Y_{\tau_{i-K}^{(j)}}^{(j)}\right)^2,
    \end{equation}
    with $K$ being a positive integer usually chosen much larger than 1 and
    $\bar{n}_{K}=\left(n-K+1\right)/K$ and $\bar{n}_{J}=\left(n- J+1\right)/J$.
    If $K$ is chosen on the order of$K=\mathcal{O}\left(n^{2 / 3}\right)$ this
    estimator is asymptotically unbiased, consistent, asymptotically normal
    distributed and converges at rate $n^{-1 / 6}$.

    Zhang (2011) proposes the (multivariate) two scales realized covariance
    (TSCV) estimator based on previous-tick times of asset $k$ and $l$,
    which simultaneously corrects for the bias due to asynchronicity and the
    bias due to microstructure noise. Previous-tick times may be computed via
    :func:`~refresh_time`.

    The TSCV estimator is defined as
    \begin{equation}
    \widehat{\langle X^{(k)},X^{(l)}\rangle}_{T}^{(TSCV)}=c\left(\left[Y^{(k)},
    Y^{(l)}\right]_{T}^{(K)}-\frac{\bar{n}_{K}}{\bar{n}_{J}}\left[Y^{(k)},
    Y^{(l)}\right]_{T}^{(J)}\right),
    \end{equation}
    where
    \begin{equation}
    \left[Y^{(k)}, Y^{(l)}\right]_{T}^{(K)}=\frac{1}{K}
    \sum_{i=K}^{\tilde{n}}\left(Y^{(k)}_{\tau^{(k)}_{i}}-Y^{(k)}_{\tau^{(k)}_{i-K}}
    \right)\left(Y^{(l)}_{\tau^{(l)}_{i}}-Y^{(l)}_{\tau^{(l)}_{i-K}}\right)
    \end{equation}
    $c=1+o_{p}\left(\tilde{n}^{-1 / 6}\right)$ is a small sample correction.
    $K$ is again a positive integer usually chosen much larger than 1 and
    $\bar{n}_{K}=\left(\tilde{n}- K+1\right) / K$ and $\bar{n}_{J}=
    \left(\tilde{n}- J+1\right) / J$.
    The author shows that if $K=\mathcal{O}\left((n^{(k)}+n^{(l)})^{2/3}\right)$
    this estimator is asymptotically unbiased, consistent, asymptotically
    normal distributed and converges at rate $\tilde{n}^{-1 / 6}$.

    .. Note:: Use :func:`~msrc` since it has better converges rate.


    References
    ----------
    Zhang, L., Mykland, P. A. and Ait-Sahalia, Y. (2005).
    A tale of two time scales: Determining integrated
    volatility with noisy high-frequency data, Journal of
    the American Statistical Association 100(472): 1394–1411.

    Zhang, L. (2011). Estimating covariation: Epps effect,
    microstructure noise, Journal of Econometrics 160.

    """
    data = refresh_time(tick_series_list)
    M = data.shape[0]

    if K is None:
        K = int(M ** (2 / 3))

    sk = (data - data.shift(K)).dropna()
    sk = sk.transpose().dot(sk)
    sk = 1 / K * sk

    sj = (data - data.shift(J)).dropna()
    sj = sj.transpose().dot(sj)
    sj = 1 / J * sj

    nj = (M - J + 1) / J
    nk = (M - K + 1) / K
    return (sk - nk/nj * sj).to_numpy()


@numba.njit
def _numba_minimum(x):
    """
    The weighting function of Christensen et al. (2010) used in
    :func:`~mrc`.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    out : numpy.ndarray
        The output of the function applied to each element of x.
    """
    return np.minimum(x, 1-x)


def mrc(tick_series_list, theta=None, g=None, bias_correction=True,
        pairwise=True, k=None):
    r"""
    The modulated realised covariance (MRC) estimator of
    Christensen et al. (2010).

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick-log-prices of one asset
        with datetime index.
    theta : float, optional, default=None
        Theta is used to determine the preaveraging window ``k``.
        If ``bias_correction`` is True (see below)
        then :math:`k = \theta \sqrt{n}`,
        else :math:`k = \theta n^{1/2+ 0.1}`.
        Hautsch & Podolskij (2013) recommend 0.4 for liquid assets
        and 0.6 for less liquid assets. If ``theta=0``, the estimator reduces
        to the standard realized covariance estimator. If ``theta=None`` and
        ``k`` is not specified explicitly, the suggested theta of 0.4 is used.
    g : function, optional, default = ``None``
        A vectorized weighting function.
        If ``g = None``, :math:`g=min(x, 1-x)`
    bias_correction : boolean, optional
        If ``True`` (default) then the estimator is optimized for convergence
        rate but it might not be p.s.d. Alternatively as described in
        Christensen et al. (2010) it can be ommited. Then k should be chosen
        larger than otherwise optimal.
    pairwise : bool, default=True
        If ``True`` the estimator is applied to each pair individually. This
        increases the data efficiency but may result in an estimate that is
        not p.s.d.
    k : int, optional, default=None
        The bandwidth parameter with which to preaverage. Alternative to theta.
        Useful for non-parametric eigenvalue regularization based on sample
        spliting.

    Returns
    -------
    mrc : numpy.ndarray
        The mrc estimate of the integrated covariance.

    Examples
    --------
    >>> np.random.seed(0)
    >>> n = 2000
    >>> returns = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], n)
    >>> returns /=  n**0.5
    >>> prices = 100 * np.exp(returns.cumsum(axis=0))
    >>> # add Gaussian microstructure noise
    >>> noise = 10 * np.random.normal(0, 1, n * 2).reshape(-1, 2)
    >>> noise *= np.sqrt(1 / n ** 0.5)
    >>> prices += noise
    >>> # sample n/2 (non-synchronous) observations of each tick series
    >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index()
    >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index()
    >>> # take logs
    >>> series_a = np.log(series_a)
    >>> series_b = np.log(series_b)
    >>> icov_c = mrc([series_a, series_b], pairwise=False)
    >>> # This is the unbiased, corrected integrated covariance matrix estimate.
    >>> np.round(icov_c, 3)
    array([[0.882, 0.453],
           [0.453, 0.934]])
    >>> # This is the unbiased, corrected realized variance estimate.
    >>> ivar_c = mrc([series_a], pairwise=False)
    >>> np.round(ivar_c, 3)
    array([[0.894]])
    >>> # Use ticks more efficiently by pairwise estimation
    >>> icov_c = mrc([series_a, series_b], pairwise=True)
    >>> np.round(icov_c, 3)
    array([[0.894, 0.453],
           [0.453, 0.916]])

    Notes
    -----
    The MRC estimator is the equivalent to the realized integrated covariance
    estimator using preaveraged returns. It is of thus of the form

    .. math::
        \begin{equation}
        \label{eqn:mrc_raw}
        \left[\mathbf{Y}\right]^{(\text{MRC})}=\frac{n}{n-K+2}
        \frac{1}{\psi_{2} K} \sum_{i=K-1}^{n} \bar{\mathbf{Y}}_{i}
        \bar{\mathbf{Y}}_{i}^{\prime},
        \end{equation}

    where :math:`\frac{n}{n-K+2}` is a finite sample correction, and

    .. math::
        \begin{equation}
        \begin{aligned}
        &\psi_{1}^{k}=k \sum_{i=1}^{k}\left(g\left(\frac{i}{k}\right)-g
        \left(\frac{i-1}{k}\right)\right)^{2}\\
        &\psi_{2}^{k}=\frac{1}{k}
        \sum_{i=1}^{k-1} g^{2}\left(\frac{i}{k}\right).
        \end{aligned}
        \end{equation}

    In this form, however, the estimator is biased. The bias corrected
    estimator is given by

    .. math::

        \begin{equation}
        \label{eqn:mrc}
        \left[\mathbf{Y}\right]^{(\text{MRC})}=\frac{n}{n-K+2}
        \frac{1}{\psi_{2} k} \sum_{i=K-1}^{n} \bar{\mathbf{Y}}_{i}
        \left(\bar{\mathbf{Y}}_{i}-\frac{\psi_{1}}{\theta^{2} \psi_{2}}
        \hat{\mathbf{\Psi}}\right)^{\prime},
        \end{equation}

    where

    .. math::
        \begin{equation}
        \hat{\mathbf{\Psi}}=\frac{1}{2 n} \sum_{i=1}^{n} \Delta_{i}\mathbf{Y}
        \left(\Delta_{i} \mathbf{Y}\right)^{\prime}.
        \end{equation}

    The rate of convergence of this estimator is determined by the
    window-length :math:`K`. Choosing
    :math:`K=\mathcal{O}(\sqrt{n})`, delivers the best rate of convergence
    of :math:`n^{-1/4}`. It is thus suggested to choose
    :math:`K=\theta \sqrt{n}`, where :math:`\theta` can be calibrated from the
    data. Hautsch and Podolskij (2013) suggest values between 0.4 (for liquid
    stocks) and 0.6 (for less liquid stocks).

    .. note::
        The bias correction may result in an estimate that is not positive
        semi-definite.

    If positive semi-definiteness is essential, the bias-correction can be
    omitted. In this case, :math:`K` should be chosen larger
    than otherwise optimal with respect to the convergence rate. Of course,
    the convergence rate is slower then.  The optimal rate of convergence
    without the bias correction is :math:`n^{-1 / 5}`, which is attained
    when :math:`K=\theta n^{1/2+\delta}` with :math:`\delta=0.1`.


    ``theta`` should be chosen between 0.3 and 0.6. It should be chosen
    higher if (i) the sampling frequency declines,
    (ii) the trading intensity of the underlying stock is low,
    (iii) transaction time sampling (TTS) is used as opposed to calendar time
    sampling (CTS). A high ``theta`` value can lead to oversmoothing when
    CTS is used. Generally the higher the sampling frequency the better.
    Since :func:`~mrc` and :func:`~msrc` are based on different approaches
    it might make sense to ensemble them. Monte Carlo results show that the
    variance estimate of the ensemble is better than each component
    individually. For covariance estimation the preaveraged
    :func:`~hayashi_yoshida` estimator has the advantage that even ticks that
    don't contribute to the covariance (due to log-summability) are used for
    smoothing. It thus uses the data more efficiently.

    References
    ----------
    Christensen, K., Kinnebrock, S. and Podolskij, M. (2010). Pre-averaging
    estimators of the ex-post covariance matrix in noisy diffusion models
    with non-synchronous data, Journal of Econometrics 159(1): 116–133.

    Hautsch, N. and Podolskij, M. (2013). Preaveraging-based estimation of
    quadratic variation in the presence of noise and jumps: theory,
    implementation, and empirical evidence,
    Journal of Business & Economic Statistics 31(2): 165–183.
    """

    if g is None:
        g = _numba_minimum

    p = len(tick_series_list)

    if pairwise and p > 1:
        indeces, values = _get_indeces_and_values(tick_series_list)
        cov = _mrc_pairwise(indeces, values, theta, g, bias_correction, k)
    else:
        if p > 1:
            data = refresh_time(tick_series_list).dropna()
            data = np.diff(data.to_numpy(), axis=0)
        else:
            data = tick_series_list[0]
            data = np.diff(data.to_numpy(), axis=0)[:, None]

        cov = _mrc(data, theta, g, bias_correction, k)

    return cov


@numba.njit(cache=False, fastmath=False, parallel=False)
def _mrc(data, theta, g, bias_correction, k):
    r"""
    The modulated realised covariance (MRC) estimator of
    Christensen et al. (2010).

    Parameters
    ----------
    data : numpy.ndarray, shape = (n, p)
        An array of univariate log_returns
        or synchronized multivariate log-returns
        (e.g. with :func:`~refresh_time`).
    theta : float, optional, default=0.4
        Theta is used to determine the preaveraging window ``k``.
        If ``bias_correction`` is True (see below)
        then :math:`k = \theta \sqrt{n}`,
        else :math:`k = \theta n^{1/2+ 0.1}`.
        Hautsch & Podolskij (2013) recommend 0.4 for liquid assets
        and 0.6 for less liquid assets. If ``theta=0``, the estimator reduces
        to the standard realized covariance estimator.
    g : function
        A vectorized weighting function.`
    bias_correction : boolean
        If ``True``, then the estimator is optimized for convergence
        rate but it might not be p.s.d. Alternatively, as described in
        Christensen et al. (2010), it can be ommited. Then k should be chosen
        larger than otherwise optimal.
    k : int
        The bandwidth parameter with which to preaverage. Alternative to theta.
        Useful for non-parametric eigenvalue regularization based on sample
        spliting.

    Returns
    -------
    mrc : numpy.ndarray
        The mrc estimate of the integrated covariance.
    """

    n, p = data.shape

    # get the bandwidth
    if k is not None and theta is not None:
        raise ValueError("Either ``theta`` or ``k`` can be specified,"
                         " but not both! One of them must be ``None``.")
    if k is None:
        if theta is None:
            theta = 0.4
        k = _get_k(n, theta, bias_correction)

    if theta is None:
        if bias_correction:
            theta = k / np.sqrt(n)
        else:
            theta = k / np.power(n, 0.6)

    # If theta is greater than zero comute the preaveraging estimator,
    # otherwise the estimator is just the realized covariance matrix.
    if theta > 0:
        psi2 = np.sum(g(np.arange(1, k)/k)**2)/k
        psi1 = np.sum((g(np.arange(1, k)/k)-g((np.arange(1, k)-1)/k))**2)*k

        weight = g(np.arange(1, k)/k)
        data_pa = _preaverage(data, weight)

        data_pa = data_pa.flatten()
        data_pa = data_pa[~np.isnan(data_pa)]
        data_pa = data_pa.reshape(-1, p)

        # The biass correction term, bc, needs to be initialized as array to
        # have a consistent type for numba.
        bc = np.zeros((p, p))

        if bias_correction:
            bc += psi1 / (theta ** 2 * psi2) * data.T @ data / n / 2

        finite_sample_correction = n / (n - k + 2)
        mrc = finite_sample_correction / (psi2 * k) * data_pa.T @ data_pa - bc
    else:
        mrc = data.T @ data

    return mrc


@numba.njit(cache=False, parallel=True, fastmath=False)
def _mrc_pairwise(indeces, values, theta, g, bias_correction, k):
    r"""
    Accelerated inner function of pairwise :func:`~mrc`.

    Parameters
    ----------
    indeces : numpy.ndarray, shape(p, n_max), dtype='uint64'
        The length is equal to the number of assets. Each 'row' contains
        the unix time of ticks of one asset.
    values : numpy.ndarray, shape(p, n_max), dtype='float64'>0
        Each 'row' contains the log-prices of ticks of one asset.
    theta : float, optional, default=0.4
        theta is used to determine the preaveraging window ``k``.
        If ``bias_correction`` is True (see below)
        then :math:`k = \theta \sqrt{n}`,
        else :math:`k = \theta n^{1/2+ 0.1}`.
        Hautsch & Podolskij (2013) recommend 0.4 for liquid assets
        and 0.6 for less liquid assets. If ``theta=0``, the estimator reduces
        to the standard realized covariance estimator.
    g : function
        A vectorized weighting function.`
    bias_correction : boolean
        If ``True``, then the estimator is optimized for convergence
        rate but it might not be p.s.d. Alternatively as described in
        Christensen et al. (2010) it can be ommited. Then k should be chosen
        larger than otherwise optimal.
    k : int
        The bandwidth parameter with which to preaverage. Alternative to theta.
        Useful for non-parametric eigenvalue regularization based on sample
        spliting.

    Returns
    -------
    cov : numpy.ndarray, 2d
        The integrated covariance matrix using the pairwise synchronized data.


    """
    p = indeces.shape[0]
    cov = np.ones((p, p))

    # don't loop over ranges but get all indeces in advance
    # to improve parallelization.
    idx = _upper_triangular_indeces(p)

    for t in prange(len(idx)):
        i, j = idx[t, :]

        # get the number of no nan values for asset i and j.
        # This is needed since nan is not defined
        # for int64, which are in the indeces. Hence, I use the fact that
        # values and indeces have the same shape and nans are only at the
        # end of an array.
        n_not_nans_i = values[i][~np.isnan(values[i])].shape[0]
        n_not_nans_j = values[i][~np.isnan(values[j])].shape[0]

        if i == j:
            data = values[i, :n_not_nans_i].reshape(-1, 1)
            data = data[1:, :] - data[:-1, :]
            cov[i, i] = _mrc(data, theta, g, bias_correction, k)[0, 0]
        else:
            merged_values, _ = _refresh_time((indeces[i, :n_not_nans_i],
                                             indeces[j, :n_not_nans_j]),
                                             (values[i, :n_not_nans_i],
                                              values[j, :n_not_nans_j]))

            # numba doesn't support boolean indexing of 2d array
            merged_values = merged_values.flatten()
            merged_values = merged_values[~np.isnan(merged_values)]
            data = merged_values.reshape(-1, 2)
            data = data[1:, :] - data[:-1, :]

            cov[i, j] = _mrc(data, theta, g, bias_correction, k)[0, 1]
            cov[j, i] = cov[i, j]

    return cov


@numba.njit
def _get_k(n, theta, bias_correction):
    """ Get the optimal bandwidth for preaveraging depending on the sample
    size and whether or not to correct for the bias.
    """
    if theta > 0:
        if bias_correction:
            k = np.ceil(np.sqrt(n)*theta)
        else:
            delta = 0.1
            k = np.ceil(np.power(n, 0.5+delta)*theta)
    else:
        k = 1

    return int(k)


@numba.njit
def parzen_kernel(x):
    r"""
    The Parzen weighting function used in the kernel realized volatility
    matrix estimator (:func:`~krvm`) of Barndorff-Nielsen et al. (2011).

    Parameters
    ----------
    x : float

    Returns
    -------
    y : float
        The weight.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading,
    Journal of Econometrics 162(2): 149– 169.
    """
    if x < 0:
        raise ValueError("x must be >= 0.")
    elif x <= 1/2:
        y = 1 - 6 * x**2 + 6 * x**3
    elif x <= 1:
        y = 2 * (1 - x)**3
    else:
        y = 0
    return y


@numba.njit
def quadratic_spectral_kernel(x):
    """
    The Quadratic Spectral weighting function used in the kernel realized
    volatility matrix estimator (:func:`~krvm`) of Barndorff-Nielsen et.
    al (2011).

    Parameters
    ----------
    x : float

    Returns
    -------
    y : float
        The weight.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading,
    Journal of Econometrics 162(2): 149– 169.
    """
    if x < 0:
        raise ValueError("x must be >= 0.")
    elif x == 0:
        y = 1
    else:
        y = 3 / (x**2) * (np.sin(x) / x - np.cos(x))
    return y


def get_bandwidth(n, var_ret, var_noise, kernel):
    """
    Compute the optimal bandwidth parameter $H$ for :func:`~krvm` according to
    Barndorff-Nielsen et al. (2011).

    Parameters
    ----------
    n : int >0
        The sample size.
    var_ret : float > 0
        The variance of the efficient return process.
    var_noise :float >=0
        The variance of the noise process.

    Returns
    -------
    H : int
        The bandwidth parameter.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading,
    Journal of Econometrics 162(2): 149– 169.
    """

    if kernel == 'parzen':
        # Parzen kernel c_star according to Table 1 of
        # Barndorff-Nielsen et al. (2011).
        c_star = 3.51

    elif kernel == 'quadratic_spectral':
        # Quadratic Spectral c_star according to Table 1 of
        # Barndorff-Nielsen et al. (2011).
        c_star = 0.46
    else:
        raise ValueError("Specified kernel not implemented.")

    xi_sq = var_noise / var_ret
    H = int(c_star * xi_sq**(2/5) * n**(3/5))
    return H


@numba.njit
def gamma(data, h):
    r"""
    The h-th realized autocovariance.

    Parameters
    ----------
    data : numpy.ndarray, shape = (p, n)
        An array of synchronized and demeaned log_returns.
        (e.g. with :func:`~refresh_time`).
    h : int
        The order of the autocovariance.

    Returns
    -------
    gamma_h : numpy.ndarray, shape = (p, p)
        The h-th realized autocovariance matrix.

    Notes
    -----
    The h-th realized autocovariance is given by
    \begin{equation}
    \boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}\right)=
    \sum_{s=h+2}^{n+1}\left(\mathbf{Y}(s)-\mathbf{Y}(s-1)\right)
    \left(\mathbf{Y}(s-h)-\mathbf{Y}(s-h-1)\right)^{\prime}, \quad h \geq 0
    \end{equation}
    and
    \begin{equation}
    \boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}\right)=
    \boldsymbol{\gamma}^{(-h)}\left(\mathbf{Y}\right)^{\prime}, \quad h < 0,
    \end{equation}
    where $\mathbf{Y}$ denotes the synchronized zero-return log-price.
    """
    if h == 0:
        gamma_h = data @ data.T
    else:
        gamma_h = data[:, abs(h):] @ data[:, :-abs(h)].T

    if h < 0:
        gamma_h = gamma_h.T

    return gamma_h


def krvm(tick_series_list, H, pairwise=True, kernel=quadratic_spectral_kernel):
    r"""
    The kernel realized volatility matrix estimator (KRVM) of Barndorff-Nielsen
    et al. (2011).

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick-log-prices of one asset
        with datetime index.
    H : int, > 0
        The bandwidth parameter for the Parzen kernel.
        Should be on the order of $n^{3/5}$.
    pairwise : bool, default=True
        If ``True`` the estimator is applied to each pair individually. This
        increases the data efficiency but may result in an estimate that is
        not p.s.d even for the p.s.d version of thiss estimator.
    kernel : function, default=quadratic_spectral_kernel
        The kernel weighting function.

    Returns
    -------
    cov : numpy.ndarray
        The intgrated covariance matrix estimate.

    Notes
    -----
    The multivariate realized kernel estimator smoothes the autocovariance
    operator and thereby achieves the optimal convergence rate in the
    multivariate setting with noise and asynchronous observation times.
    Incidentally, this estimator is similar in form to the HAC, widely used in
    the statistics and econometrics literature to deal with heteroscedastic and
    autocorrelated noise. Observations are synchronized with the
    :func:`refresh-time` scheme. In addition, $m$ observation are averaged at
    the beginning and at the end of the trading day to estimate the efficient
    price at these times. The authors call this 'jittering'. In practice the
    effect of jittering is negligible but it is needed for proving consistency.
    (It is ignored in this implementation.)
    The, with parameter $m$, jittered log-price vectors are denoted as
    $\mathbf{Y}^{(m)}(s), s=1, \ldots, n-2 m+1$.
    The kernel estimator is defined by
    \begin{equation}
    \widehat{\mathbf{\Sigma}}^{(KRVM)}=\boldsymbol{\gamma}^{(0)}
    \left(\mathbf{Y}^{(m)}\right)+\sum_{h=1}^{n-2 m} k\left(\frac{h-1}{H}
    \right)\left[\boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}^{(m)}\right)+
    \boldsymbol{\gamma}^{(-h)}\left(\mathbf{Y}^{(m)}\right)\right],
    \end{equation}
    where
    \begin{equation}
    \boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}\right)=
    \sum_{s=h+2}^{n+1}\left(\mathbf{Y}(s)-\mathbf{Y}(s-1)\right)
    \left(\mathbf{Y}(s-h)-\mathbf{Y}(s-h-1)\right)^{\prime}, \quad h \geq 0
    \end{equation}
    and
    \begin{equation}
    \boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}\right)=
    \boldsymbol{\gamma}^{(-h)}\left(\mathbf{Y}\right)^{\prime}, \quad h < 0,
    \end{equation}
    with $\mathbf{Y}$ denoting the synchronized zero-return log-price.
    $\boldsymbol{\gamma}^{(h)}$ is the $h$th realized autocovariance (:func:`gamma`).
    $k(\cdot)$ is the kernel function with its bandwidth parameter $H$. It is
    assumed that
    (i) $k(0)=1$ and $k^{\prime}(0)=0$,
    (ii) $k(\cdot)$ is twice differentiable with continuous
    derivatives, and
    (iii) $\int_{0}^{\infty} k(x)^{2} d x,
    \int_{0}^{\infty} k^{\prime}(x)^{2} d x$ and $\int_{0}^{\infty}
    k^{\prime \prime}(x)^{2} d x$ are finite. A slightly adjusted form of this
    estimator that is positive semidefinite is given by
    \begin{equation}
    \widehat{\mathbf{\Sigma}}^{(KRVM_{psd})}=\boldsymbol{\gamma}^{(0)}
    \left(\mathbf{Y}^{(m)}\right)+\sum_{h=1}^{n-2 m} k\left(\frac{h}{H}\right)
    \left[\boldsymbol{\gamma}^{(h)}\left(\mathbf{Y}^{(m)}\right)+
    \boldsymbol{\gamma}^{(-h)}\left(\mathbf{Y}^{(m)}\right)\right].
    \end{equation}
    This form requires the additional assumption $\int_{-\infty}^{\infty}
    k(x) \exp (i x \lambda) d x \geq 0$ for all $\lambda \in \mathbb{R}$.

    Choosing the right kernel function is important. The authors show, for
    example, that the estimator based on the Bartlett weight function is
    inconsistent. Instead, the Parzen kernel (:func:`parzen_kernel`) is
    suggested as a weight function that yields a consistent estimator and can
    be efficiently implemented. The bandwidth $H$ must be on the order of
    $n^{3 / 5}$. The authors choose the scalar $H$ as the average of optimal
    individual $H^{(j)}$:
    $$\bar{H}=p^{-1} \sum_{j=1}^{p} H^{(j)},$$
    where
    \begin{equation}
    H^{(j)}=c^{*} \xi_{j}^{4 / 5} n^{3 / 5},
    \end{equation}
    with
    \begin{equation}
    c^{*}=\left\{k^{\prime \prime}(0)^{2} / k_{\bullet}^{0,0}\right\}^{1 / 5},
    \end{equation}
    and
    \begin{equation}
    \xi_{j}^{2}={\Sigma}_{\epsilon, j j} / {\Sigma}_{j j}.
    \end{equation}
    $\mathbf{\Sigma}_{\epsilon}$  and $\mathbf{\Sigma}$ denote, as previously
    defined, the integrated covariance matrix of the noise and the efficient
    return process, respectively. Here these quantities are understood over the
    interval under consideration. Hence, $\xi_{j}^{2}$ can be interpreted as
    the ratio of the noise variance and the return variance.
    For the Parzen kernel $c^{*}  = 3.51$, as tabulated by the authors. It is a
    measure of the relative asymptotic efficiency of the kernel.
    ${\Sigma}_{j j}$ may be estimated via a low frequency estimator and
    ${\Sigma}_{\epsilon,j j}$ via a high frequency estimator.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading,
    Journal of Econometrics 162(2): 149– 169."""

    p = len(tick_series_list)

    if pairwise and p > 1:
        indeces, values = _get_indeces_and_values(tick_series_list)
        cov = _krvm_pairwise(indeces, values, H, kernel)
    else:
        if p > 1:
            data = refresh_time(tick_series_list).dropna()
            data = np.diff(data.to_numpy(), axis=0)
        else:
            data = tick_series_list[0]
            data = np.diff(data.to_numpy(), axis=0)[:, None]

        cov = _krvm(data.T, H, kernel)

    return cov


@numba.njit(cache=False, parallel=True, fastmath=False)
def _krvm_pairwise(indeces, values, H, kernel):
    r"""
    Accelerated inner function of pairwise :func:`~krvm`.

    Parameters
    ----------
    indeces : numpy.ndarray, shape(p, n_max), dtype='uint64'
        The length is equal to the number of assets. Each 'row' contains
        the unix time of ticks of one asset.
    values : numpy.ndarray, shape(p, n_max), dtype='float64'>0
        Each 'row' contains the log-prices of ticks of one asset.
    H : int, > 0
        The bandwidth parameter for the Parzen kernel.
        Should be on the order of $n^{3/5}$.
    kernel : function
        The kernel weighting function.

    Returns
    -------
    cov : numpy.ndarray, 2d
        The integrated covariance matrix using the pairwise synchronized data.


    """
    p = indeces.shape[0]
    cov = np.ones((p, p))

    # don't loop over ranges but get all indeces in advance
    # to improve parallelization.
    idx = _upper_triangular_indeces(p)

    for t in prange(len(idx)):
        i, j = idx[t, :]

        # get the number of no nan values for asset i and j.
        # This is needed since nan is not defined
        # for int64, which are in the indeces. Hence, I use the fact that
        # values and indeces have the same shape and nans are only at the
        # end of an array.
        n_not_nans_i = values[i][~np.isnan(values[i])].shape[0]
        n_not_nans_j = values[i][~np.isnan(values[j])].shape[0]

        if i == j:
            data = values[i, :n_not_nans_i].reshape(-1, 1)
            data = data[1:, :] - data[:-1, :]
            cov[i, i] = _krvm(data.T, H, kernel)[0, 0]
        else:
            merged_values, _ = _refresh_time((indeces[i, :n_not_nans_i],
                                             indeces[j, :n_not_nans_j]),
                                             (values[i, :n_not_nans_i],
                                              values[j, :n_not_nans_j]))

            # numba doesn't support boolean indexing of 2d array
            merged_values = merged_values.flatten()
            merged_values = merged_values[~np.isnan(merged_values)]
            data = merged_values.reshape(-1, 2)
            data = data[1:, :] - data[:-1, :]

            cov[i, j] = _krvm(data.T, H, kernel)[0, 1]
            cov[j, i] = cov[i, j]

    return cov


@numba.njit(cache=False, parallel=False, fastmath=False)
def _krvm(data, H, kernel):
    """
    Parameters
    ----------
    data : numpy.ndarray, shape = (p, n)
        An array of (jittered), synchronized and log_returns.
        (e.g. with :func:`~refresh_time`).
    H : int, > 0
        The bandwidth parameter for the Parzen kernel.
        Should be on the order of $n^{3/5}$.
    kernel : function
        The kernel weighting function.

    Returns
    -------
    cov : numpy.ndarray, 2d
        The integrated covariance matrix estimate.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2011).
    Multivariate realised kernels: consistent positive semi-definite estimators
    of the covariation of equity prices with noise and non-synchronous trading,
    Journal of Econometrics 162(2): 149– 169."""

    p, n = data.shape

    # if p.s.d estimator: c=0, else: c=1, since pairwise estimation and
    # subsequent shrinkage is advocated anyway, hard-code to 1.
    c = 1
    cov = gamma(data, 0)
    for h in range(1, n+1):
        weight = kernel((h-c) / H)

        # The Parzen kernel, for example, needs to compute only
        # H gammas after that the weight stays 0, hence early stop.
        if weight == 0:
            return cov

        g = gamma(data, h)
        cov += weight * (g + g.T)

    return cov


def hayashi_yoshida(tick_series_list, theta=None, k=None):
    r"""
    The (pairwise) Hayashi-Yoshida estimator of Hayashi and Yoshida (2005).
    This estimtor sums up all products of time-overlapping returns
    between two assets. This makes it possible to compute unbiased
    estimates of the integrated covariance between two assets that
    are sampled non-synchronously. The standard realized covariance
    estimator is biased toward zero in this case. This is known as
    the Epps effect. The function is accelerated via JIT compilation with
    Numba.
    The preaveraged version handles microstructure noise as shown in
    Christensen et al. (2010).

    Parameters
    ----------
    tick_series_list : list of pd.Series
        Each pd.Series contains tick-log-prices of one asset
        with datetime index.
    theta : float, theta>=0, default=None
        If ``theta=None`` and ``k`` is not specified explicitly,
        theta will be set to 0.
        If theta>0, the log-returns are preaveraged with theta and
        :math:`g(x) = min(x, 1-x)`. Hautsch and Podolskij (2013) suggest
        values between 0.4 (for liquid stocks) and 0.6 (for less
        liquid stocks).
        If ``theta=0``, this is the standard HY estimator.
    k : int, >=1, default=None
        The bandwidth parameter with which to preaverage. Alternative to
        ``theta``. Useful for non-parametric eigenvalue regularization based
        on sample splitting. When ``k=None`` and ``theta=None``, ``k`` will
        be set to 1. If ``k=1``, this is the standard HY estimator.

    Returns
    -------
    cov : numpy.ndarray
        The pairwise HY estimate of the integrated covariance matrix.

    Notes
    -----
    The estimator is defined as

    .. math::
        \begin{equation}
        \left\langle X^{(k)}, X^{(l)}\right\rangle_{H Y}=
        \sum_{i=1}^{n^{(k)}}\sum_{i'=1}^{n^{(l)}}
        \Delta X_{t^{(k)}_i}^{(k)}
        \Delta X_{t^{(l)}_{i^{\prime}}}^{(l)}
        \mathbf{1}_{\left\{\left(t_{i-1}^{(k)},
        t_{i}^{(k)}\right] \cap\left(t_{i^{\prime}-1}^{(l)},
        t_{i^{\prime}}^{(l)}\right]\neq \emptyset \right\}},
        \end{equation}

    where

    .. math::
        \Delta X_{t^{(j)}_i}^{(j)} :=X_{t^{(j)}_i}^{(j)} - X_{t^{(j)}_{i-1}}^{(j)}

    denotes the jth asset tick-to-tick log-return over the interval spanned from

    .. math::
        {t^{(j)}_{i-1}} \text{ to } {t^{(j)}_i}, i = 1, \cdots,  n^{(j)}.

    and :math:`n^{(j)} = |t^{(j)}| -1` denotes the number of tick-to-tick
    returns. The following diagram visualizes the products of returns that are
    part of the sum by the dashed lines.

    .. tikz::

        \draw (0,1.75) -- (11,1.75)
        (0,-0.75) -- (11,-0.75)
        (0,1.5) -- (0,2)
        (1.9,1.5) -- (1.9,2)
        (4,1.5) -- (4,2)
        (5,1.5) -- (5,2)
        (7.3,1.5) -- (7.3,2)
        (10.8,1.5) -- (10.8,2)
        (0,-0.5) -- (0,-1)
        (1.9,-0.5) -- (1.9,-1)
        (5.7,-0.5) -- (5.7,-1)
        (8,-0.5) -- (8,-1)
        (10.3,-0.5) -- (10.3,-1);
        \draw[dashed,gray]
        (1.1,1.75) -- (1.1,-0.75)
        (3,1.75) -- (3.8,-0.75)
        (4.5,1.75) -- (3.8,-0.75)
        (6.15,1.75) -- (3.8,-0.75)
        (6.15,1.75) -- (6.8,-0.75) ;

        \draw[dashed] (11,1.75) -- (12,1.75)
              (11,-0.75) -- (12,-0.75);
        \draw[very thick] (9.5,-1.4) -- (9.5,0.25)
              (9.5,0.8) -- (9.5,2.4);
        \draw   (0,0.5) node{$t_{0}^{(k)}=t_{0}^{(l)}=0$}
                (1.9,1) node{$t_{1}^{(k)}$}
                (4,1) node{$t_{2}^{(k)}$}
                (5,1) node{$t_{3}^{(k)}$}
                (7.3,1) node{$t_{4}^{(k)}$}
                (11,1) node{$t_{5}^{(k)}$}
                (9.5,0.5) node{\textbf{$T$}}
                (1.9,0) node{$t_{1}^{(l)}$}
                (5.7,0) node{$t_{2}^{(l)}$}
                (8,0) node{$t_{3}^{(l)}$}
                (10.3,0) node{$t_{4}^{(l)}$};
        \draw   (0,1.75) node[left,xshift=-0pt]{$X^{(k)}$}
        (0,-0.75) node[left,xshift=-0pt]{$X^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (0,2)--(1.9,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{t^{(k)}_1}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (1.9,2)--(4,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{t^{(k)}_2}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (4,2)--(5,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{t^{(k)}_3}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (5,2)--(7.3,2) node[midway, above,yshift=10pt,]
        {$ \Delta X_{t^{(k)}_4}^{(k)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (8,-1)--(5.7,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{t^{(l)}_3}^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (5.7,-1)--(1.9,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{t^{(l)}_2}^{(l)}$};
        \draw[decorate,decoration={brace,amplitude=12pt}]
        (1.9,-1)--(0,-1) node[midway, below,yshift=-10pt,]
        {$ \Delta X_{t^{(l)}_1}^{(l)}$};

    When returns are preaveraged with :func:`~preaverage`, the HY
    estimator of can be made robust to microstructure noise as well.
    It is then of the slightly adjusted form

    .. math::

        \begin{equation}
        \left\langle X^{(k)}, X^{(l)}\right
        \rangle_{H Y}^{\theta}=\frac{1}{
        \left(\psi_{H Y} K \right)^{2}}
        \sum_{i=K}^{n^{(k)}}
        \sum_{i'=K}^{n^{(l)}}
        \bar{Y}_{t^{(k)}_i}^{(k)}\bar{Y}_{t^{(l)}_{i'}}^{(l)}
        \mathbf{1}_{\left\{\left(t_{i-K}^{(k)},
        t_{i}^{(k)}\right] \cap\left(t_{i'-K}^{(l)},
        t_{i'}^{(l)}\right] \neq \emptyset\right)}
        \end{equation}

    where
    :math:`\psi_{HY}=\frac{1}{K} \sum_{i=1}^{K-1} g\left(\frac{i}{K}\right)`
    The preaveraged HY estimator has optimal convergence rate
    :math:`n^{-1/4}`, where :math:`n=\sum_{j=1}^{p} n^{(j)}`.
    Christensen et al. (2013) subsequently proof a central limit theorem for
    this estimator and show that it is robust to some dependence structure of
    the noise process. Since preaveraging is performed before synchronization,
    the estimator utilizes more data than other methods that cancel noise after
    synchronization. In particular, the preaveraged HY estimator even uses the
    observation :math:`t^{(j)}_2` in the figure, which does not contribute
    the the covariance due to the log-summability.

    References
    ----------
    Hayashi, T. and Yoshida, N. (2005).
    On covariance estimation of
    non-synchronously observed diffusion processes,
    Bernoulli 11(2): 359–379.

    Christensen, K., Kinnebrock, S. and Podolskij, M. (2010).
    Pre-averaging
    estimators of the ex-post covariance matrix in noisy diffusion models
    with non-synchronous data, Journal of Econometrics 159(1): 116–133.

    Hautsch, N. and Podolskij, M. (2013).
    Preaveraging-based estimation of
    quadratic variation in the presence of noise and jumps: theory,
    implementation, and empirical evidence,
    Journal of Business & Economic Statistics 31(2): 165–183.

    Christensen, K., Podolskij, M. and Vetter, M. (2013).
    On covariation estimation for multivariate continuous itˆo
    semimartingales with noise in non-synchronous observation schemes,
    Journal of Multivariate Analysis 120: 59–84.


    Examples
    --------
    >>> np.random.seed(0)
    >>> n = 10000
    >>> returns = np.random.multivariate_normal([0, 0], [[1,0.5],[0.5,1]], n)/n**0.5
    >>> prices = np.exp(returns.cumsum(axis=0))
    >>> # sample n/2 (non-synchronous) observations of each tick series
    >>> series_a = pd.Series(prices[:, 0]).sample(int(n/2)).sort_index()
    >>> series_b = pd.Series(prices[:, 1]).sample(int(n/2)).sort_index()
    >>> # take logs
    >>> series_a = np.log(series_a)
    >>> series_b = np.log(series_b)
    >>> icov = hayashi_yoshida([series_a, series_b])
    >>> np.round(icov, 3)
    array([[0.983, 0.512],
           [0.512, 0.99 ]])
    """
    indeces, values = _get_indeces_and_values(tick_series_list)
    p = indeces.shape[0]

    # get log-returns
    values = np.diff(values, axis=1)
    # do not drop first nan which results from diff since its index
    # is used to determine first interval. Instead put column of zeros.
    values = np.column_stack((np.zeros(p), values))

    cov = _hayashi_yoshida_pairwise(indeces, values, theta, k)
    return cov


@numba.njit(cache=False, parallel=True, fastmath=False)
def _hayashi_yoshida_pairwise(indeces, values, theta, k):
    r"""
    The pairwise computation of the integrated covariance matrix
    in :func:`~hayashi_yoshida` using :func:`~_hayashi_yoshida` is
    accelerated and parallelized.

    Parameters
    ----------
    indeces : numpy.ndarray, shape(p, n_max), dtype='uint64'
        The length is equal to the number of assets. Each 'row' contains
        the unix time of ticks of one asset.
    values : numpy.ndarray, shape(p, n_max), dtype='float64'>0
        Each 'row' contains the log-tick-returns of one asset.
    theta : float, theta>=0
        If ``theta=None`` and ``k`` is not specified explicitly,
        theta will be set to 0.
        If theta>0, the log-returns are preaveraged with theta and
        :math:`g(x) = min(x, 1-x)`. Hautsch and Podolskij (2013) suggest
        values between 0.4 (for liquid stocks) and 0.6 (for less
        liquid stocks).
        If ``theta=0``, this is the standard HY estimator.
    k : int, >=1
        The bandwidth parameter with which to preaverage. Alternative to
        ``theta``. Useful for non-parametric eigenvalue regularization based
        on sample splitting.

    Returns
    -------
    cov : numpy.ndarray
        The pairwise HY estimate of the integrated covariance matrix.
    """

    p = indeces.shape[0]
    cov = np.zeros((p, p))

    # don't loop over ranges but get all indeces in advance
    # to improve parallelization.
    idx = _upper_triangular_indeces(p)
    for t in prange(len(idx)):
        i, j = idx[t, :]

        # get the number of no nan values for asset i and j.
        # This is needed since nan is not defined
        # for int64, which are in the indeces. Hence, I use the fact that
        # values and indeces have the same shape and nans are only at the
        # end of an array.
        n_not_nans_i = values[i][~np.isnan(values[i])].shape[0]
        n_not_nans_j = values[i][~np.isnan(values[j])].shape[0]

        # for efficiency set slower trading asset to ``a``.
        if n_not_nans_i <= n_not_nans_j:
            a_values = values[i, :n_not_nans_i]
            a_index = indeces[i, :n_not_nans_i]
            b_values = values[j, :n_not_nans_j]
            b_index = indeces[j, :n_not_nans_j]

        else:
            b_values = values[i, :n_not_nans_i]
            b_index = indeces[i, :n_not_nans_i]
            a_values = values[j, :n_not_nans_j]
            a_index = indeces[j, :n_not_nans_j]

        hy = _hayashi_yoshida(a_index, b_index,
                              a_values, b_values,
                              k, theta)
        cov[i, j] = hy
        cov[j, i] = hy
    return cov


@numba.njit(cache=False, parallel=False, fastmath=False)
def _hayashi_yoshida(a_index, b_index, a_values, b_values, k=None, theta=None):
    """
    The inner function of :func:`~hayashi_yoshida` is accelerated
    via JIT compilation with Numba.

    Parameters
    ----------
    a_index : numpy.ndarray, 1d, dtype='uint64'
        A numpy.ndarray containing indeces of trade times. Must be
        uint64 since Numba cannot check nan otherwise. Preferably
        a should be the slower trading asset.

    b_index : numpy.ndarray, 1d, dtype='uint64'
        A numpy.ndarray containing indeces of trade times. Must be
        uint64 since Numba cannot check nan otherwise.

    a_values : numpy.ndarray, 1d
        A numpy.ndarray containing log-returns at times given by `a_index`.
        The index is determined by the last price, i.e.,
        r_t = log(p_t) - log(p_{t-1})

    b_values : numpy.ndarray, 1d
        A numpy.ndarray containing log-returns. Similar to a_values.
    k : int, default=None
        k is 1 for the standard HY estimator. When preaveraging
        is used to cancel microstructure noise, the step size has to be
        adjusted according to Eqn (27) of Christensen et al. (2010).

    Returns
    -------
    hy : float
        The HY estimate of the covariance of returns of asset a and asset b.

    """
    assert len(a_index) == len(a_values) and len(b_index) == len(b_values), \
        'indeces and values must have same length.'

    if k is not None and theta is not None:
        raise ValueError("Either ``theta`` or ``k`` can be specified,"
                         " but not both! One of them must be ``None``.")

    if theta is None:
        if k is None:
            # if no preaveraging
            k = 1
    else:
        # If ``theta`` is specified set k as recommended in
        # Christensen et al. (2010)
        k = _get_k((a_values.shape[0] + b_values.shape[0])/2, theta, True)

    if k > 1:
        # Preaverage
        weight = _numba_minimum(np.arange(1, k)/k)
        a_values = _preaverage(a_values.reshape(-1, 1), weight).flatten()
        # and adjust acc. to Eqn (27) of Christensen et al. (2010).
        # psi_HY = np.sum(g(np.arange(1, k)/k))/k = 1/4 for weight
        # function chosen as def g(x): return np.minimum(x, 1-x)
        a_values = a_values[k-1:] / (k / 4)
        b_values = _preaverage(b_values.reshape(-1, 1), weight).flatten()
        b_values = b_values[k-1:] / (k / 4)
        a_index = a_index[k-1:]
        b_index = b_index[k-1:]

    temp = np.zeros(a_index.shape[0], dtype=np.float64)
    for i in prange(k, a_index.shape[0]):
        start = a_index[i-k]
        end = a_index[i]
        start_b = np.searchsorted(b_index, start, 'right')
        # TODO limit search space e.g. end only after start. Currently
        # insignificant speedup. E.g.:
        # end_b = np.searchsorted(b_index[start_b:], end, 'left') + start_b
        end_b = np.searchsorted(b_index, end, 'left')
        # Don't do:
        # hy += np.sum(a_values[i] * b_values[start_b: end_b+k])
        # since there are problems in parallelization.
        temp[i] = np.sum(a_values[i] * b_values[start_b: end_b+k])
    hy = np.sum(temp)
    return hy


def ensemble(estimates,  var_weights, cov_weights):
    """
    Ensemble multiple covariance matrix estimates with weights given
    by ``var_weights`` and ``cov_weights`` for the diagonal and
    off-diagonal elements, respectively. This function is used in the
    ensembled pairwise integrated covariance (EPIC) estimator of Woeltjen
    (2020). The :func:`msrc` estimator , the :func:`mrc` estimator, the
    :func:`krvm` estimator and the preaveraged :func:`hayashi_yoshida`
    estimator are ensembled to compute an improved finite sample estimate
    of the pairwise integrated covariance matrix. The EPIC estimator uses every
    available tick, and compares favorable in finite samples to its
    constituents on their own. The preaveraged HY estimates of the off-diagonals
    have better finite sample properties than the other estimators so it might
    be preferable to overweight them by setting the corresponding
    ``cov_weights`` element to a number >1/4.

    Parameters
    ----------
    estimates : list of numpy.ndarrays with shape = (p, p)
        The covariance matrix estimates.
        var_weights :  numpy.ndarray
        The weights with which the  diagonal elements of the MSRC, MRC, and
        the preaveraged HY covariance estimates are weighted, respectively.
        The weights must sum to one.
    cov_weights : numpy.ndarray
        The weights with which the off-diagonal elements of the MSRC, MRC, and
        the preaveraged HY covariance estimates are weighted, respectively. The
        HY estimator uses the data more efficiently and thus may deserve a
        higher weight. The weights must sum to one.

    Returns
    -------
    cov : numpy.ndarray
        The ensemble estimate of the integrated covariance matrix.
    """

    p, p_prime = estimates[0].shape

    assert p == p_prime, "The covariance matrices must be square."

    cov = np.zeros((p, p))
    V = np.eye(p)
    C = np.ones((p, p)) - V

    for i, estimate in enumerate(estimates):
        assert estimate.shape == (p, p), \
         "All estimates must have same dimension."
        cov += (var_weights[i] * V + cov_weights[i] * C) * estimate
    return cov
