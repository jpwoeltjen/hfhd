"""
This module provides loss functions frequently encountered in the literature
on high dimensional covariance matrix estimation.
"""

import numpy as np
from hfhd import hd


def prial(S_list, sigma_hat_list, sigma, loss_func=None):
    r"""
    The percentage relative improvement in average loss (PRIAL)
    over the sample covariance matrix.

    Parameters
    ----------
    S_list : list of numpy.ndarray
        The sample covariance matrix.
    sigma_hat_list : list of numpy.ndarray
        The covariance matrix estimate using the estimator of interest.
    sigma : numpy.ndarray
        The (true) population covariance matrix.
    loss_func : function, defualt = None
        The loss function. If ``None`` the minimum variance loss function is
        used.

    Returns
    -------
    prial : float
        The PRIAL.

    Notes
    -----
    The percentage relative improvement in average loss (PRIL)
    over the sample covariance matrix is given by:

    .. math::
        \mathrm{PRIAL}_{n}\left(\widehat{\Sigma}_{n}\right):=
        \frac{\mathbb{E}\left[\mathcal{L}_{n}\left(S_{n},
        \Sigma_{n}\right)\right]-\mathbb{E}\left[\mathcal{L}_{n}
        \left(\widehat{\Sigma}_{n}, \Sigma_{n}\right)\right]}
        {\mathbb{E}\left[\mathcal{L}_{n}\left(S_{n},
        \Sigma_{n}\right)\right]-\mathbb{E}\left[\mathcal{L}_{n}
        \left(S_{n}^{*}, \Sigma_{n}\right)\right]} \times 100 \%

    """
    if loss_func is None:
        loss_func = loss_mv

    mean_loss_S = np.mean([loss_func(S, sigma) for S in S_list], axis=0)

    mean_loss_sigma_hat = np.mean([loss_func(sigma_hat, sigma)
                                   for sigma_hat in sigma_hat_list], axis=0)

    mean_loss_fsopt = np.mean([loss_func(hd.fsopt(S, sigma), sigma)
                               for S in S_list], axis=0)

    denom = mean_loss_S - mean_loss_fsopt
    num = mean_loss_S - mean_loss_sigma_hat

    if denom != 0:
        prial = num / denom * 100
    else:
        raise ValueError("""PRIAL not defined: The sample covariance attained
            the smallest possible loss.""")
    return prial


def loss_mv(sigma_hat, sigma):
    r"""
    The minimum variance loss function of Ledoit and Wolf (2018).

    Parameters
    ----------
    sigma_hat : numpy.ndarray
        The covariance matrix estimate using the estimator of interest.
    sigma : numpy.ndarray
        The (true) population covariance matrix.

    Returns
    -------
    out : float
        The minimum variance loss.

    Notes
    -----
    The minimum variance (MV)-loss function is proposed by
    Engle et al. (2019) as a loss function that is appropriate for covariance
    matrix estimator evaluation for the problem of minimum variance portfolio
    allocations under a linear constraint and large-dimensional asymptotic
    theory.

    The loss function is given by:

    .. math::
        \mathcal{L}_{n}^{\mathrm{MV}}\left(\widehat{\Sigma}_{n},
        \Sigma_{n}\right):=\frac{\operatorname{Tr}\left(\widehat{\Sigma}_{n}^{-1}
        \Sigma_{n} \widehat{\Sigma}_{n}^{-1}\right) / p}
        {\left[\operatorname{Tr}\left(\widehat{\Sigma}_{n}^{-1}\right)
        /p\right]^{2}}-\frac{1}{\operatorname{Tr}\left(\Sigma_{n}^{-1}\right)/p}.

    It can be interpreted as the true variance of the minimum variance
    portfolio constructed from the estimated covariance matrix.
    """
    p = sigma.shape[0]

    sigma_hat_inv = np.linalg.inv(sigma_hat)
    sigma_inv = np.linalg.inv(sigma)

    num = np.trace(sigma_hat_inv @ sigma @ sigma_hat_inv) / p
    denom = (np.trace(sigma_hat_inv) / p) ** 2
    return num / denom - (np.trace(sigma_inv) / p)


def loss_fr(sigma_hat, sigma):
    r"""Squared Frobenius norm scaled by 1/p.
    Same as ``np.linalg.norm(sigma_hat - sigma, 'fro')**2 *1/p``.

    Parameters
    ----------
    sigma_hat : numpy.ndarray
        The covariance matrix estimate using the estimator of interest.
    sigma : numpy.ndarray
        The (true) population covariance matrix.

    Returns
    -------
    out : float
        The minimum variance loss.

    Notes
    -----
    The loss function is given by:

    .. math::
        \mathcal{L}_{n}^{\mathrm{FR}}\left(\widehat{\Sigma}_{n},
        \Sigma_{n}\right):=\frac{1}{p}
        \operatorname{Tr}\left[\left(\widehat{\Sigma}_{n}
        -\Sigma_{n}\right)^{2}\right]

    """
    p = sigma.shape[0]
    delta = sigma_hat - sigma
    return np.trace(delta @ delta) / p


def marchenko_pastur(x, c, sigma_sq):
    r"""
    The Marchenko-Pastur distribution. This is the pdf
    of eigenvalues of a sample covariance matrix estimate of
    the true covariance matrix, which is a``sigma_sq`` scaled identity matrix.
    It depends on the concentration ratio ``c``, which is the ratio of
    the dimension divided by the number of observations.

    Parameters
    ----------
    x : float
        The value of the sample eigenvalue.
    c : float
        The concentration ratio. $c=p/n$.
    sigma_sq : float
        The value of population eigenvalues.

    Returns
    -------
    p : float
        The value of the Marchenko-Pastur distribution at the sample
        eigenvalue ``x``.

    Notes
    -----
    The Marchenko-Pastur law states that the limiting spectrum of the sample
    covariance matrix $S =   {X 'X}/n$ of independent and identically
    distributed $p$-dimensional random vectors
    $\mathbf{X}=\left(x_{1}, \ldots, x_{n}\right)$
    with mean $\mathbf{0}$ and covariance matrix
    $\mathbf{\Sigma}=\sigma^{2} \mathbf{I}_{p}$, has density
    \begin{equation}
    f_{c}(x)=\left\{\begin{array}{ll}
    \frac{1}{2 \pi x c \sigma^{2}} \sqrt{(b-x)(x-a)}, & a \leq x \leq b \\
    0, & \text { otherwise, }
    \end{array}\right.
    \end{equation}
    where the smallest and the largest eigenvalues are given by
    $a=\sigma^{2}(1-\sqrt{c})^{2}$ and $b=\sigma^{2}(1+\sqrt{c})^{2}$,
    respectively, as $p, n \rightarrow \infty$ with $p / n \rightarrow c>0$.

    References
    ----------
    Marchenko, V. A. and Pastur, L. A. (1967).
    Distribution of eigenvalues for some sets of random matrices,
    Matematicheskii Sbornik 114(4): 507–536.

    """
    a = sigma_sq*(1-np.sqrt(c))**2
    b = sigma_sq*(1+np.sqrt(c))**2

    if a <= x <= b:
        p = 1/(2*np.pi*x*c*sigma_sq)*np.sqrt((b-x)*(x-a))
    else:
        p = 0
    return p