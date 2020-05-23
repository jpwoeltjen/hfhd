import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from hfhd import loss
import numpy as np
import pandas as pd
from numpy import nan


def test_loss_fr_equal():
    sigma = np.eye(2)
    sigma_hat = np.eye(2)
    loss_fr = loss.loss_fr(sigma_hat, sigma)
    assert loss_fr == 0


def test_loss_fr_scaled():
    sigma = np.eye(2)
    sigma_hat = 2*np.eye(2)
    loss_fr = loss.loss_fr(sigma_hat, sigma)
    assert loss_fr == 1


def test_loss_mv_equal():
    sigma = np.eye(2)
    sigma_hat = np.eye(2)
    loss_mv = loss.loss_mv(sigma_hat, sigma)
    assert loss_mv == 0


def test_loss_mv_scaled():
    sigma = np.eye(2)
    sigma_hat = 2*np.eye(2)
    loss_mv = loss.loss_mv(sigma_hat, sigma)
    # print(loss_mv)
    assert loss_mv == 0


def test_loss_mv_add1():
    sigma = np.eye(2)
    sigma_hat = np.eye(2) + 1
    loss_mv = loss.loss_mv(sigma_hat, sigma)
    assert np.allclose(loss_mv, 0.25)


def test_prial_0():
    sigma = np.eye(2)
    S = np.eye(2) + 1
    sigma_hat = S
    prial = loss.prial([S], [sigma_hat], sigma)
    # print(prial)
    assert prial == 0


def test_prial_100():
    sigma = np.eye(2)
    S = np.eye(2) + 1
    sigma_hat = np.eye(2)
    prial = loss.prial([S], [sigma_hat], sigma)
    # print(prial)
    assert prial == 100


def test_prial_75():
    sigma = np.eye(2)
    S = np.eye(2) + 1
    sigma_hat = np.eye(2) + 1/3
    prial = loss.prial([S], [sigma_hat], sigma)
    # print(prial)
    assert np.allclose(prial, 75)
