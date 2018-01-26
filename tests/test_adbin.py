'''
Test functions for the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import matplotlib.pyplot as plt

# Custom
import hep_spt

# Set the random seed for reproducibility
np.random.seed(8563)


def test_adbin_hist1d():
    '''
    Test the creation of an adaptive binned histogram in 1 dimension.
    '''
    size = 1000
    bins = 20
    rg   = (-10, 10)

    sample  = np.random.normal(0., 2, size)
    weights = np.random.uniform(0, 1, size)

    # Case without weights
    v, ev, ex, ey = hep_spt.adbin_hist1d(sample, bins, rg)

    # Case with weights
    v_w, ev_w, ex_w, ey_w = hep_spt.adbin_hist1d(sample, bins, rg, wgts = weights)


def test_adbin_hist2d():
    '''
    Test the creation of an adaptive binned histogram in 2 dimensions.
    '''
    size = 100000

    smp_x   = np.random.normal(0., 2, size)
    smp_y   = np.random.normal(0., 2, size)
    weights = np.random.uniform(0, 1, size)

    # Number of bins is a power of "ndiv". The requested and actual
    # number of bins are identical.
    nbins = 16
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, ndiv = 2)
    assert nbins == len(bins)

    exp = float(size)/nbins

    assert _within_expectations(bins, exp - 0.5, exp + 0.5)

    # Number of bins is not a power of "ndiv". Requested and actual
    # number of bins are different.
    nbins = 11
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, ndiv = 3)
    assert nbins != len(bins)

    exp = float(size)/27

    assert _within_expectations(bins, exp - 2., exp + 2.)

    # Weighted case
    nbins = 8
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, wgts = weights)

    exp = float(weights.sum())/nbins

    assert _within_expectations(bins, exp - 2., exp + 2.)


def _within_expectations( bins, vmin, vmax ):
    '''
    Check whether the values in the input array are inside the expectations.
    '''
    arr = np.array([b.sw(b.arr, b.wgts) for b in bins])

    return np.logical_and(np.all(arr > vmin), np.all(arr < vmax))
