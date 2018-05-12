'''
Test functions for the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Local
import hep_spt

# Set the random seed for reproducibility
np.random.seed(8563)


def test_adbin():
    '''
    Test the behaviour of the AdBin class.
    '''
    size = 2000

    smp_x = np.random.normal(0., 2, size)
    smp_y = np.random.normal(0., 2, size)
    smp   = np.array([smp_x, smp_y]).T

    b = hep_spt.AdBin(smp)

    bins = b.divide()

    # Default division is in two bins
    assert len(bins) == 2

    # Both bins must have the same amount of entries
    bl, br = bins

    assert bl.sw(smp) == br.sw(smp)

    # Test raising of RuntimeError in "divide"
    with pytest.raises(RuntimeError):
        bl.free_memory()
        bl.divide()


def test_adbin_hist1d():
    '''
    Test the creation of an adaptive binned histogram in 1 dimension.
    '''
    # Case without weights
    arr = np.arange(10)
    _, e, _, _ = hep_spt.adbin_hist1d(arr + 0.5, 10, range=(0, 10))
    ref = np.append(arr, 10)
    assert np.allclose(e, ref)

    # Weighted case - 1
    smp = np.array([0., 1., 0.])
    wgts = np.array([1., 2., 1.])
    v, e, _, _ = hep_spt.adbin_hist1d(smp, nbins=2, weights=wgts)

    assert np.allclose(e, [0., 0.5, 1.])
    assert np.allclose(v, [2., 2.])

    # Weighted case - 2
    arr  = np.arange(10)
    wgts = 0.5*np.ones(10)
    _, e, _, _ = hep_spt.adbin_hist1d(arr + 0.5, 10, range=(0, 10), weights=wgts)
    ref = np.append(arr, 10)
    assert np.allclose(e, ref)


def test_adbin_hist1d_edges():
    '''
    Test for the "adbin_hist1d_edges" function.
    '''
    n   = 10
    smp = np.arange(n) + 0.5
    weights = 0.5*np.ones(n)

    # Non-weighted case
    edges = hep_spt.adbin_hist1d_edges(smp, nbins=len(smp), range=(0, n))

    ref = np.append(smp - 0.5, len(smp))

    assert np.allclose(edges, ref)

    # Weighted case
    edges = hep_spt.adbin_hist1d_edges(smp, nbins=len(smp), range=(0, n), weights=weights)

    ref = np.append(smp - 0.5, len(smp))

    assert np.allclose(edges, ref)

    # Test exceptions
    with pytest.raises(ValueError):
        hep_spt.adbin_hist1d_edges(smp, len(smp) + 1)


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
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, ndiv=2, free_memory=False)
    assert nbins == len(bins)

    exp = float(size)/nbins

    assert _within_expectations(bins, exp - 0.5, exp + 0.5)

    # Number of bins is not a power of "ndiv". Requested and actual
    # number of bins are different.
    nbins = 11
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, ndiv=3, free_memory=False)
    assert nbins != len(bins)

    exp = float(size)/27

    assert _within_expectations(bins, exp - 2., exp + 2.)

    # Weighted case
    nbins = 8
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins, weights=weights, free_memory=False)

    exp = float(weights.sum())/nbins

    assert _within_expectations(bins, exp - 2., exp + 2.)


def test_adbin_hist2d_rectangles():
    '''
    Test for the "adbin_hist2d_rectangles" function.
    '''
    smp_x   = np.array([ 0., 0.,  1., 1.])
    smp_y   = np.array([ 0., 1.,  0., 1.])
    weights = np.array([ 2,  1,   2,  1])

    # Non-weighted case
    nbins = 2
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins)

    recs, conts = hep_spt.adbin_hist2d_rectangles(bins, smp_x, smp_y)

    assert np.allclose(conts, [2, 2])

    # Weighted case
    nbins = 2
    bins  = hep_spt.adbin_hist2d(smp_x, smp_y, nbins)

    recs, conts = hep_spt.adbin_hist2d_rectangles(bins, smp_x, smp_y, weights=weights)

    assert np.allclose(conts, [3, 3])


def test_adbin_histnd():
    '''
    Test for the "adbin_histnd" function.
    '''
    # Check the one-dimensional case
    sample = np.array([
        np.random.uniform(0, 10, 100),
        np.random.uniform(0, 10, 100),
        np.random.uniform(0, 10, 100),
    ]).T
    weights = np.random.uniform(0, 1, 100)

    # Case without weights
    bins = hep_spt.adbin_histnd(sample, 10)

    # Case with weights
    bins = hep_spt.adbin_histnd(sample, 10, weights=weights)


def _within_expectations( bins, vmin, vmax ):
    '''
    Check whether the values in the input array are inside the expectations.
    '''
    arr = np.array([b.sw(b.array, b.weights) for b in bins])

    return np.logical_and(np.all(arr > vmin), np.all(arr < vmax))
