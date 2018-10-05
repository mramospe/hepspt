'''
Test functions for the "histograms" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import matplotlib
import numpy as np
import pytest
from scipy.stats import norm

# Local
import hep_spt


def test_cfe():
    '''
    Test the "cfe" function.
    '''
    assert np.all(hep_spt.cfe(np.array([1, 3, 5, 7])) == [2, 4, 6])


def test_errorbar_hist():
    '''
    Test the behaviour of the function "errorbar_hist".
    '''
    # Check that if the weights are integers, the default error type is
    # not based on the squared sum of weights
    arr = np.random.poisson(4, 100)

    v, c = np.unique(arr, return_counts=True)

    values, edges, ex, ey = hep_spt.errorbar_hist(v, weights=c, uncert='freq')

    assert np.any(ey[0] != ey[1])

    # With weights the error must be a single array
    wgts = np.random.uniform(0, 1, 100)

    values, edges, ex, ey = hep_spt.errorbar_hist(arr, bins=20, weights=wgts)

    assert ey.shape == (20,)

    # Check the values returned in the normalization
    arr  = np.array([-1, -1, +0.5, +0.5, +1, +1, +2, +2, +3, +3])
    wgts = np.array([0.5, 0.5, 2, 1, 1, 1, 1, 1, 1, 1])

    values, edges, _, _ = hep_spt.errorbar_hist(arr, bins=4, range=(0, 4), norm=True, norm_type='range')
    assert np.allclose(values.sum(), 1)

    values, edges, _, _ = hep_spt.errorbar_hist(arr, bins=4, range=(0, 4), norm=True, norm_type='all')
    assert np.allclose(values.sum(), 0.8)

    values, edges, _, _ = hep_spt.errorbar_hist(arr, bins=4, range=(0, 4), weights=wgts, norm=True, norm_type='all')
    assert np.allclose(values.sum(), 0.9)

    # If the uncertainty type is unknown, raise an error
    with pytest.raises(ValueError):
        hep_spt.errorbar_hist(arr, bins=20, weights=wgts, uncert='none')

    # If the normalization type is unknown, raise an error
    with pytest.raises(ValueError):
        hep_spt.errorbar_hist(arr, norm=True, norm_type='none')

    # Normalizing with empty values in the bins will raise a RuntimeWarning
    with pytest.warns(RuntimeWarning):
        hep_spt.errorbar_hist(np.random.uniform(10, 20, 100),
                              range=(30, 40),
                              norm=True)


def test_process_range():
    '''
    Test the behaviour of the function "process_range".
    '''
    arr = np.random.uniform(0, 1, 98)

    arr = np.append(arr, -2)
    arr = np.append(arr, +2)

    # 1D case
    vmin, vmax = hep_spt.histograms.process_range(arr)
    assert vmin == -2 and vmax == np.nextafter(+2, np.infty)

    vmin, vmax = hep_spt.histograms.process_range(arr, (-3, +3))
    assert vmin == -3 and vmax == +3

    # 2D case
    x = np.random.uniform(0, 1, 98)
    y = np.random.uniform(0, 1, 98)

    arr = np.array([x, y]).T

    arr = np.append(arr, (-2, +2))
    arr = np.append(arr, (+2, -2))

    vmin, vmax = hep_spt.histograms.process_range(arr)
    assert np.all(vmin == (-2, -2)) and np.all(vmax == np.nextafter((+2, +2), np.infty))

    vmin, vmax = hep_spt.histograms.process_range(arr, [(-3, -3), (+3, +3)])
    assert np.all(vmin == (-3, -3)) and np.all(vmax == (+3, +3))


def test_profile():
    '''
    Test the function "profile".
    '''
    # Equally distributed values
    x = np.arange(10)
    y = np.ones_like(x)

    with pytest.warns(RuntimeWarning):
        # The warning is actually raised by "stat_values" when calculating
        # the stantard deviation
        _, prof, _ = hep_spt.profile(x, y, bins=10)

    assert np.all(prof == 1)

    # Test also the standard deviation
    y = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    _, prof, std = hep_spt.profile(x, y, bins=5)

    assert np.allclose(prof, np.arange(1, 6))
    assert np.allclose(std, 0)

    # Raise ValueError if an unkown standard deviation type is given
    with pytest.raises(ValueError):
        hep_spt.profile(x, y, bins=5, std_type='none')


def test_pull_sym():
    '''
    Test the "pull" function with symmetric errors.
    '''
    size = 10000
    data = np.random.normal(0, 2, size)

    values, edges, ex, ey = hep_spt.errorbar_hist(data, uncert='sw2')

    centers = hep_spt.cfe(edges)

    rv = norm.pdf(centers)
    ref = float(size)*rv/rv.sum()

    pull, perr = hep_spt.pull(values, ey, ref)

    assert perr.shape == (len(values),)

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)


def test_pull_asym():
    '''
    Test the "pull" function with asymmetric errors.
    '''
    size = 1000
    data = np.random.normal(0, 2, size)

    values, edges, ex, ey = hep_spt.errorbar_hist(data, uncert='freq')

    centers = hep_spt.cfe(edges)

    rv = norm.pdf(centers)
    ref = float(size)*rv/rv.sum()

    pull, perr = hep_spt.pull(values, ey, ref)

    assert perr.shape == (2, len(values))

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)


def test_pull_sym_sym():
    '''
    Test the "pull" function with symmetric errors in the values and in the
    reference
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([3,  6, 12])
    ref        = np.array([9, 10, 26])
    ref_err    = np.array([4,  8,  5])

    pull, perr = hep_spt.pull(values, values_err, ref, ref_err)

    assert perr.shape == (len(values),)

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)
    assert np.allclose(pull, [-1, +1, -1])


def test_pull_asym_asym():
    '''
    Test the "pull" function with asymmetric errors in the values and in the
    reference
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([
        np.array([3,  6, 12]),
        np.array([4,  8,  5])
        ])
    ref     = np.array([9, 10, 26])
    ref_err = np.array([
        np.array([3,  6, 12]),
        np.array([4,  8,  5])
        ])

    pull, perr = hep_spt.pull(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)
    assert np.allclose(pull, [-1, +1, -1])


def test_pull_sym_asym():
    '''
    Test the "pull" function with symmetric errors in the values and
    asymmetric in the reference
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([4,  8,  5])
    ref        = np.array([9, 10, 26])
    ref_err    = np.array([
            np.array([3,  8, 12]),
            np.array([4,  6,  5])
            ])

    pull, perr = hep_spt.pull(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)
    assert np.allclose(pull, [-1, +1, -1])


def test_pull_asym_sym():
    '''
    Test the "pull" function with asymmetric errors in the values and
    symmetric in the reference
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([
            np.array([4,  6,  5]),
            np.array([3,  8, 12])
            ])
    ref     = np.array([9, 10, 26])
    ref_err = np.array([4,  8,  5])

    pull, perr = hep_spt.pull(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))

    p = values - ref

    assert np.all(pull[p >= 0] >= 0)
    assert np.all(pull[p < 0] < 0)
    assert np.allclose(pull, [-1, +1, -1])


def test_residual_sym():
    '''
    Test the "residual" function given symmetric errors.
    '''
    size = 10000
    data = np.random.normal(0, 2, size)

    values, edges, ex, ey = hep_spt.errorbar_hist(data, uncert='sw2')

    centers = hep_spt.cfe(edges)

    rv = norm.pdf(centers)
    ref = float(size)*rv/rv.sum()

    res, perr = hep_spt.residual(values, ey, ref)

    assert perr.shape == (len(values),)
    assert np.allclose(res, values - ref)


def test_residual_asym():
    '''
    Test the "residual" function with asymmetric errors.
    '''
    size = 1000
    data = np.random.normal(0, 2, size)

    values, edges, ex, ey = hep_spt.errorbar_hist(data, uncert='freq')

    centers = hep_spt.cfe(edges)

    rv = norm.pdf(centers)
    ref = float(size)*rv/rv.sum()

    res, perr = hep_spt.residual(values, ey, ref)

    assert perr.shape == (2, len(values))
    assert np.allclose(res, values - ref)


def test_residual_sym_sym():
    '''
    Test the "residual" function with symmetric errors in the values and in the
    reference.
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([3,  6, 12])
    ref        = np.array([9, 10, 26])
    ref_err    = np.array([4,  8,  5])

    res, perr = hep_spt.residual(values, values_err, ref, ref_err)

    assert perr.shape == (len(values),)
    assert np.allclose(res, values - ref)
    assert np.allclose(perr, [5, 10, 13])


def test_residual_asym_asym():
    '''
    Test the "residual" function with asymmetric errors in the values and in the
    reference.
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([
        np.array([3,  6, 12]),
        np.array([4,  8,  5])
        ])
    ref     = np.array([9, 10, 26])
    ref_err = np.array([
        np.array([3,  6, 12]),
        np.array([4,  8,  5])
        ])

    res, perr = hep_spt.residual(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))
    assert np.allclose(res, values - ref)
    assert np.allclose(perr[0], [5, 10, 13])
    assert np.allclose(perr[1], [5, 10, 13])


def test_residual_sym_asym():
    '''
    Test the "residual" function with symmetric errors in the values and
    asymmetric in the reference.
    '''
    values     = np.array([4, 23, 13])
    values_err = np.array([3,  8,  5])
    ref        = np.array([9, 10, 26])
    ref_err    = np.array([
            np.array([4,  6,  5]),
            np.array([3,  6, 12])
            ])

    res, perr = hep_spt.residual(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))
    assert np.allclose(res, values - ref)
    assert np.allclose(perr[0][1:], [10, 13])
    assert np.allclose(perr[1][0] , 5)


def test_residual_asym_sym():
    '''
    Test the "residual" function with asymmetric errors in the values and
    symmetric in the reference.
    '''
    values     = np.array([4, 20, 13])
    values_err = np.array([
            np.array([4,  6,  5]),
            np.array([3,  6, 12])
            ])
    ref     = np.array([9, 10, 26])
    ref_err = np.array([3,  8,  5])

    res, perr = hep_spt.residual(values, values_err, ref, ref_err)

    assert perr.shape == (2, len(values))
    assert np.allclose(res, values - ref)
    assert np.allclose(perr[0][0] , 5)
    assert np.allclose(perr[1][1:], [10, 13])


def test_weights_by_edges():
    '''
    Test the "weights_by_edges" function.
    '''
    # Simply call the function
    v = np.random.uniform(0, 1, 10000)
    n = 5
    e = np.linspace(0, 1, n)
    w = np.random.normal(0, 5, n - 1)

    wgts = hep_spt.weights_by_edges(v, e, w)

    assert wgts.shape == v.shape

    # Do an exact calculation
    v = np.arange(18, dtype=float)
    e = np.arange(20, step=2, dtype=float)
    w = np.ones(9, dtype=float)

    wgts = hep_spt.weights_by_edges(v, e, w)

    assert np.allclose(wgts, np.full(18, 1))

    # Raise if the arrays have incorrect dimensions
    with pytest.raises(TypeError):
        hep_spt.weights_by_edges(np.array([v, v]), e, w)

    with pytest.raises(TypeError):
        hep_spt.weights_by_edges(v, np.concatenate([e, e]), w)

    # Raise if the data lies outside the edges
    v = np.arange(30, dtype=float)

    with pytest.raises(ValueError):
        hep_spt.weights_by_edges(v, e, w)

    v = np.arange(-1, 20, dtype=float)

    with pytest.raises(ValueError):
        hep_spt.weights_by_edges(v, e, w)
