'''
Test functions for the "plotting" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import pytest
from scipy.stats import norm

# Local
import hep_spt


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

    # If the uncertainty type is unknown, raise an error
    with pytest.raises(ValueError):
        hep_spt.errorbar_hist(arr, bins=20, weights=wgts, uncert='none')

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
    vmin, vmax = hep_spt.plotting.process_range(arr)
    assert vmin == -2 and vmax == np.nextafter(+2, np.infty)

    vmin, vmax = hep_spt.plotting.process_range(arr, (-3, +3))
    assert vmin == -3 and vmax == +3

    # 2D case
    x = np.random.uniform(0, 1, 98)
    y = np.random.uniform(0, 1, 98)

    arr = np.array([x, y]).T

    arr = np.append(arr, (-2, +2))
    arr = np.append(arr, (+2, -2))

    vmin, vmax = hep_spt.plotting.process_range(arr)
    assert np.all(vmin == (-2, -2)) and np.all(vmax == np.nextafter((+2, +2), np.infty))

    vmin, vmax = hep_spt.plotting.process_range(arr, [(-3, -3), (+3, +3)])
    assert np.all(vmin == (-3, -3)) and np.all(vmax == (+3, +3))


def test_profile():
    '''
    Test the function "profile".
    '''
    x = np.arange(10)
    y = np.ones_like(x)

    prof = hep_spt.profile(x, y, bins=10)

    assert np.all(prof == 1)


def test_pull_sym():
    '''
    Test the "pull" function with symmetric errors.
    '''
    size = 10000
    data = np.random.normal(0, 2, size)

    values, edges, ex, ey = hep_spt.errorbar_hist(data, uncert='sw2')

    centers = hep_spt.centers_from_edges(edges)

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

    centers = hep_spt.centers_from_edges(edges)

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
