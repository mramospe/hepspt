'''
Test functions for the "plotting" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import pytest

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
