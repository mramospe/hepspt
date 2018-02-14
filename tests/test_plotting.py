'''
Test functions for the "plotting" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np

# Custom
import hep_spt


def test_process_range():
    '''
    Test the behaviour of the function "process_range".
    '''
    arr = np.random.uniform(0, 1, 98)

    arr = np.append(arr, -2)
    arr = np.append(arr, +2)

    # 1D case
    vmin, vmax = hep_spt.process_range(arr)
    assert vmin == -2 and vmax == np.nextafter(+2, np.infty)

    vmin, vmax = hep_spt.process_range(arr, (-3, +3))
    assert vmin == -3 and vmax == +3

    # 2D case
    x = np.random.uniform(0, 1, 98)
    y = np.random.uniform(0, 1, 98)

    arr = np.array([x, y]).T

    arr = np.append(arr, (-2, +2))
    arr = np.append(arr, (+2, -2))

    vmin, vmax = hep_spt.process_range(arr)
    assert np.all(vmin == (-2, -2)) and np.all(vmax == np.nextafter((+2, +2), np.infty))

    vmin, vmax = hep_spt.process_range(arr, [(-3, +3), (-3, +3)])
    assert np.all(vmin == (-3, -3)) and np.all(vmax == (+3, +3))


def test_profile():
    '''
    Test the function "profile".
    '''
    x = np.arange(10)
    y = np.ones_like(x)

    prof = hep_spt.profile(x, y, bins=10)

    assert np.all(prof == 1)
