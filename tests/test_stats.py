'''
Test functions for the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
import hep_spt

# Python
import numpy as np
from scipy.stats import norm
from scipy.stats import ks_2samp as scipy_ks_2samp


def test_poisson_freq_uncert():
    '''
    Test the functions to calculate poissonian frequentist
    uncertainties.
    '''
    sl, sr = hep_spt.calc_poisson_freq_uncert(0)
    assert sl == 0


def test_cp_freq_():
    '''
    Test the function to calculate frequentist uncertainties on
    efficiencies.
    '''
    sl, sr = hep_spt.calc_cp_freq_uncert(1, 1)
    assert sr == 0.

    sl, sr = hep_spt.calc_cp_freq_uncert(0, 1)
    assert sl == 0.

    sl, sr = hep_spt.calc_cp_freq_uncert(1, 2)
    assert np.isclose(sl, sr)


def test_ks():
    '''
    Test the Kolmogorov-Smirnov test function.
    '''
    na = 200
    nb = 300

    a  = norm.rvs(size=na, loc=0., scale=1.)
    b  = norm.rvs(size=nb, loc=0.5, scale=1.5)

    # The results without weights must be the same as those from scipy
    scipy_res   = scipy_ks_2samp(a, b)
    hep_spt_res = hep_spt.ks_2samp(a, b)

    assert scipy_res == hep_spt_res

    # With weights equal to one for each entry, the result must be the
    # same as in scipy.
    wa = np.ones(na, dtype=float)
    wb = np.ones(nb, dtype=float)

    hep_spt_res = hep_spt.ks_2samp(a, b, wa, wb)

    assert scipy_res == hep_spt_res
