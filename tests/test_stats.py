'''
Test functions for the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__ = ['miguel.ramos.pernas@cern.ch']


# Python

# Local


import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import ks_2samp as scipy_ks_2samp
import hep_spt
from hep_spt.stats.poisson import __poisson_to_gauss__


def _integral_transformer_aux(points, comp, values=None):
    '''
    Auxiliar function to perform check of the FlatDistTransform class.
    '''
    tr = hep_spt.FlatDistTransform(points, values)

    vals = tr.transform(comp)

    # Check that the values are between 0 and 1
    assert np.all(vals >= 0) and np.all(vals <= 1)

    bins = 20
    values, edges = np.histogram(vals, bins, range=(0, 1))

    centers = hep_spt.cfe(edges)

    p, residuals, _, _, _ = np.polyfit(centers, values, 0, full=True)

    # Check the mean of the values (depends on the number of bins and on the
    # length of the samples)
    assert np.isclose(p, len(points)/float(bins))

    chi2_ndof = residuals/(len(values) - 1.)

    return chi2_ndof


def test_calc_poisson_fu():
    '''
    Test for the "calc_poisson_fu" function.
    '''
    assert np.allclose(hep_spt.calc_poisson_fu(0), hep_spt.poisson_fu(0))
    assert np.allclose(hep_spt.calc_poisson_fu(10), hep_spt.poisson_fu(10))
    assert np.allclose(hep_spt.calc_poisson_fu(
        [10, 20]), hep_spt.poisson_fu([10, 20]).T)


def test_calc_poisson_llu():
    '''
    Test for the "calc_poisson_llu" function.
    '''
    assert np.allclose(hep_spt.calc_poisson_llu(0), hep_spt.poisson_llu(0))
    assert np.allclose(hep_spt.calc_poisson_llu(10), hep_spt.poisson_llu(10))
    assert np.allclose(hep_spt.calc_poisson_llu(
        [10, 20]), hep_spt.poisson_llu([10, 20]).T)


def test_clopper_pearson_int():
    '''
    Test the function to calculate frequentist intervals on
    binomial probabilities.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.clopper_pearson_int(0, 1)
    assert sl == 0.

    sl, sr = hep_spt.clopper_pearson_int(1, 1)
    assert sr == 1.

    sl, sr = hep_spt.clopper_pearson_int(1, 2)
    assert np.isclose(sr, 1. - sl)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.clopper_pearson_int([0, 1, 1], [1, 1, 2])
    assert sl[0] == 0. and sr[1] == 1. and np.isclose(sr[2], 1. - sl[2])


def test_clopper_pearson_unc():
    '''
    Test the function to calculate frequentist uncertainties on
    binomial probabilities.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.clopper_pearson_unc(0, 1)
    assert sl == 0.

    sl, sr = hep_spt.clopper_pearson_unc(1, 1)
    assert sr == 0.

    sl, sr = hep_spt.clopper_pearson_unc(1, 2)
    assert np.isclose(sl, sr)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.clopper_pearson_unc([0, 1, 1], [1, 1, 2])
    assert sl[0] == 0. and sr[1] == 0. and np.isclose(sl[2], sr[2])


def test_flatdisttransform():
    '''
    Test function for the class "FlatDistTransform"
    '''
    np.random.seed(1357)

    points = np.linspace(0., 50., 10000)
    values = np.exp(-points)
    runi = np.random.exponential(1, 10000)

    # Build a transformer from function values (not recommended, but might be
    # possible).
    _integral_transformer_aux(points, runi, values)

    # Build the transformer from a distribution, and check that it transforms
    # into a flat distribution with a nice chi2/ndof.
    chi2_ndof = _integral_transformer_aux(runi, runi)

    assert chi2_ndof < 1.


def test_gauss_unc():
    '''
    Test for the "gauss_unc" function.
    '''
    r = np.array([1, 1])
    s1 = hep_spt.gauss_unc(1)
    s2 = hep_spt.gauss_unc(r)

    assert np.allclose(s1, 1)
    assert np.allclose(s2, r)


def test_ks_2samp():
    '''
    Test the "ks_2samp" function.
    '''
    na = 200
    nb = 300

    a = norm.rvs(size=na, loc=0., scale=1.)
    b = norm.rvs(size=nb, loc=0.5, scale=1.5)

    # The results without weights must be the same as those from scipy
    scipy_res = scipy_ks_2samp(a, b)
    hep_spt_res = hep_spt.ks_2samp(a, b)

    assert np.allclose(scipy_res, hep_spt_res)

    # With weights equal to one for each entry, the result must be the
    # same as in scipy.
    wa = np.ones(na, dtype=float)
    wb = np.ones(nb, dtype=float)

    hep_spt_res = hep_spt.ks_2samp(a, b, wa, wb)

    assert np.allclose(scipy_res, hep_spt_res)


def test_stat_values():
    '''
    Test the "stat_values" function.
    '''
    # Non-weighted
    arr = np.array([1, 2, 2, 1, 3, 3])
    vals = hep_spt.stat_values(arr)

    assert np.allclose(vals.mean, 2)
    assert np.allclose(vals.var, 0.8)

    # Weighted
    wgts = np.array([2, 4, 4, 2, 6, 6])
    vals = hep_spt.stat_values(arr, weights=wgts)

    assert np.allclose(vals.mean, 7./3)
    assert np.allclose(vals.var, 2./3)

    # Test with arrays (non-weighted)
    arr = np.array([
        [1, 2, 2, 1, 3, 3],
        [2, 4, 4, 2, 6, 6]
    ])

    vals = hep_spt.stat_values(arr)

    assert np.allclose(vals.mean, 3.)
    assert np.allclose(vals.var, 32./11)

    # Test with arrays (weighted)
    wgts = np.array([
        [2, 4, 4, 2, 6, 6],
        [2, 4, 4, 2, 6, 6],
    ])

    vals = hep_spt.stat_values(arr, weights=wgts)

    assert np.allclose(vals.mean, 3.5)
    assert np.allclose(vals.var, 3)

    # Test with arrays for a given axis (non-weighted)
    vals = hep_spt.stat_values(arr, axis=0)

    assert np.allclose(vals.mean, [1.5, 3., 3., 1.5, 4.5, 4.5])
    assert np.allclose(vals.var, [0.5, 2, 2, 0.5, 4.5, 4.5])

    vals = hep_spt.stat_values(arr, axis=1)

    assert np.allclose(vals.mean, [[2], [4]])
    assert np.allclose(vals.var, [[0.8], [3.2]])
    assert np.allclose(vals.var_mean, [[0.8/6], [3.2/6]])

    # Test with arrays for a given axis (weighted)
    vals = hep_spt.stat_values(arr, weights=wgts, axis=0)

    assert np.allclose(vals.mean, [1.5, 3, 3, 1.5, 4.5, 4.5])
    assert np.allclose(vals.var, [0.5, 2, 2, 0.5, 4.5, 4.5])

    vals = hep_spt.stat_values(arr, weights=wgts, axis=1)

    assert np.allclose(vals.mean, [[7./3], [14./3]])
    assert np.allclose(vals.var, [[2./3], [8./3]])
    assert np.allclose(vals.var_mean, [[2./18], [8./18]])


def test_poisson_fu():
    '''
    Test the functions to calculate poissonian frequentist
    uncertainties.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.calc_poisson_fu(0)
    assert sl == 0

    sl, sr = hep_spt.poisson_fu(2*__poisson_to_gauss__)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.calc_poisson_fu([0, 1])
    for i, (l, r) in enumerate(zip(sl, sr)):
        lr, rr = hep_spt.calc_poisson_fu(i)
        assert np.isclose((l, r), (lr, rr)).all()

    # Calling the function with a float value must raise a TypeError
    with pytest.raises(TypeError):
        hep_spt.poisson_fu(1.)

    # Calling the function with non-positive values raises a ValueError
    with pytest.raises(ValueError):
        hep_spt.poisson_fu(-1)

    with pytest.raises(ValueError):
        hep_spt.poisson_fu([-1, 1])


def test_poisson_llu():
    '''
    Test the functions to calculate poissonian uncertainties based
    on the logarithm of likelihood.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.calc_poisson_llu(0)
    assert sl == 0

    sl, sr = hep_spt.poisson_llu(2*__poisson_to_gauss__)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.calc_poisson_llu([0, 1])
    for i, (l, r) in enumerate(zip(sl, sr)):
        lr, rr = hep_spt.calc_poisson_llu(i)
        assert np.isclose((l, r), (lr, rr)).all()

    # Calling the function with a float value must raise a TypeError
    with pytest.raises(TypeError):
        hep_spt.poisson_llu(1.)

    # Calling the function with non-positive values raises a ValueError
    with pytest.raises(ValueError):
        hep_spt.poisson_llu(-1)

    with pytest.raises(ValueError):
        hep_spt.poisson_llu([-1, 1])


def test_rv_random_sample():
    '''
    Test for the "rv_random_sample" function.
    '''
    pdf = norm(0, 2)
    smp = hep_spt.rv_random_sample(pdf, size=100)
    assert smp.shape == (100,)

    pdf = norm([0, 1], [2, 4])
    smp = hep_spt.rv_random_sample(pdf, size=100)
    assert smp.shape == (100, 2)


def test_sw2_unc():
    '''
    Test for the "sw2_unc" function.
    '''
    arr = np.array([1, 2, 3])
    assert np.allclose(hep_spt.sw2_unc(arr, bins=3), np.ones(3))
    assert np.allclose(hep_spt.sw2_unc(arr, bins=3, weights=arr), arr)


def test_wald_int():
    '''
    Test for the Wald interval function.
    '''
    p_l, p_u = hep_spt.wald_int(0, 1)
    assert p_l == 0.

    p_l, p_u = hep_spt.wald_int(1, 1)
    assert p_u == 1.


def test_wald_unc():
    '''
    Test for the Wald uncertainty function.
    '''
    s = hep_spt.wald_unc(0, 1)
    assert s == 0.

    s = hep_spt.wald_unc(1, 1)
    assert s == 0.


def test_wald_weighted_int():
    '''
    Test for the weighted Wald interval function.
    '''
    p_l, p_u = hep_spt.wald_weighted_int(np.zeros(3), np.ones(3))
    assert p_l == 0.

    p_l, p_u = hep_spt.wald_weighted_int(np.ones(3), np.ones(3))
    assert p_u == 1.

    p_l, p_u = hep_spt.wald_weighted_int(np.ones(2), np.ones(4))
    assert np.allclose([p_l, p_u], hep_spt.wald_int(2, 4))


def test_wald_weighted_unc():
    '''
    Test for the weighted Wald uncertainty function.
    '''
    s = hep_spt.wald_weighted_unc(np.zeros(3), np.ones(3))
    assert s == 0.

    s = hep_spt.wald_weighted_unc(np.ones(3), np.ones(3))
    assert s == 0.

    s = hep_spt.wald_weighted_unc(np.ones(2), np.ones(4))
    assert np.isclose(s, hep_spt.wald_unc(2, 4))


def test_wilson_int():
    '''
    Test for the Wilson interval function.
    '''
    # Test single value behaviour
    pl, pr = hep_spt.wilson_int(0, 1)
    assert pl == 0.

    pl, pr = hep_spt.wilson_int(1, 1)
    assert pr == 1.

    pl, pr = hep_spt.wilson_int(1, 2)
    assert np.isclose(pr, 1. - pl)

    # Test numpy.vectorize behaviour
    pl, pr = hep_spt.wilson_int([0, 1, 1], [1, 1, 2])
    assert pl[0] == 0. and pr[1] == 1. and np.isclose(pr[2], 1. - pl[2])


def test_wilson_unc():
    '''
    Test for the Wilson uncertainty function.
    '''
    sl, sr = hep_spt.wilson_unc(0, 1)
    assert sl == 0.

    sl, sr = hep_spt.wilson_unc(1, 1)
    assert sr == 0.

    sl, sr = hep_spt.wilson_unc(1, 2)
    assert np.isclose(sl, sr)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.wilson_unc([0, 1, 1], [1, 1, 2])
    assert sl[0] == 0. and sr[1] == 0. and np.isclose(sl[2], sr[2])
