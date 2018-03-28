'''
Test functions for the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
from scipy.stats import norm
from scipy.stats import ks_2samp as scipy_ks_2samp

# Local
import hep_spt


def _integral_transformer_aux( points, comp, values=None ):
    '''
    Auxiliar function to perform check of the FlatDistTransform class.
    '''
    tr = hep_spt.FlatDistTransform(points, values)

    vals = tr.transform(comp)

    # Check that the values are between 0 and 1
    assert np.all(vals >= 0) and np.all(vals <= 1)

    bins = 20
    values, edges = np.histogram(vals, bins, range=(0, 1))

    centers = hep_spt.centers_from_edges(edges)

    p, residuals, _, _, _ = np.polyfit(centers, values, 0, full=True)

    # Check the mean of the values (depends on the number of bins and on the
    # length of the samples)
    assert np.isclose(p, len(points)/float(bins))

    chi2_ndof = residuals/(len(values) - 1.)

    return chi2_ndof


def test_integral_transformer():
    '''
    Test function for the integral transformer instance.
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


def test_poisson_fu():
    '''
    Test the functions to calculate poissonian frequentist
    uncertainties.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.calc_poisson_fu(0)
    assert sl == 0

    sl, sr = hep_spt.poisson_fu(2*hep_spt.stats.__poisson_to_gauss__)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.calc_poisson_fu([0, 1])
    for i, (l, r) in enumerate(zip(sl, sr)):
        lr, rr = hep_spt.calc_poisson_fu(i)
        assert np.isclose((l, r), (lr, rr)).all()


def test_poisson_llu():
    '''
    Test the functions to calculate poissonian uncertainties based
    on the logarithm of likelihood.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.calc_poisson_llu(0)
    assert sl == 0

    sl, sr = hep_spt.poisson_llu(2*hep_spt.stats.__poisson_to_gauss__)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.calc_poisson_llu([0, 1])
    for i, (l, r) in enumerate(zip(sl, sr)):
        lr, rr = hep_spt.calc_poisson_llu(i)
        assert np.isclose((l, r), (lr, rr)).all()


def test_cp_fu():
    '''
    Test the function to calculate frequentist uncertainties on
    efficiencies.
    '''
    # Test single value behaviour
    sl, sr = hep_spt.cp_fu(0, 1)
    assert sl == 0.

    sl, sr = hep_spt.cp_fu(1, 1)
    assert sr == 0.

    sl, sr = hep_spt.cp_fu(1, 2)
    assert np.isclose(sl, sr)

    # Test numpy.vectorize behaviour
    sl, sr = hep_spt.cp_fu([0, 1, 1], [1, 1, 2])
    assert sl[0] == 0. and sr[1] == 0. and np.isclose(sl[2], sr[2])


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
