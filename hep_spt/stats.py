'''
Function and classes representing statistical tools.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import os
import numpy as np
from math import exp, log, sqrt
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.stats import chi2
import warnings

# Local
from hep_spt import __project_path__
from hep_spt.core import decorate

# Define confidence intervals.
__chi2_one_dof__ = chi2(1)
__one_sigma__    = __chi2_one_dof__.cdf(1)
__two_sigma__    = __chi2_one_dof__.cdf(4)
__three_sigma__  = __chi2_one_dof__.cdf(9)
__five_sigma__   = __chi2_one_dof__.cdf(25)

# Define Poisson tolerance. If this number is 100, we can
# use Stirling's approximation of log(k!) with a relative
# error of 0.001.
__poisson_from_stirling__ = 100

# Number after which the poisson uncertainty is considered to
# be the same as that of a gaussian with "std = sqrt(lambda)".
# This number has been chosen so the difference of the
# relative uncertainty is ~0.5%.
__poisson_to_gauss__ = 200


__all__ = ['jac_poisson_float_l', 'poisson_float', 'calc_poisson_freq_uncert',
           'poisson_freq_uncert_one_sigma', 'process_uncert']


@decorate(np.vectorize)
def calc_poisson_freq_uncert( m, cl = __one_sigma__ ):
    '''
    Return the lower and upper bayesian uncertainties for
    a poisson distribution with mean "k" given the chi-square.

    :param m: mean of the Poisson distribution.
    :type m: float
    :param cl: confidence level (between 0 and 1).
    :type cl: float
    :returns: lower and upper uncertainties.
    :rtype: float, float
    '''
    sm = sqrt(m)

    if m > __poisson_to_gauss__:
        # Use the gaussian approximation of the uncertainty
        s = sqrt(m)
        return s, s

    alpha = (1. - cl)/2.

    if m < 1:
        # In this case there is only an upper uncertainty, so
        # the coverage is reset so it covers the whole "cl"
        lw = 0.
        alpha *= 2.
    else:
        ileft = m - sm
        if ileft < 0:
            ileft = 0.

        rleft = np.arange(m, m + 50*sm, dtype = int)
        fleft = lambda l: poisson_float(l, rleft).sum() - alpha
        jleft = lambda l: [jac_poisson_float_l(l, rleft).sum()]

        lw = fsolve(fleft, ileft, fprime = jleft)[0]

    iright = m + sm
    rright = np.arange(0, m + 1, dtype = int)
    fright = lambda l: poisson_float(l, rright).sum() - alpha
    jright = lambda l: [jac_poisson_float_l(l, rright).sum()]

    up = fsolve(fright, iright, fprime = jright)[0]

    return process_uncert(m, lw, up)


@decorate(np.vectorize)
def jac_poisson_float_l( l, k, tol = __poisson_from_stirling__ ):
    '''
    Return the value of Jacobian of the poisson_float
    function considering that it depends exclusively on "l".

    :param l: mean(s) of the Poisson.
    :type l: numpy.ndarray
    :param k: position(s) to evaluate.
    :type k: numpy.ndarray
    :param tol: tolerance(s) from which the Stirling's approximation \
    will be used.
    :type tol: numpy.ndarray
    :returns: value(s) of the poisson jacobian.
    :rtype: float
    '''
    if l == 0:
        return 0.

    return (k*1./l - 1.)*poisson_float(l, k, tol)


@decorate(np.vectorize)
def poisson_float( l, k, tol = __poisson_from_stirling__ ):
    '''
    Calculate the Poisson distribution value for floating
    numbers. The next term to the usual Stirling's
    approximation must be used to treat the cases where l ~ k.

    :param l: mean(s) of the Poisson distribution.
    :type l: numpy.ndarray
    :param k: position(s) to evaluate.
    :type k: numpy.ndarray
    :param tol: tolerance(s) from which the Stirling's approximation \
    will be used.
    :returns: value(s) of the Poisson distribution.
    :rtype: numpy.ndarray
    '''
    if l <= 0:
        return 0.

    if k > tol:
        return exp(k*log(l*1./k) + k - l - 0.5*log(2*np.pi*k))
    else:
        return 1./gamma(k + 1)*exp(k*log(l) - l)


def poisson_freq_uncert_one_sigma( m ):
    '''
    Return the poisson frequentist uncertainty at one standard
    deviation of confidence level. The input array recasted
    to int before doing the operation.

    :param m: measured value(s).
    :type m: array-like
    :returns: lower and upper frequentist uncertainties.
    :rtype: array-like(float, float)
    '''
    m = np.array(m, dtype = np.int32)

    out = np.zeros((len(m), 2), dtype = np.float64)

    ifile = os.path.join(__project_path__, 'data/poisson_freq_uncert_one_sigma.dat')

    table = np.loadtxt(ifile)

    no_app = (m < len(table))
    mk_app = np.logical_not(no_app)

    # Non-approximated uncertainties
    out[no_app] = table[m[no_app]]

    # Approximated uncertainties. This is not time-consuming. We
    # are just returning the square root of the input number.
    if mk_app.any():
        out[mk_app] = np.array(calc_poisson_freq_uncert(m[mk_app])).T

    return out


def process_uncert( m, lw, up ):
    '''
    Calculate the uncertainties and display an error if they
    have been incorrectly calculated.

    :param m: mean value.
    :type m: float
    :param lw: lower bound.
    :type lw: float
    :param up: upper bound.
    :type up: float
    :returns: lower and upper uncertainties.
    :type: array-like(float, float)
    '''
    s_lw = m - lw
    s_up = up - m

    if any(s < 0 for s in (s_lw, s_up)):
        warnings.warn('Poisson uncertainties have been '\
                      'incorrectly calculated')

    return s_lw, s_up


if __name__ == '__main__':
    '''
    Generate the tables to store the pre-calculated values of
    some uncertainties.
    '''
    conds = {
        'cl' : __one_sigma__,

        '__poisson_from_stirling__' : __poisson_from_stirling__,
        '__poisson_to_gauss__'      : __poisson_to_gauss__
        }

    m = np.arange(__poisson_to_gauss__)
    ucts = np.array(calc_poisson_freq_uncert(m, __one_sigma__)).T

    np.savetxt('data/poisson_freq_uncert_one_sigma.dat', ucts)
