'''
Function and classes representing statistical tools.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import os, warnings
import numpy as np
from math import exp, log, sqrt
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.stats import beta, chi2, kstwobign, poisson
from scipy.stats import ks_2samp as scipy_ks_2samp

# Local
from hep_spt import __project_path__
from hep_spt.core import decorate

# Define confidence intervals.
__chi2_one_dof__ = chi2(1)
__one_sigma__    = __chi2_one_dof__.cdf(1)

# Number after which the poisson uncertainty is considered to
# be the same as that of a gaussian with "std = sqrt(lambda)".
__poisson_to_gauss__ = 200


__all__ = ['calc_poisson_fu', 'calc_poisson_llu',
           'cp_fu', 'ks_2samp',
           'gauss_u',
           'poisson_fu', 'poisson_llu',
          ]


def _access_db( name ):
    '''
    Access a database table under 'data/'.

    :param name: name of the file holding the data.
    :type name: str
    :returns: array holding the data.
    :rtype: numpy.ndarray
    '''
    ifile = os.path.join(__project_path__, 'data', name)

    table = np.loadtxt(ifile)

    return table


@decorate(np.vectorize)
def calc_poisson_fu( m, cl = __one_sigma__ ):
    '''
    Return the lower and upper frequentist uncertainties for
    a poisson distribution with mean "m".

    :param m: mean of the Poisson distribution.
    :type m: float
    :param cl: confidence level (between 0 and 1).
    :type cl: float
    :returns: lower and upper uncertainties.
    :rtype: float, float

    .. note:: This function might turn very time consuming. Consider using :func:`poisson_fu` instead.
    '''
    sm = sqrt(m)

    alpha = (1. - cl)/2.

    il, ir = _poisson_initials(m)

    if m < 1:
        # In this case there is only an upper uncertainty, so
        # the coverage is reset so it covers the whole "cl"
        lw = m
        alpha *= 2.
    else:
        fleft = lambda l: 1. - (poisson.cdf(m, l) - poisson.pmf(m, l)) - alpha

        lw = fsolve(fleft, il)[0]

    fright = lambda l: poisson.cdf(m, l) - alpha

    up = fsolve(fright, ir)[0]

    return _process_poisson_u(m, lw, up)


@decorate(np.vectorize)
def calc_poisson_llu( m, cl = __one_sigma__ ):
    '''
    Calculate poisson uncertainties based on the logarithm of likelihood.

    :param m: mean of the Poisson distribution.
    :type m: float
    :param cl: confidence level (between 0 and 1).
    :type cl: float
    :returns: lower and upper uncertainties.
    :rtype: float, float

    .. note:: This function might turn very time consuming. Consider using :func:`poisson_llu` instead.
    '''
    ns = np.sqrt(__chi2_one_dof__.ppf(cl))

    nll = lambda x: -2.*np.log(poisson.pmf(m, x))

    ref = nll(m)

    func = lambda x: nll(x) - ref - ns

    il, ir = _poisson_initials(m)

    if m < 1:
        lw = m
    else:
        lw = fsolve(func, il)[0]

    up = fsolve(func, ir)[0]

    return _process_poisson_u(m, lw, up)


@decorate(np.vectorize)
def cp_fu( k, N, cl = __one_sigma__ ):
    '''
    Return the frequentist Clopper-Pearson uncertainties of having
    "k" events in "N".

    :param k: passed events.
    :type k: int
    :param N: total number of events.
    :type N: int
    :param cl: confidence level.
    :type cl: float
    :returns: lower and upper uncertainties on the efficiency.
    :rtype: float
    '''
    p = float(k)/N

    pcl = 0.5*(1. - cl)

    # Lower uncertainty
    if k != 0:
        lw = beta(k, N - k + 1).ppf(pcl)
    else:
        lw = p

    # Upper uncertainty
    if k != N:
        up = beta(k + 1, N - k).ppf(1. - pcl)
    else:
        up = p

    return p - lw, up - p


def _poisson_u_from_db( m, database ):
    '''
    Decorator for functions to calculate poissonian uncertainties,
    which are partially stored on databases. If "m" is above the
    maximum number stored in the database, the gaussian approximation
    is taken instead.

    :param database: name of the database.
    :type database: str
    :returns: lower and upper frequentist uncertainties.
    :rtype: array-like(float, float)
    '''
    m = np.array(m, dtype = np.int32)

    scalar_input = False
    if m.ndim == 0:
        m = m[None]
        scalar_input = True

    no_app = (m < __poisson_to_gauss__)

    if np.count_nonzero(no_app) == 0:
        # We can use the gaussian approximation in all
        out = np.array(2*[np.sqrt(m)]).T
    else:
        # Non-approximated uncertainties
        table = _access_db(database)

        out = np.zeros((len(m), 2), dtype = np.float64)

        out[no_app] = table[m[no_app]]

        mk_app = np.logical_not(no_app)

        if mk_app.any():
            # Use the gaussian approximation for the rest
            out[mk_app] = np.array(2*[np.sqrt(m[mk_app])]).T

    if scalar_input:
        return np.squeeze(out)
    return out


def gauss_u( s, cl = __one_sigma__ ):
    '''
    Calculate the gaussian uncertainty for a given confidence level.

    :param s: standard deviation of the gaussian.
    :type s: float or collection(float)
    :param cl: confidence level.
    :type cl: float
    :returns: gaussian uncertainty.
    :rtype: float or collection(float)
    '''
    n = np.sqrt(__chi2_one_dof__.ppf(cl))

    return n*s


def _ks_2samp_values( arr, wgts = None ):
    '''
    Calculate the values needed to perform the Kolmogorov-Smirnov test.

    :param arr: input sample.
    :type arr: array-like
    :param wgts: possible weights.
    :type wgts: array-like
    :returns: sorted sample, stack with the cumulative distribution and
    sum of weights.
    :rtype: array-like, array-like, float
    '''
    wgts = wgts if wgts is not None else np.ones(len(arr), dtype=float)

    ix   = np.argsort(arr)
    arr  = arr[ix]
    wgts = wgts[ix]

    cs = np.cumsum(wgts)

    sw = cs[-1]

    hs = np.hstack((0, cs/sw))

    return arr, hs, sw


def ks_2samp( a, b, wa = None, wb = None ):
    '''
    Compute the Kolmogorov-Smirnov statistic on 2 samples. This is a two-sided
    test for the null hypothesis that 2 independent samples are drawn from the
    same continuous distribution. Weights for each sample are accepted. If no
    weights are provided, then the function scipy.stats.ks_2samp is called
    instead.

    :param a: first sample.
    :type a: array-like
    :param b: second sample.
    :type b: array-like
    :param wa: set of weights for "a". Same length as "a".
    :type wa: array-like or None.
    :param wb: set of weights for "b". Same length as "b".
    :type wb: array-like or None.
    :returns: test statistic and two-tailed p-value.
    :rtype: float, float
    '''
    if wa is None and wb is None:
        return scipy_ks_2samp(a, b)

    a, cwa, na = _ks_2samp_values(a, wa)
    b, cwb, nb = _ks_2samp_values(b, wb)

    m = np.concatenate([a, b])

    cdfa = cwa[np.searchsorted(a, m, side='right')]
    cdfb = cwb[np.searchsorted(b, m, side='right')]

    d = np.max(np.abs(cdfa - cdfb))

    en = np.sqrt(na*nb/float(na + nb))
    try:
        prob = kstwobign.sf((en + 0.12 + 0.11/en)*d)
    except:
        prob = 1.

    return d, prob


def poisson_fu( m ):
    '''
    Return the poisson frequentist uncertainty at one standard
    deviation of confidence level.

    :param m: measured value(s).
    :type m: array-like
    :returns: lower and upper frequentist uncertainties.
    :rtype: array-like(float, float)

    .. note:: The input array is recasted to integer type before doing the operation.
    '''
    return _poisson_u_from_db(m, 'poisson_fu.dat')


def poisson_llu( m ):
    '''
    Return the poisson uncertainty at one standard deviation of
    confidence level. The lower and upper uncertainties are defined
    by those two points with a variation of one in the value of the
    negative logarithm of the likelihood multiplied by two:

    .. math::
       \sigma_\\text{low} = n_\\text{obs} - \lambda_\\text{low}

    .. math::
       \\alpha - 2\log P(n_\\text{obs}|\lambda_\\text{low}) = 1

    .. math::
       \sigma_\\text{up} = \lambda_\\text{up} - n_\\text{obs}

    .. math::
       \\alpha - 2\log P(n_\\text{obs}|\lambda_\\text{up}) = 1

    where :math:`\\alpha = 2\log P(n_\\text{obs}|n_\\text{obs})`.

    :param m: measured value(s).
    :type m: array-like
    :returns: lower and upper frequentist uncertainties.
    :rtype: array-like(float, float)

    .. note:: The input array is recasted to integer type before doing the operation.
    '''
    return _poisson_u_from_db(m, 'poisson_llu.dat')


def _poisson_initials( m ):
    '''
    Return the boundaries to use as initial values in
    scipy.optimize.fsolve when calculating poissonian
    uncertainties.

    :param m: mean of the Poisson distribution.
    :type m: float
    :returns: upper and lower boundaries.
    :rtype: float, float
    '''
    sm = np.sqrt(m)

    il = m - sm
    if il <= 0:
        # Needed by "calc_poisson_llu"
        il = 0.1
    ir = m + sm

    return il, ir


def _process_poisson_u( m, lw, up ):
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

    # numpy.vectorize needs to know the exact type of the output
    return float(s_lw), float(s_up)


if __name__ == '__main__':
    '''
    Generate the tables to store the pre-calculated values of
    some uncertainties.
    '''
    m = np.arange(__poisson_to_gauss__)

    print('Creating databases:')
    for func in (calc_poisson_fu, calc_poisson_llu):

        ucts = np.array(func(m, __one_sigma__)).T

        name = func.__name__.replace('calc_', '') + '.dat'

        fpath = os.path.join('data', name)

        print('- {}'.format(fpath))

        np.savetxt(fpath, ucts)
