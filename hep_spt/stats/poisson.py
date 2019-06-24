r'''
Function and classes representing statistical tools.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import os, warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import poisson

# Local
from hep_spt import __project_path__
from hep_spt.core import decorate, taking_ndarray
from hep_spt.stats.core import chi2_one_dof, one_sigma

# Number after which the poisson uncertainty is considered to
# be the same as that of a gaussian with "std = sqrt(lambda)".
__poisson_to_gauss__ = 200


__all__ = ['calc_poisson_fu',
           'calc_poisson_llu',
           'gauss_unc',
           'poisson_fu',
           'poisson_llu',
           'sw2_unc'
          ]


def _access_db( name ):
    r'''
    Access a database table under 'data/'.

    :param name: name of the file holding the data.
    :type name: str
    :returns: Array holding the data.
    :rtype: numpy.ndarray
    '''
    ifile = os.path.join(__project_path__, 'data', name)

    table = np.loadtxt(ifile)

    return table


@decorate(np.vectorize)
def calc_poisson_fu( m, cl = one_sigma ):
    r'''
    Return the lower and upper frequentist uncertainties for
    a poisson distribution with mean "m".

    :param m: mean of the Poisson distribution.
    :type m: float or np.ndarray(float)
    :param cl: confidence level (between 0 and 1).
    :type cl: float or np.ndarray(float)
    :returns: Lower and upper uncertainties.
    :rtype: (float, float) or np.ndarray(float, float)

    .. note:: This function might turn very time consuming. Consider using :func:`poisson_fu` instead.
    '''
    sm = np.sqrt(m)

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

    return _process_poisson_unc(m, lw, up)


@decorate(np.vectorize)
def calc_poisson_llu( m, cl = one_sigma ):
    r'''
    Calculate poisson uncertainties based on the logarithm of likelihood.

    :param m: mean of the Poisson distribution.
    :type m: float or numpy.ndarray(float)
    :param cl: confidence level (between 0 and 1).
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper uncertainties.
    :rtype: (float, float) or numpy.ndarray(float, float)

    .. note:: This function might turn very time consuming. Consider using :func:`poisson_llu` instead.
    '''
    ns = np.sqrt(chi2_one_dof.ppf(cl))

    nll = lambda x: -2.*np.log(poisson.pmf(m, x))

    ref = nll(m)

    func = lambda x: nll(x) - ref - ns

    il, ir = _poisson_initials(m)

    if m < 1:
        lw = m
    else:
        lw = fsolve(func, il)[0]

    up = fsolve(func, ir)[0]

    return _process_poisson_unc(m, lw, up)


def gauss_unc( s, cl = one_sigma ):
    r'''
    Calculate the gaussian uncertainty for a given confidence level.

    :param s: standard deviation of the gaussian.
    :type s: float or numpy.ndarray(float)
    :param cl: confidence level.
    :type cl: float
    :returns: Gaussian uncertainty.
    :rtype: float or numpy.ndarray(float)

    .. seealso:: :func:`poisson_fu`, :func:`poisson_llu`, :func:`sw2_unc`
    '''
    n = np.sqrt(chi2_one_dof.ppf(cl))

    return n*s


def poisson_fu( m ):
    r'''
    Return the poisson frequentist uncertainty at one standard
    deviation of confidence level.

    :param m: measured value(s).
    :type m: int or numpy.ndarray(int)
    :returns: Lower and upper frequentist uncertainties.
    :rtype: numpy.ndarray(float, float)

    .. seealso:: :func:`gauss_unc`, :func:`poisson_llu`, :func:`sw2_unc`
    '''
    return _poisson_unc_from_db(m, 'poisson_fu.dat')


def poisson_llu( m ):
    r'''
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
    :type m: int or numpy.ndarray(int)
    :returns: Lower and upper frequentist uncertainties.
    :rtype: numpy.ndarray(float, float)

    .. seealso:: :func:`gauss_unc`, :func:`poisson_fu`, :func:`sw2_unc`
    '''
    return _poisson_unc_from_db(m, 'poisson_llu.dat')


@taking_ndarray
def _poisson_initials( m ):
    r'''
    Return the boundaries to use as initial values in
    scipy.optimize.fsolve when calculating poissonian
    uncertainties.

    :param m: mean of the Poisson distribution.
    :type m: float or numpy.ndarray(float)
    :returns: Upper and lower boundaries.
    :rtype: (float, float) or numpy.ndarray(float, float)
    '''
    sm = np.sqrt(m)

    il = m - sm
    ir = m + sm

    # Needed by "calc_poisson_llu"
    if il.ndim == 0:
        if il <= 0:
            il = 0.1
    else:
        il[il <= 0] = 0.1

    return il, ir


def _poisson_unc_from_db( m, database ):
    r'''
    Used in functions to calculate poissonian uncertainties,
    which are partially stored on databases. If "m" is above the
    maximum number stored in the database, the gaussian approximation
    is taken instead.

    :param m: measured value(s).
    :type m: int or numpy.ndarray(int)
    :param database: name of the database.
    :type database: str
    :returns: Lower and upper frequentist uncertainties.
    :rtype: (float, float) or numpy.ndarray(float, float)
    :raises TypeError: if the input is a (has) non-integer value(s).
    :raises ValueError: if the input value(s) is(are) not positive.
    '''
    m = np.array(m)
    if not np.issubdtype(m.dtype, np.integer):
        raise TypeError('Calling function with a non-integer value')
    if np.any(m < 0):
        raise ValueError('Values must be positive')

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


def _process_poisson_unc( m, lw, up ):
    r'''
    Calculate the uncertainties and display an error if they
    have been incorrectly calculated.

    :param m: mean value.
    :type m: float
    :param lw: lower bound.
    :type lw: float
    :param up: upper bound.
    :type up: float
    :returns: Lower and upper uncertainties.
    :type: numpy.ndarray(float, float)
    '''
    s_lw = m - lw
    s_up = up - m

    if any(s < 0 for s in (s_lw, s_up)):
        warnings.warn('Poisson uncertainties have been '\
                      'incorrectly calculated')

    # numpy.vectorize needs to know the exact type of the output
    return float(s_lw), float(s_up)


def sw2_unc( arr, bins = 20, range = None, weights = None ):
    r'''
    Calculate the errors using the sum of squares of weights.
    The uncertainty is calculated as follows:

    .. math::

       \sigma_i = \sqrt{\sum_{j = 0}^{n - 1} \omega_{i,j}^2}

    where *i* refers to the i-th bin and :math:`j \in [0, n)` refers to
    each entry in that bin with weight :math:`\omega_{i,j}`. If "weights" is
    None, then this coincides with the square root of the number of entries
    in each bin.

    :param arr: input array of data to process.
    :param bins: see :func:`numpy.histogram`.
    :type bins: int, sequence of scalars or str
    :param range: range to process in the input array.
    :type range: None or tuple(float, float)
    :param weights: possible weights for the histogram.
    :type weights: None or numpy.ndarray(value-type)
    :returns: Symmetric uncertainty.
    :rtype: numpy.ndarray

    .. seealso:: :func:`gauss_unc`, :func:`poisson_fu`, :func:`poisson_llu`
    '''
    if weights is not None:
        values = np.histogram(arr, bins, range, weights = weights*weights)[0]
    else:
        values = np.histogram(arr, bins, range)[0]

    return np.sqrt(values)


if __name__ == '__main__':
    r'''
    Generate the tables to store the pre-calculated values of
    some uncertainties.
    '''
    m = np.arange(__poisson_to_gauss__)

    print('Creating databases:')
    for func in (calc_poisson_fu, calc_poisson_llu):

        ucts = np.array(func(m, one_sigma)).T

        name = func.__name__.replace('calc_', r'') + '.dat'

        fpath = os.path.join('data', name)

        print('- {}'.format(fpath))

        np.savetxt(fpath, ucts)
