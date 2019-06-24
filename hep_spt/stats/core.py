r'''
Core of function and classes for statistical calculations.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']

# Python
import numpy as np
from collections import namedtuple
from scipy.interpolate import interp1d
from scipy.stats import chi2, kstwobign
from scipy.stats import ks_2samp as scipy_ks_2samp

# Define confidence intervals.
chi2_one_dof = chi2(1)
one_sigma    = chi2_one_dof.cdf(1)

__all__ = ['FlatDistTransform',
           'ks_2samp',
           'rv_random_sample',
           'stat_values']


class FlatDistTransform(object):
    r'''
    Instance to transform values following an unknown distribution :math:`f(x)`
    into a flat distribution. This class takes into account the inverse
    transform sampling theorem, which states that, given a distribution
    :math:`f(x)` where :math:`x\\in[a, b]` then, given a random variable
    following a flat distribution *r*,

    .. math::
       F(x) - F(x_0) = \int_{x_0}^x f(x) dx = \int_0^r r dr = r

    where :math:`F(x)` is the primitive of :math:`f(x)`. This allows us to
    generate values following the distribution :math:`f(x)` given values from
    a flat distribution

    .. math::
       x = F^{-1}(r + F(x_0))

    In this class, the inverse process is performed. From a given set of values
    of a certain distribution, we build a method to generate numbers following
    a flat distribution.

    The class performs an interpolation to get the transformated values from
    an array holding the cumulative of the distribution. The function
    :func:`scipy.interpolate.interp1d` is used for this purpose.
    '''
    def __init__( self, points, values=None, kind='cubic' ):
        r'''
        Build the class from a given set of values following a certain
        distribution (the use of weights is allowed), or x and y values of
        a PDF. This last method is not recommended, since the precision
        relies on the dispersion of the values, sometimes concentrated around
        peaking regions which might not be well described by an interpolation.

        :param points: x-values of the distribution (PDF).
        :type points: numpy.ndarray
        :param values: weights or PDF values to use.
        :type values: numpy.ndarray
        :param kind: kind of interpolation to use. For more details see \
        :func:`scipy.interpolate.interp1d`.
        :type kind: str or int
        '''
        srt = points.argsort()

        points = points[srt]
        if values is None:
            c = np.linspace(1./len(points), 1., len(points))
        else:
            c  = np.cumsum(values[srt])
            c *= 1./c[-1]

        self._trans = interp1d(points, c,
                               copy=False,
                               kind=kind,
                               bounds_error=False,
                               fill_value=(0, 1)
        )

    def transform( self, values ):
        r'''
        Return the value of the transformation of the given values.

        :param values: values to transform.
        :type values: numpy.ndarray
        '''
        return self._trans(values)


def _ks_2samp_values( arr, weights = None ):
    r'''
    Calculate the values needed to perform the Kolmogorov-Smirnov test.

    :param arr: input sample.
    :type arr: numpy.ndarray
    :param weights: possible weights.
    :type weights: None or numpy.ndarray
    :returns: Sorted sample, stack with the cumulative distribution and
    sum of weights.
    :rtype: numpy.ndarray, numpy.ndarray, float
    '''
    weights = weights if weights is not None else np.ones(len(arr), dtype=float)

    ix  = np.argsort(arr)
    arr = arr[ix]
    weights = weights[ix]

    cs = np.cumsum(weights)

    sw = cs[-1]

    hs = np.hstack((0, cs/sw))

    return arr, hs, sw


def ks_2samp( a, b, wa = None, wb = None ):
    r'''
    Compute the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent
    samples are drawn from the same continuous distribution.
    Weights for each sample are accepted. If no weights are provided, then
    the function :func:`scipy.stats.ks_2samp` is called instead.

    :param a: first sample.
    :type a: numpy.ndarray
    :param b: second sample.
    :type b: numpy.ndarray
    :param wa: set of weights for "a". Same length as "a".
    :type wa: numpy.ndarray or None.
    :param wb: set of weights for "b". Same length as "b".
    :type wb: numpy.ndarray or None.
    :returns: Test statistic and two-tailed p-value.
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


def rv_random_sample( func, size = 10000, **kwargs ):
    r'''
    Create a random sample from the given rv_frozen object.
    This is usually used after creating a :class:`scipy.stats.rv_discrete`
    or :class:`scipy.stats.rv_continuous` class.

    :param func: function to use for the generation.
    :type func: :class:`scipy.stats.rv_frozen`
    :param size: size of the sample.
    :type size: int
    :param kwargs: any other argument to :class:`scipy.stats.rv_frozen.rvs`.
    :type kwargs: dict
    :returns: Generated sample.
    :rtype: numpy.ndarray
    '''
    args = np.array(func.args)

    if len(args.shape) == 1:
        size = (size,)
    else:
        size = (size, args.shape[1])

    return func.rvs(size=size, **kwargs)


# Tuple to hold the return values of the function "stat_values"
StatValues = namedtuple('StatValues', ('mean', 'var', 'std', 'var_mean', 'std_mean'))


def stat_values( arr, axis = None, weights = None ):
    r'''
    Calculate mean and variance and standard deviations of the sample and the
    mean from the given array.
    Weights are allowed.
    The definition of the aforementioned quantities are:

    - Mean:

    .. math::
       \\bar{x} = \\sum_{i=0}^{n - 1}{\\frac{x_i}{n}}

    - Weighted mean:

    .. math::
       \\bar{x}^w = \\frac{\\sum_{i=0}^{n - 1}{\omega_i x_i}}{\\sum_{i=0}^{n - 1}{\omega_i}}

    - Variance of the sample:

    .. math::
       \sigma_s = \sum_{i=0}^{n - 1}{\\frac{(x_i - \\bar{x})^2}{n - 1}}

    - Weighted variance of the sample:

    .. math::
       \sigma^w_s = \\frac{N'}{(N' - 1)}\\frac{\sum_{i=0}^{n - 1}{\omega_i(x_i - \\bar{x}^w)^2}}{\sum_{i=0}^{n - 1}{\omega_i}}

    where :math:`\omega_i` refers to the weights associated with the value
    :math:`x_i`, and in the last equation N' refers to the number of non-zero
    weights. The variance and standard deviations of the mean are then given by:

    - Standard deviation of the mean:

    .. math::
       s_\\bar{x} = \\sqrt{\\frac{\sigma_s}{n}}

    - Weighted standard deviation of the mean:

    .. math::
       s^w_\\bar{x} = \\sqrt{\\frac{\sigma^w_s}{N'}}

    :param arr: input array of data.
    :type arr: numpy.ndarray
    :param axis: axis or axes along which to calculate the values for "arr".
    :type axis: None or int or tuple(int)
    :param weights: array of weights associated to the values in "arr".
    :type weights: None or numpy.ndarray
    :returns: Mean, variance, standard deviation, variance of the mean and \
    standard deviation of the mean.
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    '''
    keepdims = (axis is not None)

    asum = lambda a: np.sum(a, axis=axis, keepdims=keepdims, dtype=float)

    if weights is None:

        mean = np.mean(arr, axis=axis, keepdims=keepdims)

        if keepdims:
            lgth = arr.shape[axis]
        else:
            lgth = arr.size

        var = asum((arr - mean)**2)/(lgth - 1.)

        var_mean = var/lgth

    else:
        sw = asum(weights)

        # We can not use "count_nonzero" ince it does not keep
        # the dimensions through "keepdims"
        nzw = asum(weights != 0)

        mean = asum(weights*arr)/sw
        var  = asum(weights*(arr - mean)**2)*nzw/(sw*(nzw - 1.))

        var_mean = var/nzw

    std = np.sqrt(var)

    std_mean = np.sqrt(var_mean)

    return StatValues(mean, var, std, var_mean, std_mean)
