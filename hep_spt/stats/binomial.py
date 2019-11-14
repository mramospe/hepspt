r'''
Function involving the binomial distribution.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']

# Python
import numpy as np
from scipy.stats import beta, norm

# Local
from hep_spt.core import decorate, taking_ndarray
from hep_spt.stats.core import one_sigma


__all__ = ['clopper_pearson_int', 'clopper_pearson_unc',
           'wald_int', 'wald_unc',
           'wald_weighted_int', 'wald_weighted_unc',
           'wilson_int', 'wilson_unc']


@taking_ndarray
def clopper_pearson_int( k, N, cl = one_sigma ):
    r'''
    Return the frequentist Clopper-Pearson interval of having
    "k" events in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper bounds for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    p = k.astype(float)/N

    pcl = 0.5*(1. - cl)

    if k.ndim == 0:
        if k != 0:
            lw = beta(k, N - k + 1).ppf(pcl)
        else:
            lw = p

        if k != N:
            up = beta(k + 1, N - k).ppf(1. - pcl)
        else:
            up = p
    else:

        # Solve for low uncertainty
        lw = np.array(p)
        cd = (k != 0)

        if pcl.ndim == 0:
            lpcl = pcl
        else:
            lpcl = pcl[cd]

        lk, lN = k[cd], N[cd]

        lw[cd] = beta(lk, lN - lk + 1).ppf(lpcl)

        # Solve for upper uncertainty
        up = np.array(p)
        cd = (k != N)

        if pcl.ndim == 0:
            upcl = pcl
        else:
            upcl = pcl[cd]

        uk, uN = k[cd], N[cd]

        up[cd] = beta(uk + 1, uN- uk).ppf(1. - upcl)

    return lw, up


@taking_ndarray
def clopper_pearson_unc( k, N, cl = one_sigma ):
    r'''
    Return the frequentist Clopper-Pearson uncertainties of having
    "k" events in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper uncertainties for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    p = k*1./N
    p_l, p_u = clopper_pearson_int(k, N, cl)
    return p - p_l, p_u - p


@taking_ndarray
def wald_int( k, N, cl = one_sigma ):
    r'''
    Calculate the symmetric Wald interval of having "k" elements
    in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper bounds for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    p = k*1./N
    s = wald_unc(k, N, cl)
    return p - s, p + s


@taking_ndarray
def wald_unc( k, N, cl = one_sigma ):
    r'''
    Calculate the symmetric Wald uncertainty of having "k" elements
    in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: symmetric uncertainty.
    :rtype: float or numpy.ndarray(float)
    '''
    z = norm.isf((1. - cl)/2.)
    p = k*1./N
    return z*np.sqrt(p*(1. - p)/N)


@taking_ndarray
def wald_weighted_int( k, N, cl = one_sigma ):
    r'''
    Calculate the symmetric Wald interval for a weighted sample,
    where "k" is the array of weights in the survival sample
    and "N" in the main sample.

    :param k: passed weights.
    :type k: numpy.ndarray(float)
    :param N: total weights.
    :type N: numpy.ndarray(float)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper bounds for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    sw_k = np.sum(k, axis=None)
    sw_N = np.sum(N, axis=None)
    p = sw_k*1./sw_N
    s = wald_weighted_unc(k, N, cl)
    return p - s, p + s


@taking_ndarray
def wald_weighted_unc( k, N, cl = one_sigma ):
    r'''
    Calculate the symmetric Wald uncertainty for a weighted sample,
    where "k" is the array of weights in the survival sample
    and "N" in the main sample.

    :param k: passed weights.
    :type k: numpy.ndarray(float)
    :param N: total weights.
    :type N: numpy.ndarray(float)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: symmetric uncertainty.
    :rtype: float or numpy.ndarray(float)
    '''
    z = norm.isf((1. - cl)/2.)

    sN = np.sum(N, axis=None)
    W1 = np.sum(k, axis=None)
    W2 = sN - W1

    vw1 = np.sum(k*k, axis=None)
    vw2 = np.sum(N*N, axis=None) - vw1

    return z*np.sqrt((W1**2*vw2 + W2**2*vw1)/sN**4)


@taking_ndarray
def wilson_int( k, N, cl = one_sigma ):
    r'''
    Calculate the Wilson interval of having "k" elements in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper bounds for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    iN = 1./N
    p  = k*iN
    z  = norm.isf((1. - cl)/2.)
    z2 = z*z

    c  = p + z2*0.5*iN
    s  = z*iN*np.sqrt(p*(1. - p)*N + 0.25*z2)
    d  = 1. + z2*iN

    p_l = (c - s)/d
    p_u = (c + s)/d

    zero = np.isclose(p, 0.)
    one  = np.isclose(p, 1.)

    if p_l.ndim != 0:
        p_l[zero] = 0.
    elif zero:
        p_l = 0.

    if p_u.ndim != 0:
        p_u[one] = 1.
    elif one:
        p_u = 1.

    return p_l, p_u


@taking_ndarray
def wilson_unc( k, N, cl = one_sigma ):
    r'''
    Calculate the Wilson uncertainties of having "k" elements in "N".

    :param k: passed events.
    :type k: int or numpy.ndarray(int)
    :param N: total number of events.
    :type N: int or numpy.ndarray(int)
    :param cl: confidence level.
    :type cl: float or numpy.ndarray(float)
    :returns: Lower and upper uncertainties for the probability.
    :rtype: float or numpy.ndarray(float)
    '''
    p = k*1./N
    p_l, p_u = wilson_int(k, N, cl)
    return p - p_l, p_u - p
