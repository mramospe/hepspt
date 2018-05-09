'''
Auxiliar mathematical functions.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
from hep_spt.core import taking_ndarray

# Python
import numpy as np
from functools import reduce


__all__ = ['bit_length', 'gcd', 'ibinary_repr', 'is_power_2', 'lcm', 'next_power_2']


@taking_ndarray
def bit_length( arg ):
    '''
    Get the length of the binary representation of the given value(s).
    This function is equivalent to :func:`int.bit_length`, but can take arrays
    as an input.

    :param arg: array of values.
    :type arg: int or numpy.ndarray(int)
    :returns: length of the binary representation.
    :rtype: numpy.ndarray(int)
    '''
    if arg.ndim == 0:

        r = np.binary_repr(arg)

        if r != '0':
            return len(r)
        else:
            return 0

    coc = arg // 2
    rem = arg % 2

    lgth = np.logical_or(rem != 0, coc != 0).astype(int)

    idxs = coc != 0
    if len(idxs) != 0:
        lgth[idxs] += bit_length(coc[idxs])

    return lgth


def _gcd_array( a, b ):
    '''
    Auxiliar function to be used in :func:`gcd` to calculate
    the greatest common divisor of numbers stored in two :class:`numpy.ndarray`
    objects.

    :param a: first numbers.
    :type a: numpy.ndarray(int)
    :param b: second numbers.
    :type b: numpy.ndarray(int)
    :param args: any other numbers.
    :type args: tuple(int or numpy.ndarray(int))
    :returns: Output of one step in the calculation of the greatest common \
    divisor with arrays.
    :rtype: numpy.ndarray(int)
    '''
    idxs = b != 0
    if len(b) != 0:

        ai, bi = a[idxs], b[idxs]

        ai, bi = bi, ai % bi

        a[idxs] = _gcd_array(ai, bi)

    return a


@taking_ndarray
def gcd( a, b, *args ):
    '''
    Calculate the greatest common divisor of a set of numbers.

    :param a: first number(s).
    :type a: int or numpy.ndarray(int)
    :param b: second number(s).
    :type b: int or numpy.ndarray(int)
    :param args: any other numbers.
    :type args: tuple(int or numpy.ndarray(int))
    :returns: Greatest common divisor of a set of numbers.
    :rtype: int or numpy.ndarray(int)
    '''
    if len(args) == 0:
        if a.ndim == b.ndim == 0:
            while b:
                a, b = b, a % b
            return a
        else:
            return _gcd_array(a, b)
    else:
        return reduce(gcd, args + (a, b))


@taking_ndarray
def ibinary_repr( arg ):
    '''
    Get the binary representation of the given value(s).
    This function is equivalent to :func:`numpy.binary_repr`, but the returned
    value is an integer.

    :param arg: array of values.
    :type arg: int or numpy.ndarray(int)
    :returns: values in binary representation (as integers).
    :rtype: numpy.ndarray(int)
    '''
    if arg.ndim == 0:
        return int(np.binary_repr(arg))

    coc = arg // 2
    rem = arg % 2

    idxs = coc != 0
    if len(idxs) != 0:
        rem[idxs] += 10*ibinary_repr(coc[idxs])

    return rem


@taking_ndarray
def is_power_2( arg ):
    '''
    Determine whether the input number(s) is a power of 2 or not. Only
    works with positive numbers.

    :param arg: input number(s).
    :type arg: int or numpy.ndarray(int)
    :returns: Whether the input number(s) is(are) a power of 2 or not.
    :rtype: bool or numpy.ndarray(bool)
    '''
    return np.logical_and(arg > 0, ((arg & (arg - 1)) == 0))


@taking_ndarray
def lcm( a, b, *args ):
    '''
    Calculate the least common multiple of a set of numbers.

    :param a: first number(s).
    :type a: int or numpy.ndarray(int)
    :param b: second number(s).
    :type b: int or numpy.ndarray(int)
    :param args: any other numbers.
    :type args: tuple(int or numpy.ndarray(int))
    :returns: Least common multiple of a set of numbers.
    :rtype: int or numpy.ndarray(int)
    '''
    if len(args) == 0:
        return a*b//gcd(a, b)
    else:
        return reduce(lcm, args + (a, b))


@taking_ndarray
def next_power_2( arg ):
    '''
    Calculate the next number(s) greater than that(those) given and being a power(s) of 2.

    :param arg: input number(s).
    :type arg: int or numpy.ndarray(int)
    :returns: Next power of 2 to the given number.
    :rtype: int or numpy.ndarray(int)

    .. note: If the input number is a power of two, it will return the \
    same number.
    '''
    return 1 << bit_length(arg - 1)
