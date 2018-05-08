'''
Auxiliar mathematical functions.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
from hep_spt.core import decorate

# Python
import numpy as np
from functools import reduce


__all__ = ['gcd', 'is_power_2', 'lcm', 'next_power_2']


@decorate(np.vectorize)
def gcd( a, b, *args ):
    '''
    Calculate the greatest common divisor of a set of numbers.

    :param a: first number.
    :type a: int or numpy.ndarray(int)
    :param b: second number.
    :type b: int or numpy.ndarray(int)
    :param args: any other numbers.
    :type args: tuple(int or numpy.ndarray(int))
    :returns: Greatest common divisor of a set of numbers.
    :rtype: int or numpy.ndarray(int)
    '''
    if len(args) == 0:
        while b:
            a, b = b, a % b
        return a
    else:
        return reduce(gcd, args + (a, b))


@decorate(np.vectorize)
def is_power_2( n ):
    '''
    Determine whether the input number(s) is a power of 2 or not. Only
    works with positive numbers.

    :param n: input number(s).
    :type n: int or numpy.ndarray(int)
    :returns: Whether the input number(s) is(are) a power of 2 or not.
    :rtype: bool or numpy.ndarray(bool)
    '''
    return n > 0 and ((n & (n - 1)) == 0)


@decorate(np.vectorize)
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


@decorate(np.vectorize)
def next_power_2( n ):
    '''
    Calculate the next number(s) greater than that(those) given and being a power(s) of 2.

    :param n: input number(s).
    :type n: int or numpy.ndarray(int)
    :returns: Next power of 2 to the given number.
    :rtype: int or numpy.ndarray(int)

    .. note: If the input number is a power of two, it will return the \
    same number.
    '''
    return 1 << int(n - 1).bit_length()
