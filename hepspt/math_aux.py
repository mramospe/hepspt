'''
Auxiliar mathematical functions.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


def is_power_2( n ):
    '''
    :param n: input number.
    :type n: int
    :returns: whether the input number is a power of 2 or not.
    :rtype: bool
    '''
    return n > 0 and ((n & (n - 1)) == 0)


def next_power_2( n ):
    '''
    :param n: input number.
    :type n: int
    :returns: next power of 2 to the given number.
    :rtype: int

    .. note: If the input number is a power of two, it will return
    the same number.
    '''
    return 1 << (n - 1).bit_length()
