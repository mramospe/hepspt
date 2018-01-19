'''
Test functions for the "math_aux" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
import hepspt

# Python
import numpy as np


def test_gcd():
    '''
    Test the function to calculate the greatest common divisor of a set
    of numbers.
    '''
    assert hepspt.gcd(4, 6) == 2
    assert hepspt.gcd(4, 6, 8) == 2
    assert np.all(hepspt.gcd([4, 3], [6, 9]) == [2, 3])


def test_lcm():
    '''
    Test the function to calculate the least common multiple of a set
    of numbers.
    '''
    assert hepspt.lcm(11, 4) == 44
    assert hepspt.lcm(8, 4, 2) == 8
    assert np.all(hepspt.lcm([10, 121], [100, 242]) == [100, 242])


def test_power_2():
    '''
    Test the functions related wih evaluating and creating powers of 2.
    '''
    n = 15

    assert hepspt.is_power_2(n) == False

    nt = hepspt.next_power_2(n)

    assert hepspt.is_power_2(nt) == True
