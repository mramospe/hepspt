'''
Test functions for the "math_aux" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
import hepspt


def test_power_2():
    '''
    Test the functions related wih evaluating and creating powers of 2.
    '''
    n = 15

    assert hepspt.is_power_2(n) == False

    nt = hepspt.next_power_2(n)

    assert hepspt.is_power_2(nt) == True
