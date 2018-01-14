'''
Test functions for the "math_aux" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


def test_power_2():
    '''
    Test the functions related wih evaluating and creating powers of 2.
    '''
    n = 15

    assert math_aux.is_power_2(n) == False

    nt = math_aux.next_power_2(n)

    assert math_aux.is_power_2(nt) == True
