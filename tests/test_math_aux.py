'''
Test functions for the "math_aux" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import pytest

# Local
import hep_spt


def test_bit_length():
    '''
    Test for the "bit_length" function. It must be equivalent to int.bit_length.
    '''
    # Test the function taking scalars
    assert hep_spt.bit_length(0) == 0
    assert hep_spt.bit_length(1) == 1
    assert hep_spt.bit_length(2) == 2
    assert hep_spt.bit_length(3) == 2

    for i in np.random.randint(0, 300, 20):
        assert hep_spt.bit_length(i) == int(i).bit_length()

    # Test the function taking arrays
    assert np.all(hep_spt.bit_length([0, 1, 2, 3]) == [0, 1, 2, 2])

    values = np.random.randint(0, 300, 20)

    std_python = tuple(map(lambda i: int(i).bit_length(), values))

    assert np.all(hep_spt.bit_length(values) == std_python)

    values = np.array([(0, 1), (2, 3)])
    ref    = np.array([(0, 1), (2, 2)])

    assert np.all(hep_spt.bit_length(values) == ref)

    # Check the exceptions
    with pytest.raises(TypeError):
        hep_spt.bit_length(4.5)

    with pytest.raises(TypeError):
        hep_spt.bit_length(np.array([1.1, 3.2]))


def test_gcd():
    '''
    Test the function to calculate the greatest common divisor of a set
    of numbers.
    '''
    # Test single value behaviour
    assert hep_spt.gcd(4, 6) == 2
    assert hep_spt.gcd(4, 6, 8) == 2

    # Test numpy.vectorize behaviour
    assert np.all(hep_spt.gcd([4, 3], [6, 9]) == [2, 3])

    a_vals = np.array([(4, 3), (8, 5)])
    b_vals = np.array([(6, 9), (4, 10)])
    ref    = np.array([(2, 3), (4, 5)])

    assert np.all(hep_spt.gcd(a_vals, b_vals) == ref)

    # Check the exceptions
    with pytest.raises(TypeError):
        hep_spt.gcd(4.5, 3)

    with pytest.raises(TypeError):
        hep_spt.gcd(np.array([1, 3]), np.array([1]))

    with pytest.raises(TypeError):
        hep_spt.gcd(3, np.array([1, 3]))


def test_ibinary_repr():
    '''
    Test for the "ibinary_repr" function. This must be equivalent to
    numpy.binary_repr but returning integers instead of strings.
    '''
    # Test the function taking scalars
    assert hep_spt.ibinary_repr(0) == 0
    assert hep_spt.ibinary_repr(1) == 1
    assert hep_spt.ibinary_repr(2) == 10
    assert hep_spt.ibinary_repr(3) == 11

    for i in np.random.randint(0, 100, 20):
        assert str(hep_spt.ibinary_repr(i)) == np.binary_repr(i)

    # Test the function taking arrays
    assert np.all(hep_spt.ibinary_repr([0, 1, 2, 3]) == [0, 1, 10, 11])

    values = np.random.randint(0, 100, 20)

    std_numpy = tuple(map(np.binary_repr, values))

    assert np.all(hep_spt.ibinary_repr(values).astype(str) == std_numpy)

    values = np.array([(0, 1), (2, 3)])
    ref    = np.array([(0, 1), (10, 11)])

    assert np.all(hep_spt.ibinary_repr(values) == ref)

    # Check the exceptions
    with pytest.raises(TypeError):
        hep_spt.ibinary_repr(4.5)

    with pytest.raises(TypeError):
        hep_spt.ibinary_repr(np.array([1.1, 3.2]))


def test_lcm():
    '''
    Test the function to calculate the least common multiple of a set
    of numbers.
    '''
    # Test single value behaviour
    assert hep_spt.lcm(11, 4) == 44
    assert hep_spt.lcm(8, 4, 2) == 8

    # Test numpy.vectorize behaviour
    assert np.all(hep_spt.lcm([10, 121], [100, 242]) == [100, 242])


def test_is_power_2():
    '''
    Test the functions related wih evaluating and creating powers of 2.
    '''
    assert not hep_spt.is_power_2(15)
    assert hep_spt.is_power_2(8)
    assert np.all(hep_spt.is_power_2([2, 3]) == np.array([True, False]))


def test_next_power_2():
    '''
    Test for the function "next_power_2".
    '''
    assert hep_spt.next_power_2(2) == 2
    assert hep_spt.next_power_2(15) == 16
    assert np.all(hep_spt.next_power_2([2, 3]) == [2, 4])
