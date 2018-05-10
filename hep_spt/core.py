'''
Some functions shared among modules.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
from functools import wraps

__all__ = []


def decorate( deco ):
    '''
    Function to wrap a decorator so it preserves the name and
    docstring of the original function.

    :param deco: raw decorator which was meant to be used.
    :type deco: function-like
    :returns: Decorator which preserves the name and docstring of the \
    original function.
    :rtype: function
    '''
    def _deco_wrapper( func ):
        '''
        This is the wrapper over the decorator.
        '''
        decorated_function = deco(func)

        @wraps(func)
        def _wrapper( *args, **kwargs ):
            '''
            Wrap the original function.
            '''
            return decorated_function(*args, **kwargs)

        return _wrapper

    return _deco_wrapper


@decorate
def taking_ndarray( func ):
    '''
    Decorator for functions which take :class:`numpy.ndarray` instances
    as arguments.
    The array is not copied in the process.
    '''
    def _wrapper( *args, **kwargs ):
        '''
        Wrap the original function.
        '''
        args = tuple(np.array(x, copy=False) for x in args)

        kwargs = {k: np.array(x, copy=False) for k, x in kwargs.items()}

        return func(*args, **kwargs)

    return _wrapper
