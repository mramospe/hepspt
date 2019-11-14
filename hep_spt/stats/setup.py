r'''
Configuration file for the modules in the "stats" subpackage.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


def configuration( parent_package = '', top_path = '' ):
    r'''
    Function to do the configuration.
    '''
    from numpy.distutils.misc_util import Configuration

    config = Configuration('stats', parent_package, top_path)

    return config


if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(configuration=configuration)
