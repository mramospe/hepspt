r'''
Configuration file for the modules in the "hep_spt" package.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


def configuration( parent_package = '', top_path = '' ):
    r'''
    Function to do the configuration.
    '''
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Add data packages
    config.add_data_dir('data')
    config.add_data_dir('mpl')

    # Add subpackages
    config.add_subpackage('cpython')
    config.add_subpackage('stats')

    return config


if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(configuration=configuration)
