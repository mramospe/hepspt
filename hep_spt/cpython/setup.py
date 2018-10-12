'''
Configuration file for the C-API extension.
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'

# Python
import glob


def configuration( parent_package = '', top_path = '' ):
    '''
    Function to do the configuration.
    '''
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cpython', parent_package, top_path)

    headers = glob.glob('*.h')

    # Add CPYTHON extension
    from numpy.distutils.misc_util import get_info

    math_aux_cpy_src = ['cpython/math_aux_cpy.c']
    config.add_extension('math_aux_cpy',
                         sources=math_aux_cpy_src,
                         depends=cpython_h + math_aux_cpy_src,
                         extra_info=get_info('npymath'),
    )

    return config


if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(configuration=configuration)
