#!/usr/bin/env python
'''
Setup script for the hep_spt package
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import glob
import os
import textwrap

#
# Version of the package. Before a new release is made
# just the "version_info" must be changed. The options
# for the fourth tag are "dev", "alpha", "beta",
# "cand", "final" or "post".
#
version_info = (0, 0, 0, 'dev', 4)

tag = version_info[3]

if tag != 'final':
    if tag == 'alpha':
        frmt = '{}a{}'
    elif tag == 'beta':
        frmt = '{}b{}'
    elif tag == 'cand':
        frmt = '{}rc{}'
    elif tag == 'post':
        frmt = '{}.post{}'
    elif tag == 'dev':
        frmt = '{}.dev{}'
    else:
        raise ValueError('Unable to parse version information')

version = frmt.format('.'.join(map(str, version_info[:3])), version_info[4])


def create_version_file():
    '''
    Create the file version.py given the version of the package.
    '''
    version_file = open('hep_spt/version.py', 'wt')
    version_file.write(textwrap.dedent("""\
    '''
    Auto-generated module holding the version of the hep_spt package
    '''

    __version__ = "{}"
    __version_info__ = {}

    __all__ = ['__version__', '__version_info__']
    """.format(version, version_info)))
    version_file.close()


def install_requirements():
    '''
    Get the installation requirements from the "requirements.txt" file.
    '''
    reqs = []
    with open('requirements.txt') as f:
        for line  in f:
            li = line.strip()
            if not li.startswith('#'):
                reqs.append(li)
    return reqs


def configuration( parent_package = '', top_path = '' ):
    '''
    Function to do the configuration.
    '''
    from numpy.distutils.misc_util import Configuration
    config = Configuration(

        name = 'hep_spt',

        version = version,

        description = 'Provides statistical and plotting tools using general '\
        'python packages, focused to High Energy Physics.',

        # Read the long description from the README
        long_description = open('README.rst').read(),

        # Keywords to search for the package
        keywords = 'physics hep statistics plotting',

        # Requisites
        install_requires = install_requirements(),

        # Test requirements
        setup_requires = ['pytest-runner'],

        tests_require = ['pytest'],
    )

    config.add_subpackage('hep_spt')

    return config


if __name__ == '__main__':

    try:
        import numpy
    except:
        RuntimeError('Numpy not found. Please install it before setting up this package.')

    from numpy.distutils.core import setup
    setup(configuration=configuration)

    create_version_file()
