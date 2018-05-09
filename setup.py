#!/usr/bin/env python
'''
Setup script for the hep_spt package
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os
from setuptools import setup, find_packages, Extension

#
# Version of the package. Before a new release is made
# just the "version_info" must be changed. The options
# for the fourth tag are "dev", "alpha", "beta",
# "cand", "final" or "post".
#
version_info = (0, 0, 0, 'dev', 2)

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

# To determine the CPython modules available on the given directory
def cpython_module( directory ):

    extensions = []
    for path, _, fnames in os.walk(directory):
        for f in filter(lambda s: s.endswith('.c'), fnames):

            full_path = os.path.join(path, f)

            ext = Extension(full_path[:-2].replace('/', '.'), [full_path])

            extensions.append(ext)

    return extensions

# Setup function
setup(

    name = 'hep_spt',

    version = version,

    description = 'Provides statistical and plotting tools using general '\
    'python packages, focused to High Energy Physics.',

    # Read the long description from the README
    long_description = open('README.rst').read(),

    # Keywords to search for the package
    keywords = 'physics hep statistics plotting',

    # Find all the packages in this directory
    packages = find_packages(),

    # Data files
    package_data = {'hep_spt': ['data/*', 'mpl/*']},

    # C-API source
    ext_modules = cpython_module('hep_spt/cpython'),

    # Requisites
    install_requires = ['matplotlib', 'numpy', 'pytest', 'scipy'],

    # Test requirements
    setup_requires = ['pytest-runner'],

    tests_require = ['pytest'],
    )


# Create a module with the versions
version_file = open('hep_spt/version.py', 'wt')
version_file.write("""\
'''
Auto-generated module holding the version of the hep_spt package
'''

__version__ = "{}"
__version_info__ = {}

__all__ = ['__version__', '__version_info__']
""".format(version, version_info))
version_file.close()
