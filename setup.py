#!/usr/bin/env python
'''
Setup script for the hep_spt package
'''

__author__ = 'Miguel Ramos Pernas'
__email__ = 'miguel.ramos.pernas@cern.ch'


# Python


import glob
import os
import setuptools
import subprocess
import sys
import textwrap


class CheckFormatCommand(setuptools.Command):
    '''
    Check the format of the files in the given directory. This script takes only
    one argument, the directory to process. A recursive look-up will be done to
    look for python files in the sub-directories and determine whether the files
    have the correct format.
    '''
    description = 'check the format of the files of a certain type in a given directory'

    user_options = [
        ('directory=', 'd', 'directory to process'),
        ('file-type=', 't', 'file type (python|all)'),
    ]

    def initialize_options(self):
        '''
        Running at the begining of the configuration.
        '''
        self.directory = None
        self.file_type = None

    def finalize_options(self):
        '''
        Running at the end of the configuration.
        '''
        if self.directory is None:
            raise Exception('Parameter --directory is missing')
        if not os.path.isdir(self.directory):
            raise Exception('Not a directory {}'.format(self.directory))
        if self.file_type is None:
            raise Exception('Parameter --file-type is missing')
        if self.file_type not in ('python', 'all'):
            raise Exception('File type must be either "python" or "all"')

    def run(self):
        '''
        Execution of the command action.
        '''
        matched_files = []
        for root, _, files in os.walk(self.directory):
            for f in files:
                if self.file_type == 'python' and not f.endswith('.py'):
                    continue
                matched_files.append(os.path.join(root, f))

        process = subprocess.Popen(['autopep8', '--diff'] + matched_files,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode < 0:
            raise RuntimeError('Call to autopep8 exited with error {}\nMessage:\n{}'.format(
                abs(returncode), stderr))

        if len(stdout):
            raise RuntimeError(
                'Found differences for files in directory "{}" with file type "{}"'.format(self.directory, self.file_type))


#
# Version of the package. Before a new release is made
# just the "version_info" must be changed. The options
# for the fourth tag are "dev", "alpha", "beta",
# "cand", "final" or "post".
#
version_info = (0, 0, 0, 'dev', 5)

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

    VERSION = "{}"
    VERSION_INFO = {}

    __all__ = ['VERSION', 'VERSION_INFO']
    """.format(version, version_info)))
    version_file.close()


def install_requirements():
    '''
    Get the installation requirements from the "requirements.txt" file.
    '''
    reqs = []
    with open('requirements.txt') as f:
        for line in f:
            li = line.strip()
            if not li.startswith('#'):
                reqs.append(li)
    return reqs


def configuration(parent_package='', top_path=''):
    '''
    Function to do the configuration.
    '''
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(quiet=True)

    config.add_subpackage('hep_spt')
    config.add_data_files(('hep_spt', 'LICENSE.txt'))

    return config


def setup_package():
    '''
    Set up the package.
    '''
    try:
        import numpy
    except:
        RuntimeError(
            'Numpy not found. Please install it before setting up this package.')

    metadata = dict(
        name='hep_spt',
        version=version,
        configuration=configuration,
        cmdclass={'check_format': CheckFormatCommand},
        description='Provides statistical and plotting tools using general '
        'python packages, focused to High Energy Physics.',
        long_description=open('README.rst').read(),
        keywords='physics hep statistics plotting',
        install_requires=install_requirements(),
    )

    create_version_file()

    from numpy.distutils.core import setup
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
