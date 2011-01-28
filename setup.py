#!/usr/bin/env python

import os
import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('refl1d', parent_package, top_path)
    config.set_options(quiet=True) # silence debug/informational messages

    # Add subpackages (top level name spaces) and data directories.
    # Note that subpackages may have their own setup.py to drill down further.
    # Note that 'dream' is not a subpackage in our setup (no __init__.py as
    # this name may already be used), so we define our dream substructure here.
    config.add_subpackage('amqp_map')
    config.add_data_dir('bin')
    config.add_data_dir(os.path.join('doc', 'examples'))
    config.add_data_dir('examples')
    config.add_subpackage('dream.dream')
    config.add_data_dir(os.path.join('dream', 'examples'))
    config.add_subpackage('models')
    config.add_subpackage('park')
    config.add_subpackage('refl1d')
    config.add_data_dir('refltests')
    config.add_data_dir('tests')

    config.add_data_files('refl1d.ico')
    config.add_data_files('refl1d.iss')
    config.add_data_files('*.txt')
    config.add_data_files('test.py')

    for line in open('refl1d/__init__.py').readlines():
        if (line.startswith('__version__')):
            exec(line.strip())
            config.version = __version__
            break

    return config


if __name__ == '__main__':
    if len(sys.argv) == 1: sys.argv.append('install')
    setup(**configuration(top_path='').todict())
