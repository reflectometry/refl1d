#!/usr/bin/env python

import os
import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('pak2010', parent_package, top_path)

    config.add_subpackage('amqp_map')
    config.add_subpackage('dream')
    config.add_subpackage('models')
    config.add_subpackage('park')
    config.add_subpackage('refl1d')

    config.add_data_dir('bin')
    config.add_data_dir('examples')
    config.add_data_dir('refltests')
    config.add_data_dir('tests')

    config.get_version(os.path.join('version.py'))  # sets config.version

    return config


if __name__ == '__main__':
    if len(sys.argv) == 1: sys.argv.append('install')
    setup(**configuration(top_path='').todict())
