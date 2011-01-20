#!/usr/bin/env python

import os
import sys

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('park', parent_package, top_path)
    config.set_options(quiet=True) # silence debug/informational messages

    config.add_subpackage('amqp')
    config.add_data_dir(os.path.join('amqp', 'example'))
    config.add_data_dir('bin')
    config.add_subpackage('examples')
    config.add_subpackage('service')

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
