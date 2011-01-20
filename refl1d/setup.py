#!/usr/bin/env python

import os.path

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None):
    config = Configuration('refl1d', parent_package, top_path)
    config.set_options(quiet=True) # silence debug/informational messages

    # Create the reflectometry library extension.
    srcpath = os.path.join(config.package_path, 'lib')
    sources = [os.path.join(srcpath,f)
               for f in ("reflmodule.cc","methods.cc","reflectivity.cc",
                         "magnetic.cc","resolution.c","contract_profile.cc")]

    config.add_extension('reflmodule',
                         sources=sources,
                         #f2py_options=['--no-wrap-functions'],
                         #define_macros = define_macros,
                         )

    # Add subpackages.
    # Note that for convenience, we define subpackages of mystic here instead
    # of using separate setup.py files in the subdirectories.
    config.add_subpackage('mystic')
    config.add_subpackage('mystic.examples')
    config.add_subpackage('mystic.optimizer')
    config.add_subpackage('mystic.optimizer.newton')

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
