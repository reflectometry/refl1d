#!/usr/bin/env python

import os.path

from numpy.distutils.misc_util import Configuration


def configuration(parent_package='',
                  top_path=None
                  ):
    config = Configuration('model', parent_package, top_path)

    config.add_data_files('README.txt')

    # reflectometry library sources
    srcpath = os.path.join(config.package_path, 'lib')
    sources = [ os.path.join(srcpath,'src',pattern)
                for pattern in ['*.c','*.cc'] ]
    sources += [os.path.join(srcpath,'reflmodule.cc'),
                os.path.join(srcpath,'methods.cc')]

    config.add_extension('reflmodule',
                         sources=sources,
                         #f2py_options=['--no-wrap-functions'],
                         #define_macros = define_macros,
                         )

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
