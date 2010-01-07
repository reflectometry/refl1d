#!/usr/bin/env python

from setuptools import setup, find_packages
import fix_setuptools_chmod

dist = setup(
        name = 'mystic',
        version = '0.2',
        packages = find_packages(),
        install_requires = ['numpy'],
)

# End of file
