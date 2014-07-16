#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

# Use our own nose-based test harness
if sys.argv[1] == 'test':
    from subprocess import call
    sys.exit(call([sys.executable, 'test.py']+sys.argv[2:]))

sys.dont_write_bytecode = True

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

# Grab the openmp extension builder from the bumps package
try:
    import bumps
except:
    from os.path import dirname, abspath, join as joinpath
    bumps_path = joinpath(dirname(dirname(abspath(__file__))),'bumps')
    sys.path.insert(0, bumps_path)
from bumps.openmp_ext import openmp_ext

import refl1d


packages = find_packages()

# reflmodule extension
def reflmodule_config():
    sources = [os.path.join('refl1d','lib',f)
               for f in ("reflmodule.cc","methods.cc","reflectivity.cc",
                         "magnetic.cc","contract_profile.cc")]
    module = Extension('refl1d.reflmodule', sources=sources)
    return module

#TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')

sys.dont_write_bytecode = False
dist = setup(
        name = 'refl1d',
        version = refl1d.__version__,
        author='Paul Kienzle',
        author_email='pkienzle@nist.gov',
        url='http://www.reflectometry.org/danse/model1d.html',
        description='1-D reflectometry modelling',
        long_description=open('README.txt').read(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: Public Domain',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        packages = packages,
        #package_data = gui_resources.package_data(),
        scripts = ['bin/refl1d_cli.py','bin/refl1d_gui.py'],
        ext_modules = [reflmodule_config()],
        requires = [], # ['bumps>=0.9'],
        cmdclass = {'build_ext': openmp_ext(default=False)},
        )

# End of file
