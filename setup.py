#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

# Use our own nose-based test harness
if sys.argv[1] == 'test':
    from subprocess import call
    sys.exit(call([sys.executable, 'test.py']+sys.argv[2:]))

#sys.dont_write_bytecode = True

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

packages = ['refl1d', 'refl1d.view', 'refl1d.lib']

version = None
for line in open(os.path.join("refl1d", "__init__.py")):
    if "__version__" in line:
        version = line.split('"')[1]
#print("Version: "+version)


#TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')

#sys.dont_write_bytecode = False
dist = setup(
    name='refl1d',
    version=version,
    author='Paul Kienzle',
    author_email='pkienzle@nist.gov',
    url='http://github.com/reflectometry/refl1d',
    description='1-D reflectometry modeling',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        ],
    packages=packages,
    #package_data=gui_resources.package_data(),
    scripts=['bin/refl1d_cli.py', 'bin/refl1d_gui.py'],
    entry_points={
        'console_scripts': ['refl1d=refl1d.main:cli'],
        'gui_scripts': ['refl1d_gui=refl1d.main:gui']
    },
    install_requires=['bumps>=0.7.16', 'numpy', 'scipy', 'matplotlib', 'periodictable', 'numba'],
    extras_require={'full': ['wxpython', 'ipython']},
    )

# End of file
