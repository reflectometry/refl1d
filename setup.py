#!/usr/bin/env python
import sys
import os

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

import refl1d

packages = find_packages(exclude=['amqp_map','models','park'])
packages += find_packages('dream')

if len(sys.argv) == 1:
    sys.argv.append('install')

# reflmodule extension
def reflmodule_config():
    sources = [os.path.join('refl1d','lib',f)
               for f in ("reflmodule.cc","methods.cc","reflectivity.cc",
                         "magnetic.cc","resolution.c","contract_profile.cc")]
    module = Extension('refl1d.reflmodule', sources=sources)
    return module

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
        package_dir = { 'dream': 'dream/dream' },
        packages = packages,
        package_data = {
            #'periodictable' : ['xsf/*.nff', 'xsf/f0_WaasKirf.dat', 'xsf/read.me'],
        },
        scripts = ['bin/reflworkerd'],
        ext_modules = [reflmodule_config()],
        #data_files = periodictable.data_files(),
        install_requires = ['numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'httplib2'],
)

# End of file

