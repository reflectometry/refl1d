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

packages = ['refl1d', 'refl1d.view']

version = None
for line in open(os.path.join("refl1d", "__init__.py")):
    if "__version__" in line:
        version = line.split('"')[1]
#print("Version: "+version)

extra_compile_args =  {'msvc': ['/EHsc']}
extra_link_args =  {}

class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in extra_compile_args:
           for e in self.extensions:
               e.extra_compile_args = extra_compile_args[c]
        if c in extra_link_args:
            for e in self.extensions:
                e.extra_link_args = extra_link_args[c]
        build_ext.build_extensions(self)

# reflmodule extension
def reflmodule_config():
    if sys.platform == "darwin":
        # Python is not finding C++ headers on Mac unless the
        # minimum OSX version is bumped from the default 10.6 up
        # to 10.10.  Don't know if this is because of the mac
        # setup (older development libraries not installed) or
        # because of the anaconda build (targetted to 10.6) or
        # some combination.  Override by setting the deployment
        # target on the command line.  Curiously, Xcode can
        # target c++ code to 10.7 on the same machine.
        #os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.10')
        os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.13')

    S = ("reflmodule.cc", "methods.cc",
         "reflectivity.cc", "magnetic.cc",
         "contract_profile.cc",
         "convolve.c", "convolve_sampled.c",
        )

    Sdeps = ("erf.c", "methods.h", "rebin.h", "rebin2D.h", "reflcalc.h")
    sources = [os.path.join('refl1d', 'lib', f) for f in S]
    depends = [os.path.join('refl1d', 'lib', f) for f in Sdeps]
    return Extension('refl1d.reflmodule',
                     sources=sources,
                     depends=depends,
                     language="c++",
                     #py_limited_api="cp32",
                     )

# SCF extension
def SCFmodule_config():
    from numpy import get_include
    return Extension('refl1d.calc_g_zs_cex',
                     sources=[os.path.join('refl1d', 'lib', 'calc_g_zs_cex.c')],
                     include_dirs=[get_include()],
                     #language="c++",
                     #py_limited_api="cp32",
                     )

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
        'Programming Language :: Python',
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
    ext_modules=[reflmodule_config(), SCFmodule_config()],
    install_requires=['bumps>=0.7.16', 'numpy', 'scipy', 'matplotlib', 'periodictable'],
    extras_require={'full': ['wxpython']},
    cmdclass={'build_ext': build_ext_subclass},
    )

# End of file
