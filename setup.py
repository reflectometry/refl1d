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

#from distutils.core import setup, Extension
from setuptools import setup, Extension

packages = ['refl1d', 'refl1d.view']

version = None
for line in open(os.path.join("refl1d", "__init__.py")):
    if "__version__" in line:
        version = line.split('"')[1]
#print("Version: "+version)


# reflmodule extension
def reflmodule_config():
    extra_compile_args = None
    if sys.platform == "darwin":
        os.environ['CXX'] = 'c++'
    if os.name == 'nt':
        # Note: This could be done by extending the build_ext class instead
        # http://stackoverflow.com/a/5192738
        import distutils.ccompiler
        compiler = distutils.ccompiler.get_default_compiler()
        for arg in sys.argv:
            if arg.startswith('--compiler='):
                _, compiler = arg.split('=')
        #if compiler == 'msvc':
        #    extra_compile_args = ['/EHsc']

    S = ("reflmodule.cc", "methods.cc",
         "reflectivity.cc", "magnetic.cc",
         "contract_profile.cc",
         "convolve.c", "convolve_sampled.c",
        )

    Sdeps = ("erf.c", "methods.h", "rebin.h", "rebin2D.h", "reflcalc.h")
    sources = [os.path.join('refl1d', 'lib', f) for f in S]
    depends = [os.path.join('refl1d', 'lib', f) for f in Sdeps]
    module = Extension('refl1d.reflmodule',
                       sources=sources, depends=depends,
                       extra_compile_args=extra_compile_args,
                      )
    return module

# SCF extension
def SCFmodule_config():
    from numpy import get_include
    return Extension('refl1d.calc_g_zs_cex',
                     sources=[os.path.join('refl1d', 'lib', 'calc_g_zs_cex.c')],
                     include_dirs=[get_include()])

#TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')

#sys.dont_write_bytecode = False
dist = setup(
    name='refl1d',
    version=version,
    author='Paul Kienzle',
    author_email='pkienzle@nist.gov',
    url='http://www.reflectometry.org/danse/model1d.html',
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
    ext_modules=[reflmodule_config(), SCFmodule_config()],
    install_requires=['bumps>=0.7.5', 'periodictable'],
    )

# End of file
