#!/usr/bin/env python

# Copyright (C) 2006-2011, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Author: James Krycka, Paul Kienzle

"""
This script builds the Refl1D application and documentation from source and
runs unit tests and doc tests.  It supports building on Windows and Linux.

Usually, you downloaded this script into a top-level directory (the root)
and run it from there which downloads the files from the application
repository into a subdirectory (the package directory).  For example if
test1 is the root directory, we might have:
  E:/work/test1/master_builder.py
               /refl1d/master_builder.py
               /refl1d/...

Alternatively, you can download the whole application repository and run
this script from the application's package directory where it is stored.
The script determines whether it is executing from the root or the package
directory and makes the necessary adjustments.  In this case, the root
directory is defined as one-level-up and the repository is not downloaded
(as it is assumed to be fully present).  In the example below test1 is the
implicit root (i.e. top-level) directory.
  E:/work/test1/refl1d/master_builder.py
               /refl1d/...

Getting "git pull" support on Windows requires some work.  You first need to
install Git Bash from msysgit (github.com has a link).  To build, start
gitbash, and set up your ssh keys using::

    $ eval `ssh-agent -s`
    $ ssh-add

If you want to use PuTTY instead, you must convert your private key to a
putty key and drop it on your desktop.  Click the private key icon to start
pagent with your key.  Uncomment os.environ['GIT_SSH'] below, and set it to
the location of plink.exe on your system.
"""
from __future__ import print_function

import os
import sys
import shutil
import subprocess
from distutils.util import get_platform

# python 3 uses input rather than raw_input
try:
    input = raw_input
except Exception:
    pass

GIT = "git"
MAKE = "make"

# Windows commands to run utilities
PYTHON = sys.executable
if os.name == "nt":
    SYSBIN = r"C:\Program Files (x86)"
    if not os.path.exists(SYSBIN):
        SYSBIN = r"C:\Program Files"
    GIT = SYSBIN+r"\Git\mingw64\bin\git.exe"
    INNO = SYSBIN+r"\Inno Setup 5\ISCC.exe"  # command line operation

    if not os.path.exists(GIT):
        print("missing git: "+GIT+" --- source will not be updated",
              file=sys.stderr)
    if not os.path.exists(INNO):
        print("missing inno setup: "+INNO+" --- installer will not be built",
              file=sys.stderr)

    # Put PYTHON in the environment and add the python directory and its
    # corresponding script directory (for nose, sphinx, pip, etc) to the path.
    PYTHONDIR = os.path.dirname(os.path.abspath(PYTHON))
    SCRIPTDIR = os.path.join(PYTHONDIR, 'Scripts')
    #LATEXDIR = 'C:/Program Files/MikTeX 2.9/miktex/bin/x64'
    LATEXDIR = 'z:/miktex/miktex/bin'  # Portable miktex distribution
    os.environ['PATH'] = ";".join((PYTHONDIR, SCRIPTDIR, LATEXDIR, os.environ['PATH']))
    os.environ['PYTHON'] = "/".join(PYTHON.split("\\"))
    #os.environ['GIT_SSH'] = r"C:\Program Files (x86)\PuTTY\plink.exe"
    MAKE = r"C:\mingw\bin\mingw32-make"
    #MAKE = r"make.bat"
else:
    # Support for wx in virtualenv on mac
    if hasattr(sys, "real_prefix"):
        PYTHON = os.path.join(sys.real_prefix, 'bin', 'python')
        os.environ['PYTHON'] = PYTHON
        os.environ['PYTHONHOME'] = sys.prefix

# Name of the package
PKG_NAME = "refl1d"
# Name of the application we're building
APP_NAME = "Refl1D"


# Required versions of Python packages and utilities to build the application.
MIN_PYTHON = "2.6"
MAX_PYTHON = "3.0"
MIN_MATPLOTLIB = "1.0.0"
MIN_NUMPY = "1.3.0"
MIN_SCIPY = "0.7.0"
MIN_WXPYTHON = "2.8.10.1"
MIN_SETUPTOOLS = "0.6c9"
MIN_GCC = "3.4.4"
MIN_PYPARSING = "1.5.2"
MIN_PERIODICTABLE = "1.3"
# Required versions of Python packages to run tests.
MIN_NOSE = "0.11"
# Required versions of Python packages and utilities to build documentation.
MIN_SPHINX = "1.0.3"
MIN_DOCUTILS = "0.5"
MIN_PYGMENTS = "1.0"
MIN_JINJA2 = "2.5.2"
#MIN_MATHJAX = "1.0.1"
# Required versions of Python packages and utilities to build Windows frozen
# image and Windows installer.
MIN_PY2EXE = "0.6.9"
MIN_INNO = "5.3.10"

# Determine the full directory paths of the top-level, source, and installation
# directories based on the directory where the script is running.  Here the
# top-level directory refers to the parent directory of the package.
RUN_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
head, tail = os.path.split(RUN_DIR)
if tail == PKG_NAME:
    TOP_DIR = head
else:
    TOP_DIR = RUN_DIR

# Set the python path
PLATFORM = '.%s-%s'%(get_platform(), sys.version[:3])
BUMPS_DIR = os.path.join(TOP_DIR, "bumps")
REFL_DIR = os.path.join(TOP_DIR, "refl1d")
PERIODICTABLE_DIR = os.path.join(TOP_DIR, "periodictable")
BUMPS_LIB = os.path.join(BUMPS_DIR, 'build', 'lib'+PLATFORM)
REFL_LIB = os.path.join(REFL_DIR, 'build', 'lib'+PLATFORM)
paths = [BUMPS_LIB, REFL_LIB, PERIODICTABLE_DIR]
sys.path = paths + sys.path
os.environ['PYTHONPATH'] = os.pathsep.join(paths)

BUMPS_NEW = '"%s" clone https://github.com/bumps/bumps.git'%GIT
REFL_NEW = '"%s" clone https://github.com/reflectometry/refl1d.git'%GIT
PERIODICTABLE_NEW = '"%s" clone https://github.com/pkienzle/periodictable.git'%GIT
REPO_UPDATE = '"%s" pull'%GIT

# Note: can't put target files in the dist directory since all files in dist
# are being collected and stored in the bundle.
INSTALLER_DIR = REFL_DIR


def get_version():
    "Determine package version"
    global PKG_VERSION
    # Get the version string of the application for use later.
    # This has to be done after we have checked out the repository.
    for line in open(os.path.join(REFL_DIR, PKG_NAME, '__init__.py')).readlines():
        if line.startswith('__version__'):
            PKG_VERSION = line.split('=')[1].strip()[1:-1]
            break
    else:
        raise RuntimeError("Could not find package version")

#==============================================================================

def checkout_code():
    "download or update source code"
    print("Checking out application code from the repository ...\n")

    if not os.path.exists(GIT):
        print("missing git --- source not checked out")
        return

    if RUN_DIR == TOP_DIR:
        exec_cmd(PERIODICTABLE_NEW, cwd=TOP_DIR)
        exec_cmd(BUMPS_NEW, cwd=TOP_DIR)
        exec_cmd(REFL_NEW, cwd=TOP_DIR)
    else:
        exec_cmd(REPO_UPDATE, cwd=PERIODICTABLE_DIR)
        exec_cmd(REPO_UPDATE, cwd=BUMPS_DIR)
        exec_cmd(REPO_UPDATE, cwd=REFL_DIR)

    get_version()  # reset version number in case it was updated remotely


def create_archive(version=None):
    "create source distribution"
    if version is None:
        version = PKG_VERSION

    # Create zip and tar archives of the source code and a manifest file
    # containing the names of all files.
    print("Creating an archive of the source code ...\n")

    try:
        # Create zip and tar archives in the dist subdirectory.
        exec_cmd("%s setup.py sdist --formats=zip,gztar --dist-dir=%s"
                 %(PYTHON, INSTALLER_DIR), cwd=REFL_DIR)
        exec_cmd("%s setup.py sdist --formats=zip,gztar --dist-dir=%s"
                 %(PYTHON, INSTALLER_DIR), cwd=BUMPS_DIR)
    except Exception:
        print("*** Failed to create source archive ***")


def build_package():
    "build and install the package"
    print("Building the %s package ...\n" %(PKG_NAME,))

    exec_cmd("%s setup.py build" %(PYTHON,), cwd=BUMPS_DIR)
    exec_cmd("%s setup.py build" %(PYTHON,), cwd=REFL_DIR)


def build_documentation():
    "build the documentation"
    print("Running the Sphinx utility to build documentation ...\n")
    doc_dir = os.path.join(REFL_DIR, "doc")
    html_dir = os.path.join(doc_dir, "_build", "html")
    latex_dir = os.path.join(doc_dir, "_build", "latex")
    pdf = os.path.join(latex_dir, APP_NAME+".pdf")

    # Delete any left over files from a previous build.
    # Create documentation in HTML and PDF format.
    sphinx_cmd = '"%s" -m sphinx.__init__ -b %%s -d _build/doctrees -D latex_paper_size=letter . %%s'%PYTHON
    exec_cmd(sphinx_cmd%("html", html_dir), cwd=doc_dir)
    if True:  # build pdf as well
        exec_cmd(sphinx_cmd%("latex", latex_dir), cwd=doc_dir)
        exec_cmd("pdflatex %s.tex"%APP_NAME, cwd=latex_dir)
        exec_cmd("pdflatex %s.tex"%APP_NAME, cwd=latex_dir)
        exec_cmd("pdflatex %s.tex"%APP_NAME, cwd=latex_dir)
        exec_cmd("makeindex -s python.ist %s.idx"%APP_NAME, cwd=latex_dir)
        exec_cmd("pdflatex %s.tex"%APP_NAME, cwd=latex_dir)
        exec_cmd("pdflatex %s.tex"%APP_NAME, cwd=latex_dir)
        # Copy PDF to the html directory where the html can find it.
        if os.path.isfile(pdf):
            shutil.copy(pdf, html_dir)


def create_binary():
    "create the standalone executable (windows or mac, depending)"
    if sys.platform == 'darwin':
        print("Using py2app to create a Mac OS X app...\n")
        exec_cmd("%s setup_py2app.py"%PYTHON, cwd=REFL_DIR)

    elif os.name == 'nt':
        print("Using py2exe to create a Win32 executable ...\n")
        exec_cmd("%s setup_py2exe.py"%PYTHON, cwd=REFL_DIR)

    else:
        print("No binary build for %s %s\n" % (sys.platform, os.name))


def create_installer(version=None):
    if not version:
        version = PKG_VERSION
    if not os.path.exists(INSTALLER_DIR):
        os.mkdir(INSTALLER_DIR)
    if sys.platform == 'darwin':
        _create_osx_installer(version)
    elif os.name == 'nt':
        _create_windows_installer(version)
    else:
        print("No installer for %r"%sys.platform)

def _create_osx_installer(version):
    "create mac dmg"
    print("Running command to create the dmg")
    exec_cmd([PYTHON, "extra/build_dmg.py", APP_NAME, PKG_VERSION],
             cwd=REFL_DIR)

def _create_windows_installer(version):
    "create the windows installer"
    if not os.path.exists(INNO):
        print("missing INNO --- no installer")
    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller for
    # the application.
    print("Running Inno Setup Compiler to create Win32 "
          "installer/uninstaller ...\n")

    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller.
    # Override the output specification in <PKG_NAME>.iss to put the executable
    # and the manifest file in the top-level directory.
    arch = "x64" if sys.maxsize > 2**32 else "x86"
    exec_cmd([INNO, "/Q", "/O"+INSTALLER_DIR,
              "/DVERSION="+version, "/DARCH="+arch, PKG_NAME+".iss"],
             cwd=REFL_DIR)


def run_tests():
    "run the test suite"
    # Run unittests and doctests using a test script.
    # Running from a test script allows customization of the system path.
    print("Running periodictable tests ...\n")
    exec_cmd("%s test.py"%PYTHON, cwd=PERIODICTABLE_DIR)
    print("Running bumps tests ...\n")
    exec_cmd("%s test.py"%PYTHON, cwd=BUMPS_DIR)
    print("Running refl tests ...\n")
    exec_cmd("%s test.py"%PYTHON, cwd=REFL_DIR)

def check_dependencies():
    "check that required packages are installed"

    import platform
    from pkg_resources import parse_version as PV

    # ------------------------------------------------------
    python_ver = platform.python_version()
    print("Using Python %s"%python_ver)
    print("")
    if PV(python_ver) < PV(MIN_PYTHON) or PV(python_ver) >= PV(MAX_PYTHON):
        print("ERROR - build requires Python >= %s, but < %s"
              %(MIN_PYTHON, MAX_PYTHON))
        sys.exit()

    req_pkg = {}

    # ------------------------------------------------------
    try:
        from matplotlib import __version__ as mpl_ver
    except Exception:
        mpl_ver = "0"
    finally:
        req_pkg["matplotlib"] = (mpl_ver, MIN_MATPLOTLIB)

    # ------------------------------------------------------
    try:
        from numpy import __version__ as numpy_ver
    except Exception:
        numpy_ver = "0"
    finally:
        req_pkg["numpy"] = (numpy_ver, MIN_NUMPY)

    # ------------------------------------------------------
    try:
        from scipy import __version__ as scipy_ver
    except Exception:
        scipy_ver = "0"
    finally:
        req_pkg["scipy"] = (scipy_ver, MIN_SCIPY)

    # ------------------------------------------------------
    try:
        from wx import __version__ as wx_ver
    except Exception:
        wx_ver = "0"
    finally:
        req_pkg["wxpython"] = (wx_ver, MIN_WXPYTHON)

    # ------------------------------------------------------
    try:
        from setuptools import __version__ as setup_ver
    except Exception:
        setup_ver = "0"
    finally:
        req_pkg["setuptools"] = (setup_ver, MIN_SETUPTOOLS)

    # ------------------------------------------------------
    try:
        shell = os.name != 'nt'
        p = subprocess.Popen("gcc -dumpversion", stdout=subprocess.PIPE,
                             shell=shell)
        gcc_ver = p.stdout.read().strip()
    except Exception:
        gcc_ver = "0"
    finally:
        req_pkg["gcc"] = (gcc_ver, MIN_GCC)

    # ------------------------------------------------------
    try:
        from pyparsing import __version__ as parse_ver
    except Exception:
        parse_ver = "0"
    finally:
        req_pkg["pyparsing"] = (parse_ver, MIN_PYPARSING)

    # ------------------------------------------------------
    try:
        from periodictable import __version__ as ptab_ver
    except Exception:
        ptab_ver = "0"
    finally:
        req_pkg["periodictable"] = (ptab_ver, MIN_PERIODICTABLE)

    # ------------------------------------------------------
    try:
        from nose import __version__ as nose_ver
    except Exception:
        nose_ver = "0"
    finally:
        req_pkg["nose"] = (nose_ver, MIN_NOSE)

    # ------------------------------------------------------
    try:
        from sphinx import __version__ as sphinx_ver
    except Exception:
        sphinx_ver = "0"
    finally:
        req_pkg["sphinx"] = (sphinx_ver, MIN_SPHINX)

    # ------------------------------------------------------
    try:
        from docutils import __version__ as docutils_ver
    except Exception:
        docutils_ver = "0"
    finally:
        req_pkg["docutils"] = (docutils_ver, MIN_DOCUTILS)

    # ------------------------------------------------------
    try:
        from pygments import __version__ as pygments_ver
    except Exception:
        pygments_ver = "0"
    finally:
        req_pkg["pygments"] = (pygments_ver, MIN_PYGMENTS)

    # ------------------------------------------------------
    try:
        from jinja2 import __version__ as jinja2_ver
    except Exception:
        jinja2_ver = "0"
    finally:
        req_pkg["jinja2"] = (jinja2_ver, MIN_JINJA2)

    # ------------------------------------------------------
    if os.name == 'nt':
        try:
            from py2exe import __version__ as py2exe_ver
        except Exception:
            py2exe_ver = "0"
        finally:
            req_pkg["py2exe"] = (py2exe_ver, MIN_PY2EXE)

        if os.path.isfile(INNO):
            req_pkg["Inno Setup Compiler"] = ("?", MIN_INNO)
        else:
            req_pkg["Inno Setup Compiler"] = ("0", MIN_INNO)

    # ------------------------------------------------------
    error = False
    for key in req_pkg:
        if req_pkg[key][0] == "0":
            print("====> %s not found; version %s or later is required - ERROR"
                  % (key, req_pkg[key][1]))
            error = True
        elif req_pkg[key][0] == "?":
            print("Found %s" %(key))  # version is unknown
        elif PV(req_pkg[key][0]) >= PV(req_pkg[key][1]):
            print("Found %s %s" %(key, req_pkg[key][0]))
        else:
            print("Found %s %s but minimum tested version is %s - WARNING"
                  % (key, req_pkg[key][0], req_pkg[key][1]))
            error = True

    if error:
        ans = input("\nDo you want to continue (Y|N)? [N]: ")
        if ans.upper() != "Y":
            sys.exit()
    else:
        print("\nSoftware dependencies have been satisfied")


def exec_cmd(command, cwd=None):
    """Runs the specified command in a subprocess."""

    if isinstance(command, list):
        command_str = " ".join('"%s"' % p if ' ' in p else p for p in command)
        shell = os.name == 'nt'
    else:
        command_str = command
        shell = True
    print("%s$ %s" % (os.getcwd(), command_str))
    result = subprocess.call(command, shell=shell, cwd=cwd)
    if result != 0:
        sys.exit(result)

BUILD_POINTS = [
    ('deps', check_dependencies),
    ('update', checkout_code),
    ('build', build_package),
    ('test', run_tests),
    ('docs', build_documentation),  # Needed by windows installer
    ('zip', create_archive),
    ('exe', create_binary),
    ('installer', create_installer),
]

def main():
    points, _ = zip(*BUILD_POINTS)
    start = BUILD_POINTS[0][0]
    only = False
    if len(sys.argv) > 1:
        # Display help if requested.
        if (len(sys.argv) > 1 and sys.argv[1] not in points
                or len(sys.argv) > 2 and sys.argv[2] != 'only'
                or len(sys.argv) > 3):
            print("\nUsage: python master_builder.py [<start>] [only]\n")
            print("Build start points:")
            for point, function in BUILD_POINTS:
                print("  %-10s %s"%(point, function.__doc__))
            print("Add 'only' to the command to only perform a single step")
            sys.exit()
        else:
            start = sys.argv[1]
            only = len(sys.argv) > 2 and sys.argv[2] == 'only'

    get_version()
    print("\nBuilding the %s-%s application from the %s repository ...\n"
          % (APP_NAME, PKG_VERSION, PKG_NAME))
    print("Current working directory  = %s"%RUN_DIR)
    print("Top-level (root) directory = %s"%TOP_DIR)
    print("Package (source) directory = %s"%REFL_DIR)

    started = False
    for point, function in BUILD_POINTS:
        if point == start:
            started = True
        if not started:
            continue
        print("%s %s %s"%("/"*5, point, "/"*25))
        function()
        if only:
            break

if __name__ == "__main__":
    main()
