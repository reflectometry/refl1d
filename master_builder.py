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

# Author: James Krycka

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
"""

import os
import sys
import shutil
import subprocess

# Windows commands to run utilities
SVN    = "svn"
PYTHON = "python"
INNO   = r"C:\Program Files\Inno Setup 5\ISCC.exe"  # command line operation

# URL of the Subversion repository where the source code lives
SVN_REPO_URL = "svn://svn@danse.us/reflectometry/trunk/refl1d"
# Name of the package
PKG_NAME = "refl1d" # temporary name
# Name of the application we're building
APP_NAME = "refl1d" # temporary name
# Relative path for local install under our build tree; this is used in place
# of the default installation path on Windows of C:\PythonNN\Lib\site-packages
LOCAL_INSTALL = "local-site-packages"

# Required versions of Python packages and utilities to build the application.
MIN_PYTHON = "2.5"
MAX_PYTHON = "3.0"
MIN_MATPLOTLIB = "0.99.0"
MIN_NUMPY = "1.2.1"
MIN_SCIPY = "0.7.0"
MIN_WXPYTHON = "2.8.10.0"
MIN_SETUPTOOLS = "0.6c9"
MIN_GCC = "3.4.4"
MIN_PYPARSING = "1.5.5"
MIN_PERIODICTABLE = "1.3"
# Required versions of Python packages to run tests.
MIN_NOSE = "0.11"
# Required versions of Python packages and utilities to build documentation.
MIN_SPHINX = "1.0"
MIN_DOCUTILS = "0.5"
MIN_PYGMENTS = "1.0"
MIN_JINJA2 = "2.2"
#MIN_MATHJAX = "1.0.1"
#MIN_PYPNG = "0.0.11"
# Required versions of Python packages and utilities to build Windows frozen
# image and Windows installer.
MIN_PY2EXE = "0.6.9"
MIN_INNO = "5.3.10"

# Create a line separator string for printing
SEPARATOR = "\n" + "/"*79

# Determine the full directory paths of the top-level, source, and installation
# directories based on the directory where the script is running.  Here the
# top-level directory refers to the parent directory of the package.
RUN_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
head, tail = os.path.split(RUN_DIR)
if tail == PKG_NAME:
    TOP_DIR = head
else:
    TOP_DIR = RUN_DIR
SRC_DIR = os.path.join(TOP_DIR, PKG_NAME)
INS_DIR = os.path.join(TOP_DIR, LOCAL_INSTALL)

#==============================================================================

def build_it():
    # Check the system for all required dependent packages.
    if not (len(sys.argv) > 1 and '-s' in sys.argv[1:]):
        check_dependencies()

    # Checkout code from repository.
    if not (len(sys.argv) > 1 and '-u' in sys.argv[1:]):
        checkout_code()

    # Get the version string of the application for use later.
    # This has to be done after we have checked out the repository.
    for line in open(os.path.join(SRC_DIR, 'refl1d', '__init__.py')).readlines():
        if (line.startswith('__version__')):
            exec(line.strip())
            break

    # Create an archive of the source code.
    if not (len(sys.argv) > 1 and '-a' in sys.argv[1:]):
        create_archive(__version__)

    # Install the application in a local directory tree.
    install_package()

    # Build HTML and PDF documentaton using sphinx.
    # This step is done before building the Windows installer so that PDF
    # documentation can be included in the installable product.
    if not (len(sys.argv) > 1 and '-d' in sys.argv[1:]):
        build_documentation()

    # Create a Windows executable file using py2exe.
    if not (len(sys.argv) > 1 and '-w' in sys.argv[1:]):
        if os.name == 'nt':
            create_windows_exe()

    # Create a Windows installer/uninstaller exe using the Inno Setup Compiler.
    if not (len(sys.argv) > 1 and '-w' in sys.argv[1:]):
        if os.name == 'nt':
            create_windows_installer(__version__)

    # Run unittests and doctests using a test script.
    if not (len(sys.argv) > 1 and '-t' in sys.argv[1:]):
        run_tests()

'''
    # Run unittests using Nose.
    if not (len(sys.argv) > 1 and '-t' in sys.argv[1:]):
        run_unittests()

    # Run doctests using Nose.
    if not (len(sys.argv) > 1 and '-t' in sys.argv[1:]):
        run_doctests()
'''

def checkout_code():
    # Checkout the application code from the repository into a directory tree
    # under the top level directory.
    print SEPARATOR
    print "\nStep 1 - Checking out application code from the repository ...\n"

    if RUN_DIR == TOP_DIR:
        os.chdir(TOP_DIR)
        exec_cmd("%s checkout %s %s" %(SVN, SVN_REPO_URL, PKG_NAME))
    else:
        os.chdir(SRC_DIR)
        exec_cmd("%s update" %SVN)


def create_archive(version=None):
    # Create zip and tar archives of the source code and a manifest file
    # containing the names of all files.
    print SEPARATOR
    print "\nStep 2 - Creating an archive of the source code ...\n"
    os.chdir(SRC_DIR)

    try:
        # Create zip and tar archives in the dist subdirectory.
        exec_cmd("%s setup.py sdist --formats=zip,gztar" %(PYTHON))
    except:
        print "*** Failed to create source archive ***"
    else:
        # Copy the archives and its manifest to the top-level directory.
        shutil.move(os.path.join("dist", PKG_NAME+"-"+str(version)+".zip"),
                    os.path.join(TOP_DIR, APP_NAME+"-"+str(version)+"-source.zip"))
        shutil.move(os.path.join("dist", PKG_NAME+"-"+str(version)+".tar.gz"),
                    os.path.join(TOP_DIR, APP_NAME+"-"+str(version)+"-source.tar.gz"))
        shutil.copy(os.path.join(SRC_DIR, "MANIFEST"),
                    os.path.join(TOP_DIR, APP_NAME+"-"+str(version)+"-source-manifest.txt"))


def install_package():
    # Install the application package in a private directory tree.
    # If the INS_DIR directory already exists, warn the user.
    # Intermediate work files are stored in the <SRC_DIR>/build directory tree.
    print SEPARATOR
    print "\nStep 3 - Installing the %s package in %s...\n" %(PKG_NAME, INS_DIR)
    os.chdir(SRC_DIR)

    if os.path.isdir(INS_DIR):
        print "WARNING: In order to build", APP_NAME, "cleanly, the local build"
        print "directory", INS_DIR, "needs to be deleted."
        print "Do you want to delete this directory and continue (D)"
        print "            or leave contents intact and continue (C)"
        print "            or exit the build script (E)"
        answer = raw_input("Please choose either (D|C|E)? [E]: ")
        if answer.upper() == "D":
            shutil.rmtree(INS_DIR, ignore_errors=True)
        elif answer.upper() == "C":
            pass
        else:
            sys.exit()

    # Perform the installation to a private directory tree and create the
    # PYTHONPATH environment variable to pass this info to the py2exe build
    # script later on.
    exec_cmd("%s setup.py -q install --install-lib=%s" %(PYTHON, INS_DIR))
    os.environ["PYTHONPATH"] = INS_DIR

    # WORKAROUND: copy reflmodule.pyd from the INS_DIR path to the SRC_DIR
    # path so that later py2exe will find it - py2exe finds .pyc files in
    # INS_DIR but looks for reflmodule.pyd in SRC_DIR ???
    shutil.copy(os.path.join(INS_DIR, PKG_NAME, "refl1d", "reflmodule.pyd"),
                os.path.join(SRC_DIR, "refl1d"))


def build_documentation():
    # Run the Sphinx utility to build the application's documentation.
    print SEPARATOR
    print "\nStep 4 - Running the Sphinx utility to build documentation ...\n"
    os.chdir(os.path.join(SRC_DIR, "doc"))

    # Delete any left over files from a previous build.
    exec_cmd("make clean")
    # Create documentation in HTML format.
    exec_cmd("make html")
    # Create documentation in PDF format.
    exec_cmd("make pdf")
    # Copy PDF to the doc directory where the py2exe script will look for it.
    pdf = os.path.join("_build", "latex", "Refl1D.pdf")
    if os.path.isfile(pdf):
        shutil.copy(pdf, ".")


def create_windows_exe():
    # Use py2exe to create a Win32 executable along with auxiliary files in the
    # <SRC_DIR>/dist directory tree.
    print SEPARATOR
    print "\nStep 5 - Using py2exe to create a Win32 executable ...\n"
    os.chdir(SRC_DIR)

    exec_cmd("%s setup_py2exe.py" %PYTHON)


def create_windows_installer(version=None):
    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller for
    # the application.
    print SEPARATOR
    print "\nStep 6 - Running Inno Setup Compiler to create Win32 "\
          "installer/uninstaller ...\n"
    os.chdir(SRC_DIR)

    # First create an include file to convey the application's version
    # information to the Inno Setup compiler.  If the include file exists, then
    # append the directive at the end.
    f = open("%s.iss-include" %APP_NAME, "a")
    f.write('#define MyAppVersion "%s"\n' %version)  # version must be quoted
    f.close()

    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller.
    # Override the output specification in <APP_NAME>.iss to put the executable
    # and the manifest file in the top-level directory.
    exec_cmd("%s /Q /O%s %s.iss" %(INNO, TOP_DIR, APP_NAME))


def run_tests():
    # Run unittests and doctests using a test script.
    # Running from a test script allows customization of the system path.
    print SEPARATOR
    print "\nStep 7 - Running tests from test.py (using Nose) ...\n"
    os.chdir(os.path.join(INS_DIR, PKG_NAME))

    exec_cmd("%s test.py" %PYTHON)


'''
def run_unittests():
    # Run Nose unittests.
    print SEPARATOR
    print "\nStep 7 - Running Nose unittests ...\n"
    os.chdir(INS_DIR)

    exec_cmd("nosetests -v %s" %PKG_NAME)


def run_doctests():
    # Run Nose doctests.
    print SEPARATOR
    print "\nStep 8 - Running Nose doctests ...\n"
    os.chdir(INS_DIR)

    exec_cmd("nosetests -v --with-doctest %s" %PKG_NAME)
'''

def check_dependencies():
    """
    Checks that the system has the necessary Python packages installed.
    """

    import platform
    from pkg_resources import parse_version as PV

    # ------------------------------------------------------
    python_ver = platform.python_version()
    print "Using Python", python_ver
    print ""
    if PV(python_ver) < PV(MIN_PYTHON) or PV(python_ver) >= PV(MAX_PYTHON):
        print "ERROR - build requires Python >= %s, but < %s" %(MIN_PYTHON,
                                                                MAX_PYTHON)
        sys.exit()

    req_pkg = {}

    # ------------------------------------------------------
    try:
        from matplotlib import __version__ as mpl_ver
    except:
        mpl_ver = "0"
    finally:
        req_pkg["matplotlib"] = (mpl_ver, MIN_MATPLOTLIB)

    # ------------------------------------------------------
    try:
        from numpy import __version__ as numpy_ver
    except:
        numpy_ver = "0"
    finally:
        req_pkg["numpy"] = (numpy_ver, MIN_NUMPY)

    # ------------------------------------------------------
    try:
        from scipy import __version__ as scipy_ver
    except:
        scipy_ver = "0"
    finally:
        req_pkg["scipy"] = (scipy_ver, MIN_SCIPY)

    # ------------------------------------------------------
    try:
        from wx import __version__ as wx_ver
    except:
        wx_ver = "0"
    finally:
        req_pkg["wxpython"] = (wx_ver, MIN_WXPYTHON)

    # ------------------------------------------------------
    try:
        from setuptools import __version__ as setup_ver
    except:
        setup_ver = "0"
    finally:
        req_pkg["setuptools"] = (setup_ver, MIN_SETUPTOOLS)

    # ------------------------------------------------------
    try:
        if os.name == 'nt': flag = False
        else:               flag = True
        p = subprocess.Popen("gcc -dumpversion", stdout=subprocess.PIPE,
                             shell=flag)
        gcc_ver = p.stdout.read().strip()
    except:
        gcc_ver = "0"
    finally:
        req_pkg["gcc"] = (gcc_ver, MIN_GCC)

    # ------------------------------------------------------
    try:
        from pyparsing import __version__ as parse_ver
    except:
        parse_ver = "0"
    finally:
        req_pkg["pyparsing"] = (parse_ver, MIN_PYPARSING)

    # ------------------------------------------------------
    try:
        from periodictable import __version__ as ptab_ver
    except:
        ptab_ver = "0"
    finally:
        req_pkg["periodictable"] = (ptab_ver, MIN_PERIODICTABLE)

    # ------------------------------------------------------
    try:
        from nose import __version__ as nose_ver
    except:
        nose_ver = "0"
    finally:
        req_pkg["nose"] = (nose_ver, MIN_NOSE)

    # ------------------------------------------------------
    try:
        from sphinx import __version__ as sphinx_ver
    except:
        sphinx_ver = "0"
    finally:
        req_pkg["sphinx"] = (sphinx_ver, MIN_SPHINX)

    # ------------------------------------------------------
    try:
        from docutils import __version__ as docutils_ver
    except:
        docutils_ver = "0"
    finally:
        req_pkg["docutils"] = (docutils_ver, MIN_DOCUTILS)

    # ------------------------------------------------------
    try:
        from pygments import __version__ as pygments_ver
    except:
        pygments_ver = "0"
    finally:
        req_pkg["pygments"] = (pygments_ver, MIN_PYGMENTS)

    # ------------------------------------------------------
    try:
        from jinja2 import __version__ as jinja2_ver
    except:
        jinja2_ver = "0"
    finally:
        req_pkg["jinja2"] = (jinja2_ver, MIN_JINJA2)

    # ------------------------------------------------------
    if os.name == 'nt':
        try:
            from py2exe import __version__ as py2exe_ver
        except:
            py2exe_ver = "0"
        finally:
            req_pkg["py2exe"] = (py2exe_ver, MIN_PY2EXE)

        if os.path.isfile(INNO):
            req_pkg["Inno Setup Compiler"] = ("?", MIN_INNO)
        else:
            req_pkg["Inno Setup Compiler"] = ("0", MIN_INNO)

    # ------------------------------------------------------
    error = False
    for key, values in req_pkg.items():
        if req_pkg[key][0] == "0":
            print "====> %s not found; version %s or later is required - ERROR" \
                %(key, req_pkg[key][1])
            error = True
        elif req_pkg[key][0] == "?":
            print "Found %s" %(key)  # version is unknown
        elif PV(req_pkg[key][0]) >= PV(req_pkg[key][1]):
            print "Found %s %s" %(key, req_pkg[key][0])
        else:
            print "Found %s %s but minimum tested version is %s - WARNING" \
                %(key, req_pkg[key][0], req_pkg[key][1])
            error = True

    if error:
        ans = raw_input("\nDo you want to continue (Y|N)? [N]: ")
        if ans.upper() != "Y":
            sys.exit()
    else:
        print "\nSoftware dependencies have been satisfied"


def exec_cmd(command):
    """Runs the specified command in a subprocess."""

    if os.name == 'nt': flag = False
    else:               flag = True

    subprocess.call(command, shell=flag)


if __name__ == "__main__":
    if len(sys.argv)>1:
        # Display help if requested.
        if len(sys.argv) > 1 and '-h' in sys.argv[1:]:
            print "\nUsage: python master_builder.py [options]\n"
            print "Options:"
            print "  -a  Skip build of source archive"
            print "  -d  Skip build of documentation"
            print "  -h  Show this help text"
            print "  -s  Skip software dependency checks"
            print "  -t  Skip all tests (unittests and doctests)"
            print "  -w  Skip build of Windows executable and installer"
            print "  -u  Skip checkout or update from repository"
            sys.exit()

    print "\nBuilding the %s application from the %s repository ...\n" \
        %(APP_NAME, PKG_NAME)
    print "Current working directory  =", RUN_DIR
    print "Top-level (root) directory =", TOP_DIR
    print "Package (source) directory =", SRC_DIR
    print "Installation directory     =", INS_DIR
    print ""

    build_it()
