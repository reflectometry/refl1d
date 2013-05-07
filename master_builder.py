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

import os
import sys
import shutil
import subprocess

GIT="git"
MAKE="make"

#PYTHON = sys.executable
os.environ['PYTHONHOME'] = sys.prefix
PYTHON = os.path.join(sys.real_prefix,'bin','python')
os.environ['PYTHON'] = PYTHON

# Windows commands to run utilities
if os.name == "nt":
    SYSBIN=r"C:\Program Files (x86)"
    if not os.path.exists(SYSBIN): SYSBIN=r"C:\Program Files"
    GIT = SYSBIN+r"\Git\bin\git.exe"
    INNO   = SYSBIN+r"\Inno Setup 5\ISCC.exe"  # command line operation

    if not os.path.exists(GIT):
        print >>sys.stderr, "missing git: "+GIT
        sys.exit(1)
    if not os.path.exists(INNO):
        print >>sys.stderr, "missing inno setup: "+INNO
        sys.exit(1)

    # Put PYTHON in the environment and add the python directory and its
    # corresponding script directory (for nose, sphinx, pip, etc) to the path.
    PYTHONDIR = os.path.dirname(os.path.abspath(PYTHON))
    SCRIPTDIR = os.path.join(PYTHONDIR,'Scripts')
    os.environ['PATH'] = ";".join((PYTHONDIR,SCRIPTDIR,os.environ['PATH']))
    os.environ['PYTHON'] = "/".join(PYTHON.split("\\"))
    #os.environ['GIT_SSH'] = r"C:\Program Files (x86)\PuTTY\plink.exe"
    MAKE = r"C:\mingw\bin\make"

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

# Relative path for local install under our build tree; this is used in place
# of the default installation path on Windows of C:\PythonNN\Lib\site-packages
LOCAL_INSTALL = "local-site-packages"

# Determine the full directory paths of the top-level, source, and installation
# directories based on the directory where the script is running.  Here the
# top-level directory refers to the parent directory of the package.
RUN_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
head, tail = os.path.split(RUN_DIR)
if tail == PKG_NAME:
    TOP_DIR = head
else:
    TOP_DIR = RUN_DIR
INS_DIR = os.path.join(TOP_DIR, LOCAL_INSTALL)
os.environ['PYTHONPATH'] = INS_DIR

BUMPS_DIR = os.path.join(TOP_DIR, "bumps")
BUMPS_NEW = '"%s" clone -b bumps-split https://github.com/reflectometry/bumps.git'%GIT
SRC_DIR = os.path.join(TOP_DIR, "refl1d")
SRC_NEW = '"%s" clone https://github.com/reflectometry/refl1d.git'%GIT
REPO_UPDATE = '"%s" pull'%GIT


def get_version():
    "Determine package version"
    global PKG_VERSION
    # Get the version string of the application for use later.
    # This has to be done after we have checked out the repository.
    for line in open(os.path.join(SRC_DIR, PKG_NAME, '__init__.py')).readlines():
        if (line.startswith('__version__')):
            PKG_VERSION = line.split('=')[1].strip()[1:-1]
            break
    else:
        raise RuntimeError("Could not find package version")

#==============================================================================

def checkout_code():
    "download or update source code"
    print "Checking out application code from the repository ...\n"

    if RUN_DIR == TOP_DIR:
        os.chdir(TOP_DIR)
        exec_cmd(BUMPS_NEW)
        exec_cmd(SRC_NEW)
    else:
        os.chdir(BUMPS_DIR)
        exec_cmd(REPO_UPDATE)
        os.chdir(SRC_DIR)
        exec_cmd(REPO_UPDATE)

    get_version()  # reset version number in case it was updated remotely


def create_archive(version=None):
    "create source distribution"
    if version == None: version = PKG_VERSION

    # Create zip and tar archives of the source code and a manifest file
    # containing the names of all files.
    print "Creating an archive of the source code ...\n"

    
    try:
        # Create zip and tar archives in the dist subdirectory.
        os.chdir(SRC_DIR)
        exec_cmd("%s setup.py sdist --formats=zip,gztar" %(PYTHON))
    except:
        print "*** Failed to create source archive ***"
    else:
        # Copy the archives and its source listing to the top-level directory.
        # The location of the file that contains the source listing and the
        # name of the file varies depending on what package is used to import
        # setup, so its copy is made optional while we are making setup changes.
        shutil.move(os.path.join("dist", PKG_NAME+"-"+str(version)+".zip"),
                    os.path.join(TOP_DIR, PKG_NAME+"-"+str(version)+"-source.zip"))
        shutil.move(os.path.join("dist", PKG_NAME+"-"+str(version)+".tar.gz"),
                    os.path.join(TOP_DIR, PKG_NAME+"-"+str(version)+"-source.tar.gz"))
        listing = os.path.join(SRC_DIR, PKG_NAME+".egg-info", "SOURCES.txt")
        if os.path.isfile(listing):
            shutil.copy(listing,
                os.path.join(TOP_DIR, PKG_NAME+"-"+str(version)+"-source-list.txt"))


def install_package():
    "build and install the package"
    # If the INS_DIR directory already exists, warn the user.
    # Intermediate work files are stored in the <SRC_DIR>/build directory tree.
    print "Installing the %s package in %s...\n" %(PKG_NAME, INS_DIR)

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
    if not os.path.exists(INS_DIR):
        os.makedirs(INS_DIR)

    os.chdir(BUMPS_DIR)
    exec_cmd("%s setup.py install --install-lib=%s" %(PYTHON, INS_DIR))
    os.chdir(SRC_DIR)
    exec_cmd("%s setup.py install --install-lib=%s" %(PYTHON, INS_DIR))


def build_documentation():
    "build the documentation"
    print "Running the Sphinx utility to build documentation ...\n"
    os.chdir(os.path.join(SRC_DIR, "doc"))

    # Delete any left over files from a previous build.
    # Create documentation in HTML and PDF format.
    exec_cmd(MAKE+" clean")
    exec_cmd(MAKE+" html")
    #exec_cmd(MAKE+" pdf")
    # Copy PDF to the html directory where the html can find it.
    pdf = os.path.join("_build", "latex", APP_NAME+".pdf")
    if os.path.isfile(pdf):
        shutil.copy(pdf, os.path.join("_build","html"))


def create_windows_exe():
    "create the standalone windows executable"
    if os.name != 'nt': return
    # Use py2exe to create a Win32 executable along with auxiliary files in the
    # <SRC_DIR>/dist directory tree.
    print "Using py2exe to create a Win32 executable ...\n"
    os.chdir(SRC_DIR)

    exec_cmd("%s setup_py2exe.py" %PYTHON)


def create_windows_installer(version=None):
    "create the windows installer"
    if os.name != 'nt': return
    if not version: version = PKG_VERSION
    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller for
    # the application.
    print "Running Inno Setup Compiler to create Win32 "\
          "installer/uninstaller ...\n"
    os.chdir(SRC_DIR)

    # First create an include file to convey the application's version
    # information to the Inno Setup compiler.
    f = open("iss-version", "w")
    f.write('#define MyAppVersion "%s"\n' %version)  # version must be quoted
    f.close()

    # Run the Inno Setup Compiler to create a Win32 installer/uninstaller.
    # Override the output specification in <PKG_NAME>.iss to put the executable
    # and the manifest file in the top-level directory.
    exec_cmd("%s /Q /O%s %s.iss" %(INNO, TOP_DIR, PKG_NAME))


def run_tests():
    "run the test suite"
    # Run unittests and doctests using a test script.
    # Running from a test script allows customization of the system path.
    print "Running tests from test.py (using Nose) ...\n"
    #os.chdir(os.path.join(INS_DIR, PKG_NAME))
    os.chdir(SRC_DIR)

    exec_cmd("%s test.py" %PYTHON)

def check_dependencies():
    "check that required packages are installed"

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

    shell = os.name != 'nt'
    print os.getcwd(),"$",command
    result = subprocess.call(command, shell=shell)
    if result != 0: sys.exit(result)

BUILD_POINTS = [
  ('deps', check_dependencies),
  ('update', checkout_code),
  ('build', install_package),
  ('test', run_tests),
  ('docs', build_documentation),  # Needed by windows installer
  ('zip', create_archive),
  ('exe', create_windows_exe),
  ('installer', create_windows_installer),
]



def main():

    start = BUILD_POINTS[0][0]
    only = False
    if len(sys.argv)>1:
        # Display help if requested.
        if (len(sys.argv) > 1 and sys.argv[1] not in zip(*BUILD_POINTS)[0]
                or len(sys.argv) > 2 and sys.argv[2] != 'only'
                or len(sys.argv) > 3):
            print "\nUsage: python master_builder.py [<start>] [only]\n"
            print "Build start points:"
            for point,fn in BUILD_POINTS:
                print "  %-10s %s"%(point,fn.__doc__)
            print "Add 'only' to the command to only perform a single step"
            sys.exit()
        else:
            start = sys.argv[1]
            only = len(sys.argv)>2 and sys.argv[2] == 'only'

    get_version()
    print "\nBuilding the %s-%s application from the %s repository ...\n" \
        %(APP_NAME, PKG_VERSION, PKG_NAME)
    print "Current working directory  =", RUN_DIR
    print "Top-level (root) directory =", TOP_DIR
    print "Package (source) directory =", SRC_DIR
    print "Installation directory     =", INS_DIR

    started = False
    for point,fn in BUILD_POINTS:
        if point==start: started = True
        if not started: continue
        print "/"*5,point,"/"*25
        fn()
        if only: break
        
if __name__ == "__main__":
     main()
