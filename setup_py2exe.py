#!/usr/bin/env python

# Copyright (C) 2006-2010, University of Maryland
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
This script uses py2exe to create dist\refl1d.exe and dist\refl1d_gui.exe for
running the Refl1D application in either CLI or GUI mode.

These executables start the application and import the rest of the application
code stored in library.zip.  The python interpreter and other required python
packages and dlls are also placed in the zip file.  Additional resource files
that are needed when Refl1D is run are copied to the dist directory tree.  On
completion, the contents of the dist directory tree can be used by the Inno
Setup Compiler (via a separate script) to build a Windows installer/uninstaller
for deployment of the Refl1D application.  For testing purposes, refl1d.exe or
refl1d_gui.exe can be run from the dist directory.
"""

import os
import sys

# Force build before continuing
#os.system('"%s" setup.py build'%sys.executable)

# Remove the current directory from the python path
here = os.path.abspath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != here]

import glob

from distutils.core import setup
from distutils.util import get_platform

import numpy.core

# Augment the setup interface with the py2exe command and make sure the py2exe
# option is passed to setup.
import py2exe # @UnresolvedImport @UnusedImport except on windows

if len(sys.argv) == 1:
    sys.argv.append('py2exe')
#print "\n".join(sys.path)

platform = '.%s-%s' % (get_platform(), sys.version[:3])
packages = [
    os.path.abspath('../periodictable'),
    os.path.abspath('../bumps'),
    #os.path.abspath('../bumps/build/lib'+platform),
    os.path.abspath('build/lib'+platform),
]
sys.path = packages + sys.path
print("=== Python Path ===\n"+"\n".join(sys.path))

#import wx  # May need this to force wx to be included
import matplotlib
matplotlib.use('WXAgg')
import periodictable
import bumps
import refl1d

# Retrieve the application version string.
version = refl1d.__version__

# A manifest is required to be included in a py2exe image (or accessible as a
# file in the image directory) when wxPython is included so that the Windows XP
# theme is used when rendering wx widgets.  The manifest must be matched to the
# version of Python that is being used.

manifest_template = """
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity
    version="%(version)s"
    processorArchitecture="%(arch)s"
    name="%(appname)s"
    type="win32">
  </assemblyIdentity>
  <description>Refl1D</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel
          level="asInvoker"
          uiAccess="false">
        </requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.VC90.CRT"
        version="9.0.21022.8"
        processorArchitecture="%(arch)s"
        publicKeyToken="1fc8b3b9a1e18e3b">
      </assemblyIdentity>
    </dependentAssembly>
  </dependency>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity
        type="win32"
        name="Microsoft.Windows.Common-Controls"
        version="6.0.0.0"
        processorArchitecture="%(arch)s"
        publicKeyToken="6595b64144ccf1df"
        language="*">
      </assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>
"""
is_64bits = sys.maxsize > 2**32
manifest_properties = {
    "arch": "amd64" if is_64bits else "x86",
    "appname": "Refl1d",
    "version": "5.0.0.0",
}
manifest = manifest_template % manifest_properties

# Create a list of all files to include along side the executable being built
# in the dist directory tree.  Each element of the data_files list is a tuple
# consisting of a path (relative to dist\) and a list of files in that path.
data_files = []

# Add resource files that need to reside in the same directory as the image.
data_files.append( ('.', [os.path.join(here, 'LICENSE.txt')]) )
data_files.append( ('.', [os.path.join(here, 'README.rst')]) )
data_files.append( ('.', [os.path.join(here, 'bin', 'refl1d_launch.bat')]) )
data_files.append( ('.', [os.path.join(here, 'extra', 'refl1d.ico')]) )

# Add application specific data files from the refl1d\refl1d-data folder.
data_files += bumps.data_files()

# Add data files from the matplotlib\mpl-data folder and its subfolders.
# For matploblib prior to version 0.99 see the examples at the end of the file.
data_files += matplotlib.get_py2exe_datafiles()

# Add data files from the periodictable\xsf folder.
data_files += periodictable.data_files()

# Add example directories and their files.  An empty directory is ignored.
# Note that Inno Setup will determine where these files will be placed such as
# C:\My Documents\... instead of the installation folder.
for path in glob.glob(os.path.join('examples', '*')):
    if os.path.isdir(path):
        for file in glob.glob(os.path.join(path, '*.*')):
            data_files.append( (path, [file]) )
    else:
        data_files.append( ('examples', [path]) )

for path in glob.glob(os.path.join('doc', 'examples', '*')):
    if os.path.isdir(path):
        for file in glob.glob(os.path.join(path, '*.*')):
            data_files.append( (path, [file]) )
    else:
        data_files.append( ('doc', [path]) )

# Add PDF documentation to the dist staging directory.
pdf = os.path.join('doc', '_build', 'latex', 'Refl1D.pdf')
if os.path.isfile(pdf):
    data_files.append( ('doc', [pdf]) )

# Add the Microsoft Visual C++ 2008 redistributable kit if we are building with
# Python 2.6 or 2.7.  This kit will be installed on the target system as part
# of the installation process for the frozen image.  Note that the Python 2.5
# interpreter requires msvcr71.dll which is included in the Python25 package,
# however, Python 2.6 and 2.7 require the msvcr90.dll but they do not bundle it
# with the Python26 or Python27 package.  Thus, for Python 2.6 and later, the
# appropriate dll must be present on the target system at runtime.
pypath = os.path.dirname(sys.executable)
vcredist = 'vcredist_%s.exe'%("x64" if is_64bits else "x32")
data_files.append( ('.', [os.path.join(pypath, vcredist)]) )

# numpy depends on some DLLs that are not being pulled in automatically
# anaconda license says that the redistribution of the Intel MKL libraries
# is permitted (https://docs.continuum.io/anaconda/eula, 2016-06-03).
libdir = os.path.join(pypath, 'Library', 'bin')
missing_libs = ['libiomp5md', 'mkl_core', 'mkl_def']
data_files.append(('.', [os.path.join(libdir, k+'.dll') for k in missing_libs]))


# Specify required packages to bundle in the executable image.
packages = [
    'numpy', 'scipy', 'matplotlib', 'pytz', 'pyparsing',
    'periodictable', 'bumps', 'refl1d', 'refl1d.names', 'refl1d.errors',
    'wx', 'wx.py.path', 'IPython', 'pyreadline',
    ]

# Specify files to include in the executable image.
includes = [ ]


# Specify files to exclude from the executable image.
# - We can safely exclude Tk/Tcl and Qt modules because our app uses wxPython.
# - We do not use ssl services so they are omitted.
# - We can safely exclude the TkAgg matplotlib backend because our app uses
#   "matplotlib.use('WXAgg')" to override the default matplotlib configuration.
# - On the web it is widely recommended to exclude certain lib*.dll modules
#   but this does not seem necessary any more (but adding them does not hurt).
# - Python25 requires mscvr71.dll, however, Win XP includes this file.
# - Since we do not support Win 9x systems, w9xpopen.dll is not needed.
# - For some reason cygwin1.dll gets included by default, but it is not needed.

excludes = ['Tkinter', 'PyQt4', '_ssl', 'tkagg', 'zmq','pyzmq','sympy'] #, 'numpy.distutils.tests']

dll_excludes = ['libgdk_pixbuf-2.0-0.dll', 'libgobject-2.0-0.dll', 'libgdk-win32-2.0-0.dll',
                'tcl84.dll', 'tk84.dll', 'tcl85.dll', 'tk85.dll',
                'QtGui4.dll', 'QtCore4.dll',
                #'msvcr71.dll', 'msvcp71.dll',
                'msvcp90.dll', 'msvcr90.dll',
                #'libiomp5md.dll', 'libifcoremd.dll', 'libmmd.dll',
                #'svml_dispmd.dll','libifportMD.dll',
                'MPR.dll', 'API-MS-Win-Core-LocalRegistry-L1-1-0.dll',
                ]

class Target():
    """This class stores metadata about the distribution in a dictionary."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.version = version

ICON_FILE = os.path.join(here, 'extra', 'refl1d.ico')
clientCLI = Target(
    name = 'Refl1D',
    description = 'Refl1D CLI application',
    script = os.path.join('bin', 'refl1d_cli.py'),  # module to run on application start
    dest_base = 'refl1d',  # file name part of the exe file to create
    icon_resources = [(1, ICON_FILE)],  # also need to specify in data_files
    bitmap_resources = [],
    other_resources = [(24, 1, manifest % dict(prog='Refl1D'))] )

clientGUI = Target(
    name = 'Refl1D',
    description = 'Refl1D GUI application',
    script = os.path.join('bin', 'refl1d_gui.py'),  # module to run on application start
    dest_base = 'refl1d_gui',  # file name part of the exe file to create
    icon_resources = [(1, ICON_FILE)],  # also need to specify in data_files
    bitmap_resources = [],
    other_resources = [(24, 1, manifest % dict(prog='Refl1D'))] )

# Now we do the work to create a standalone distribution using py2exe.
#
# When the application is run in console mode, a console window will be created
# to receive any logging or error messages and the application will then create
# a separate GUI application window.
#
# When the application is run in windows mode, it will create a GUI application
# window and no console window will be provided.  Output to stderr will be
# written to <app-image-name>.log.
bundle = 3 if is_64bits else 1
setup(
      console=[clientCLI],
      windows=[clientGUI],
      options={
          'py2exe': {
              'packages': packages,
              'includes': includes,
              'excludes': excludes,
              'dll_excludes': dll_excludes,
              'compressed': 1,   # standard compression
              'optimize': 0,     # no byte-code optimization
              'dist_dir': "dist",# where to put py2exe results
              'xref': False,     # display cross reference (as html doc)
              'bundle_files': bundle, # bundle python25.dll in library.zip
              'custom_boot_script': 'py2exe_boot.py',
          }
      },
      # Since we are building two exe's, do not put the shared library in each
      # of them.  Instead create a single, separate library.zip file.
      ### zipfile=None,               # bundle library.zip in exe
      data_files=data_files,          # list of files to copy to dist directory
)
