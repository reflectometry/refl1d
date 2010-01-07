#!/usr/bin/env python

"""
Create parkApp.exe using py2exe where the output is in the dist\ directory tree.
This is a self contained Win32 distribution of the KsRefl application
(formerly known as ReflPark) using the PARK and Reflectometry repositories.

TODO: Change "parkApp" to "ksreflApp" throughout the PARK repository.
TODO: Reduce the modules pulled into the ksrefl.zip to what is actually needed.
"""
import os
import sys
import glob
from distutils.core import setup
from distutils.filelist import findall

import matplotlib
import py2exe

import periodictable
import refl1d

if len(sys.argv) == 1:
    sys.argv.append('py2exe')

# Retrieve version information.

class Target:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        # for the version info resources
        self.version = refl1d.__version__
        self.company_name = "NIST"
        self.copyright = "BSD style copyright"
        self.name = "REFL1D"


data_files = []
# Add matplotlib data files from the mpl-data folder and subfolders.
# The compiled program will look for it in \mpl-data for these files.
# Note that the location of these data files varies dependnig on the version of
# MatPlotLib that is being used.
# Note that glob will not find files that have no extension such as matplotlibrc.
#
# The technique for obtaining MatPlotLib auxliliary files was adapted from the
# examples and discussion on http://www.py2exe.org/index.cgi/MatPlotLib.
matplotlibdatadir = matplotlib.get_data_path()
mplData = ('mpl-data', glob.glob(os.path.join(matplotlibdatadir,'*.*')))
data_files.append(mplData)
mplData = ('mpl-data', [os.path.join(matplotlibdatadir,'matplotlibrc')])
data_files.append(mplData)
mplData = (r'mpl-data\images',glob.glob(os.path.join(matplotlibdatadir,r'images\*.*')))
data_files.append(mplData)
mplData = (r'mpl-data\fonts\afm',glob.glob(os.path.join(matplotlibdatadir,r'fonts\afm\*.*')))
data_files.append(mplData)
mplData = (r'mpl-data\fonts\pdfcorefonts',glob.glob(os.path.join(matplotlibdatadir,r'fonts\pdfcorefonts\*.*')))
data_files.append(mplData)
mplData = (r'mpl-data\fonts\ttf',glob.glob(os.path.join(matplotlibdatadir,r'fonts\ttf\*.*')))
data_files.append(mplData)

data_files += periodictable.data_files()


# Add required packages.
packages = ['numpy', 'scipy', 'matplotlib', 'pytz', 'pyparsing', 'refl1d',
            'periodictable', 'mystic']


# Specify include and exclude files.
includes = []

excludes = ['Tkinter', 'PyQt4']

dll_excludes = ['MSVCR71.dll',
                'w9xpopen.exe',
                'libgdk_pixbuf-2.0-0.dll',
                'libgobject-2.0-0.dll',
                'libgdk-win32-2.0-0.dll',
                'cygwin1.dll',
                'tcl84.dll',
                'tk84.dll',
                'QtGui4.dll',
                'QtCore4.dll']


# This will create a console window on starup in which the KsRefl reflectometry
# client/server application is run that will create a separate GUI app window.
target_console = Target(
      description = 'Refl1D: Reflectometry console fitting application.',
      script = 'main.py',
      #dest_base = "parlApp"
      )


# Now do the work to create a standalone distribution using py2exe.
setup(
       console = [ "main.py" ],

       options={
                 'py2exe': {
                    'dll_excludes': dll_excludes,
                    'packages': packages,
                    'includes': includes,
                    'excludes': excludes,
                    'compressed': 1,
                    'optimize': 0,
                    #'bundle_files': 1, # bundle python25.dll in executable file
                 },
       },
       zipfile = None,                 # bundle library.zip in executable file
       data_files=data_files,
)
