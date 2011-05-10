# This program is in the public domain.
"""
Setup file for constructing OS X applications.

Run using::

    % python setup-app.py
"""

import sys
from distutils.core import setup
import py2app

if len(sys.argv) == 1:
    sys.argv.append('py2app')


# TODO: Combine with setup-py2exe so that consistency is easier.
packages = []
includes = []
excludes = ['Tkinter', 'PyQt4', '_ssl', '_tkagg', 'numpy.distutils.test']
subs = [s+'*.*' for s in ('','pubsub1/','pubsub2/','utils/',
                          'core/','core/arg1/','core/kwargs/',)]
PACKAGE_DATA = { 'wx.lib.pubsub': subs }

from distutils.core import setup
import py2app
import refl1d
import periodictable
from refl1d.gui.utilities import resource as refl1d_resource

NAME = 'Refl1D'
# Until we figure out why packages=... doesn't work reliably,
# use py2app_main with explicit imports of everything we
# might need.
SCRIPT = 'py2app_main.py'
#SCRIPT = 'bin/refl1d_gui.py'
VERSION = refl1d.__version__
ICON = 'extra/refl1d.icns'
ID = 'Refl1D'
COPYRIGHT = 'This program is public domain'
DATA_FILES = refl1d.data_files() + periodictable.data_files()

plist = dict(
    CFBundleIconFile            = ICON,
    CFBundleName                = NAME,
    CFBundleShortVersionString  = ' '.join([NAME, VERSION]),
    CFBundleGetInfoString       = NAME,
    CFBundleExecutable          = NAME,
    CFBundleIdentifier          = 'gov.nist.ncnr.%s' % ID,
    NSHumanReadableCopyright    = COPYRIGHT
)


app_data = dict(script=SCRIPT, plist=plist)
py2app_opt = dict(argv_emulation=True,
                  packages=packages,
                  includes=includes,
                  excludes=excludes,
                  iconfile=ICON,
                  optimize=2)
options = dict(py2app=py2app_opt,)

setup(
  data_files = DATA_FILES,
  package_data = PACKAGE_DATA,
  app = [app_data],
  options = options,
)
