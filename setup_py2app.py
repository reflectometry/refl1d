# This program is in the public domain.
"""
Setup file for constructing OS X applications.

Run using::

    % python setup-app.py
"""

import os
import sys
import shutil

import py2app # @UnresolvedImport @UnusedImport except on mac
from distutils.core import setup
from distutils.util import get_platform

# Put the build lib on the start of the path.
# For packages with binary extensions, need platform.  If it is a pure
# script library, use an empty platform string.
platform = '.%s-%s'%(get_platform(),sys.version[:3])
sys.path.insert(0, os.path.abspath('../periodictable'))
sys.path.insert(0, os.path.abspath('../bumps/build/lib'+platform))
sys.path.insert(0, os.path.abspath('build/lib'+platform))

# Remove the current directory from the python path
here = os.path.abspath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != here]
#print "\n".join(sys.path)


# Force build of bumps and refl1d before continuing
os.environ['PYTHONPATH'] = '../periodictable:../bumps/build/lib'+platform
os.system('cd ../bumps && "%s" setup.py build'%sys.executable)
os.system('"%s" setup.py build'%sys.executable)

if len(sys.argv) == 1:
    sys.argv.append('py2app')


# TODO: Combine with setup-py2exe so that consistency is easier.
packages = ['numpy', 'scipy', 'matplotlib', 'pytz', 'periodictable', 
            'bumps', 'refl1d', 'IPython', 'wx']
includes = []
excludes = ['Tkinter', 'PyQt4', '_ssl', '_tkagg', 'numpy.distutils.test']
PACKAGE_DATA = {}

import refl1d
import bumps
import periodictable

NAME = 'Refl1D'
# Until we figure out why packages=... doesn't work reliably,
# use py2app_main with explicit imports of everything we
# might need.
#SCRIPT = 'py2app_main.py'
SCRIPT = 'bin/refl1d_gui.py'
VERSION = refl1d.__version__
ICON = 'extra/refl1d.icns'
ID = 'Refl1D'
COPYRIGHT = 'This program is public domain'
DATA_FILES = bumps.data_files() + periodictable.data_files()

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

PRODUCT = NAME+" "+VERSION
PRODUCTDASH = NAME+"-"+VERSION
APP="dist/%s.app"%PRODUCT
DMG="dist/%s.dmg"%PRODUCTDASH
def build_app():
    setup(
          data_files = DATA_FILES,
          package_data = PACKAGE_DATA,
          app = [app_data],
          options = options,
          )
    os.system('cp -a extra/appbin/* "dist/%s.app"'%NAME)

def build_dmg():
    """DMG builder; should include docs"""
    # Remove previous build if it is still sitting there
    if os.path.exists(APP): shutil.rmtree(APP)
    if os.path.exists(DMG): os.unlink(DMG)
    os.rename("dist/%s.app"%NAME, APP)
    os.system('cd dist && ../extra/dmgpack.sh "%s" "%s.app" ../doc/_build/html ../doc/examples'%(PRODUCTDASH,PRODUCT))
    os.system('chmod a+r "%s"'%DMG)

if __name__ == "__main__":
    build_app()
    build_dmg()
