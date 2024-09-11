# This program is in the public domain.
"""
Setup file for constructing OS X applications.

Run using::

    % python setup-app.py
"""

import os
import sys
import shutil

# import macholib_patch

import PyInstaller.__main__
from distutils.core import setup
from distutils.util import get_platform

# Put the build lib on the start of the path.
# For packages with binary extensions, need platform.  If it is a pure
# script library, use an empty platform string.
platform = ".%s-%s" % (get_platform(), sys.version[:3])
sys.path.insert(0, os.path.abspath("../periodictable"))
sys.path.insert(0, os.path.abspath("../bumps"))
sys.path.insert(0, os.path.abspath("build/lib" + platform))

# Remove the current directory from the python path
here = os.path.abspath(os.path.dirname(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != here]
# print "\n".join(sys.path)


# Force build of bumps and refl1d before continuing
os.environ["PYTHONPATH"] = "../periodictable:../bumps"
# os.system('cd ../bumps && "%s" setup.py build'%sys.executable)
print('"%s" setup.py build' % sys.executable)
os.system('"%s" setup.py build' % sys.executable)

if len(sys.argv) == 1:
    sys.argv.append("py2app")


# TODO: Combine with setup-py2exe so that consistency is easier.
packages = []
includes = [
    "readline",
    "ipython",
    "numpy",
    "scipy",
    "matplotlib",
    "pytz",
    "wx",
    "periodictable",
    "bumps",
    "refl1d",
    "pygments",
    "scipy.special._ufuncs_cxx",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy.sparse.csgraph._validation",
    "scipy.spatial.ckdtree",
    "scipy.spatial.*",
    "scipy.spatial.transform.*",
    "scipy._lib.messagestream",
    "scipy.special.cython_special",
    "pygments.lexers.*",
    "pygments.styles.default",
]
excludes = ["Tkinter", "PyQt4", "_ssl", "_tkagg", "numpy.distutils.test"]
PACKAGE_DATA = {}

import refl1d
import bumps
import periodictable

NAME = "Refl1D"
# Until we figure out why packages=... doesn't work reliably,
# use py2app_main with explicit imports of everything we
# might need.
# SCRIPT = 'py2app_main.py'
SCRIPT = "bin/refl1d_gui.py"
# SCRIPT = 'bin/refl1d_cli.py'
VERSION = refl1d.__version__
ICON = "extra/refl1d.icns"
ID = "Refl1D"
COPYRIGHT = "This program is public domain"
DATA_FILES = bumps.data_files() + periodictable.data_files()

plist = dict(
    CFBundleIconFile=ICON,
    CFBundleName=NAME,
    CFBundleShortVersionString=" ".join([NAME, VERSION]),
    CFBundleGetInfoString=NAME,
    CFBundleExecutable=NAME,
    CFBundleIdentifier="gov.nist.ncnr.%s" % ID,
    NSHumanReadableCopyright=COPYRIGHT,
)


app_data = dict(script=SCRIPT, plist=plist)
py2app_opt = dict(
    argv_emulation=True,
    matplotlib_backends="wx",
    verbose_interpreter=True,
    no_chdir=True,
    emulate_shell_environment=True,
    packages=packages,
    includes=includes,
    excludes=excludes,
    iconfile=ICON,
    optimize=0,
)
options = dict(
    py2app=py2app_opt,
)


def build_app():
    setup(
        data_files=DATA_FILES,
        package_data=PACKAGE_DATA,
        app=[app_data],
        options=options,
    )
    # Add cli interface to the app directory
    os.system('cp -p extra/appbin/* "dist/%s.app"' % NAME)


if __name__ == "__main__":
    run = [
        SCRIPT,
        "--name=Refl1D_gui",
        #'--onefile',
        "--windowed",
        "--icon=extra/refl1d.icns",
        "--osx-bundle-identifier=gov.nist.ncnr.refl1d",
    ]
    for dest, files in DATA_FILES:
        run.extend(["--add-data={}:{}".format(df, dest) for df in files])
    run.extend(["--hidden-import={}".format(hi) for hi in includes])
    PyInstaller.__main__.run(run)
    # import shutil
    # for dest, files in DATA_FILES:
    #    for fn in files:
    #        os.makedirs('dist/refl1d_cli.app/Contents/Resources/{}'.format(dest),exist_ok=True)
    #        shutil.copy2(fn, 'dist/refl1d_cli.app/Contents/Resources/{}/'.format(dest))
