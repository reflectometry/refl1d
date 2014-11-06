.. _contributing:

********************
Contributing Changes
********************

.. contents:: :local:

The best way to contribute to the reflectometry package is to work
from a copy of the source tree in the revision control system.

The refl1d project is hosted on github at:

    https://github.com/reflectometry/refl1d

You will need the git source control software for your computer.  This can
be downloaded from the `git page <http://www.git-scm.com/>`_, or you can use
an integrated development environment (IDE) such as Eclipse and PyCharm, which
may have git built in.

Simple patches
--------------

If you want to make one or two tiny changes, it is easiest to clone the
project, make the changes, document and test, then send a patch.

Clone the project as follows::

    git clone https://github.com/reflectometry/refl1d.git

You will need bumps and periodictable to run.  If you are fixing bugs in the
scattering length density calculator or the fitting engine, you will want to
clone the repositories as sister directories to the refl1d source tree::

    git clone https://github.com/bumps/bumps.git
    git clone https://github.com/pkienzle/periodictable.git

If you are only working with the refl1d modeling code, then you can install
bumps and periodictable using pip::

    pip install periodictable bumps

To run the package from the source tree use the following::

    cd refl1d
    python run.py

This will first build the package into the build directory then run it.
Any changes you make in the source directory will automatically be used in
the new version.

As you make changes to the package, you can see what you have done using git::

    git status
    git diff

Please update the documentation and add tests for your changes.  We use
doctests on all of our examples that we know our documentation is correct.
More thorough tests are found in test directory.  With the nosetest package,
you can run the tests using::

    python tests.py

Nose is available on linux form apt-get

When all the tests run, create a patch and send it to paul.kienzle@nist.gov::

    git diff > patch

Larger changes
--------------

For a larger set of changes, you should fork refl1d on github, and issue pull
requests for each part.

Once you have create the fork, the clone line is slightly different::

    git clone https://github.com/YourGithubAccount/refl1d

After you have tested your changes, you will need to push them to your github
fork::

    git log
    git commit -a -m "short sentence describing what the change is for"
    git push

Good commit messages are a bit of an art.  Ideally you should be able to
read through the commit messages and create a "what's new" summary without
looking at the actual code.

Make sure your fork is up to date before issuing a pull request.  You can
track updates to the original refl1d package using::

    git remote add refl1d https://github.com/reflectometry/refl1d
    git fetch refl1d
    git merge refl1d/master
    git push

When making changes, you need to take care that they work on different
versions of python.   In particular, RHEL6, Centos6.5, Rocks and
ScientificLinux all run python 2.6, most linux/windows/mac users run
python 2.7, but some of the more bleeding edge distributions run 3.3/3.4.
The anaconda distribution makes it convenient to maintain multiple independent
environments
Even better is to test against all python versions 2.6, 2.7, 3.3, 3.4::

    pythonX.Y tests.py
    pythonX.Y run.py

When all the tests run, issue a pull request from your github account.

Building Documentation
======================

Building the package documentation requires a working Sphinx installation,
and latex to build the pdf. As of this writing we are using sphinx 1.2.

The command line to build the docs is as follows::

    (cd doc && make clean html pdf)

You can see the result by pointing your browser to::

    doc/_build/html/index.html
    doc/_build/latex/Refl1d.pdf

Note that this only works with a unix-like environment for now since we are
using *make*.  On windows, you can run sphinx directly from python::

    cd doc
    python -m sphinx.__init__ -b html -d _build/doctrees . _build/html

ReStructured text format does not have a nice syntax for superscripts and
subscripts.  Units such as |g/cm^3| are entered using macros such as
\|g/cm^3| to hide the details.  The complete list of macros is available in

        doc/sphinx/rst_prolog

In addition to macros for units, we also define cdot, angstrom and degrees
unicode characters here.  The corresponding latex symbols are defined in
doc/sphinx/conf.py.

There is a bug in older sphinx versions (e.g., 1.0.7) in which latex tables
cannot be created.  You can fix this by changing::

    self.body.append(self.table.colspec)

to::

    self.body.append(self.table.colspec.lower())

in site-packages/sphinx/writers/latex.py.

Windows Installer
=================

To build a windows standalone executable with py2exe you may first need
to create an empty file named *Lib\\numpy\\distutils\\tests\\__init__.py*
in your python directory (usually *C:\\Python2.7*.  Without this file,
py2exe raises an error when it is searching for the parts of the numpy
package.  This may be fixed on recent versions of numpy. Next, update the
__version__ tag in refl1d/__init__.py to mark it as your own.

Now you can build the standalone executable using::

    python setup_py2exe

This creates a dist subdirectory in the source tree containing
everything needed to run the application including python and
all required packages.

To build the Windows installer, you will need two more downloads:

    - Visual C++ 2008 Redistributable Package (x86) 11/29/2007
    - `Inno Setup <http://www.jrsoftware.org/isdl.php>`_ 5.3.10 QuickStart Pack

The C++ redistributable package is needed for programs compiled with the
Microsoft Visual C++ compiler, including the standard build of the Python
interpreter for Windows.  It is available as vcredist_x86.exe from the
`Microsoft Download Center <http://www.microsoft.com/downloads/>`_.
Be careful to select the version that corresponds to the one used
to build the Python interpreter --- different versions can have the
same name.  For the Python 2.6 standard build, the file is 1.7 Mb
and is dated 11/29/2007.  We have a copy (:slink:`%(vcredist)s`) on
our website for your convenience.  Save it to the *C:\\Python26*
directory so the installer script can find it.

Inno Setup creates the installer executable.  When installing Inno Setup,
be sure to choose the 'Install Inno Setup Preprocessor' option.

With all the pieces in place, you can run through all steps of the
build and install by changing to the top level python directory and
typing::

    python master_builder.py

This creates the redistributable installer refl1d-<version>-win32.exe for
Windows one level up in the directory tree.  In addition, source archives
in zip and tar.gz format are produced as well as text files listing the
contents of the installer and the archives.

OS/X Installer
==============

To build a Mac OS/X standalone executable you will need the py2app package.
This should already be available in your mac python environment.

Build the executable using::

    python setup_py2app

This creates a *.dmg* file in the *dist* directory with the Refl1D app
inside.