.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

| Windows installer: :slink:`%(winexe)s`
| Source: :slink:`%(srczip)s`

Recent versions of the Refl1D application are available for Windows
from `github <https://github.com/reflectometry/refl1d/releases/latest>`_.
The file `Refl1D-VERSION-exe.zip` contains python, the application, the
supporting libraries, and everything else needed to run the application.

To install, download and extract the zip file. Go to the extracted directory
and click on `refl1d_gui.bat`. This will open a dialog saying that the
application is untrusted with a "Don't run" button at the bottom. Click
on "more info" and a "Run anyway" button will appear. For command line
operation, open a `cmd` command terminal, change to the extracted directory
and type `refl1d.bat`.

The installed python is a full version of python. If your specialized
reflectometry models need additional python packages, then you can
use `python -m pip` in the extracted directory to install them.

Linux users will need to install from using pip::

    pip install refl1d wxpython

Note that the binary versions will lag the release version until the release
process is automated.  Windows and Mac users may want to install using pip as
well to get the version with the latest
`changes <https://github.com/reflectometry/refl1d/blob/master/CHANGES.rst>`_.

Installing from source
======================

Installing the application from source requires a working python environment.
See below for operating system specific instructions.

Our base scientific python environment contains the following packages as
well as their dependencies:

    - python 3
    - numpy
    - scipy
    - matplotlib
    - wxpython

Once your environment is in place, you can install directly from PyPI
using pip::

    pip install refl1d

This will install refl1d, bumps and periodictable.

You can run the program by typing:

    python -m refl1d.main

If this fails, then follow the instructions in :ref:`contributing` to install
from the source archive directly.

Windows
-------

There are couple of options for setting up a python environment on windows:

  - `python.org <https://www.python.org/>`_, and
  - `Anaconda <https://www.anaconda.com/distribution/>`_.

With most pypi packages now bundled with wheels, it is now easy to set up a
development environment using the official python package.  Similarly,
anaconda provides binaries for all the refl1d dependencies.

You will need a C/C++ compiler.  If you already have Microsoft Visual C
installed you are done. If not, you can use the MinGW compiler that is supplied
with your python environment or download your own.  You can set MinGW
as the default compiler by creating the file *Lib\distutils\\distutils.cfg*
in your python directory (e.g., *C:\\Python3.9*) with the following content::

    [build]
    compiler=mingw32

Once the python is prepared, you can install the periodic table and bumps
package using the Windows console.

Linux
-----

Linux distributions will provide the base required packages.  You
will need to refer to your distribution documentation for details.

On debian/ubuntu, the command will be something like::

    sudo apt-get install python3-{matplotlib,numpy,scipy,wxgtk4.0,pyparsing,setuptools}

For development you also want nose and sphinx::

    sudo apt-get install python-{nose,sphinx}

Latex is needed to build the pdf documentation.

OS/X
----

Similar to windows, you can install the official python distribution or
use Anaconda.  You will need to install the Xcode command line utilities
to get the compiler.

To run the interactive interface on OS/X you may need to use::

    pythonw -m refl1d.main --edit
