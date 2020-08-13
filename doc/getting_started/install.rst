.. _installing:

**************************
Installing the application
**************************

.. contents:: :local:

Recent versions of the Refl1D application are available for windows and mac
from `<http://www.ncnr.nist.gov/reflpak>`_. The installer walks through the
steps of setting the program up to run on your machine and provides the
sample data used in the tutorial.

Linux users will need to install from using pip::

    pip install refl1d wxpython

Note that the binary versions will lag the release version until the release
process is automated.  Windows and Mac users may want to install using pip as
well to get the version with the latest
`changes <https://github.com/reflectometry/refl1d/blob/master/CHANGES.rst>`_.

..
    - Windows installer: :slink:`%(winexe)s`
    - Apple installer: :slink:`%(macapp)s`
    - Source: :slink:`%(srczip)s`

Installing from source
======================

Installing the application from source requires a working python environment.
See below for operating system specific instructions.

Our base scientific python environment contains the following packages.
The versions listed are a snapshot of our current configuration, though
both older and more recent versions are likely to work:

    - python 2.7
    - matplotlib 1.3.1
    - numpy 1.9.0
    - scipy 0.14.0
    - wxPython 2.9.5.0
    - setuptools 7.0
    - pyparsing 1.5.6
    - pip 1.4.1

Python 3.3/3.4 will work for batch processing, but wxPython is not yet
supported.

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
in your python directory (e.g., *C:\\Python2.7*) with the following content::

    [build]
    compiler=mingw32

Once the python is prepared, you can install the periodic table and bumps
package using the Windows console.  To start the console, click the "Start"
icon on your task bar and select "Run...".  In the Run box, type "cmd".

Linux
-----

Linux distributions will provide the base required packages.  You
will need to refer to your distribution documentation for details.

On debian/ubuntu, the command will be something like::

    sudo apt-get install python-{matplotlib,numpy,scipy,wxgtk2.8,pyparsing,setuptools}

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
