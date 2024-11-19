.. _installing:

*****************
Installing Refl1D
*****************

.. contents:: :local:

| Windows installer: :slink:`%(winexe)s`
| Source: :slink:`%(srczip)s`


Installing Latest Release
=========================

A Windows installer is available from `github <https://github.com/reflectometry/refl1d/releases/latest>`_.
The file `Refl1D-VERSION-exe.zip` contains python, the application, the
supporting libraries and everything else needed to run the application.

To install, download and extract the zip file. Go to the extracted directory
and click on `refl1d_gui.bat`. This will open a dialog saying that the
application is untrusted with a "Don't run" button at the bottom. Click
on "more info" and a "Run anyway" button will appear. For command line
operation, open a `cmd` command terminal, change to the extracted directory
and type `refl1d.bat`.

The installed python is a full version of python. If your specialized
reflectometry models need additional python packages, then you can
use `python -m pip` in the extracted directory to install them.

Refl1D is also available on all platforms from PyPI using pip::

    pip install refl1d
    
    # if you also want to run the webview, an optional extra is available
    pip install refl1d[webview]


Installing from source
======================

Requirements
------------

    - `Python <https://www.python.org/downloads/>`_ >= 3.10 
    - git

You can either use Python from the `python.org <https://www.python.org/downloads/>_` site, or if you prefer, 
you can use miniforge. Miniforge is an open version of the `conda` program, and is pre-configured to use the `conda-forge` repository.
This repo is the most useful for open scientific application development.

You can get an installer for your system (MacOS, Windows or Linux) here: https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3

Run the installer, then from a terminal window (PowerShell or Terminal in Windows) run::

    conda init 

then close and re-open your terminal.


Setup Environment
-----------------

You can create a new virtual environment for Refl1D, or install it in your base environment.  To create a new environment, run::

    # if using miniforge
    conda create -n refl1d
    conda activate refl1d

    # if using regular python
    python -m venv refl1d
    source refl1d/bin/activate


Installation
------------

To install the application from source, clone the repository and install the
dependencies::

    git clone https://github.com/reflectometry/refl1d.git
    cd refl1d
    pip install .

If you want to run the webview, you can install the optional extra::

    pip install .[webview]


Installing for Development
==========================

Refl1d depends closely on the `bumps <https://github.com/bumps/bumps>`_,
which also goes through frequent development. If you are also working with the
scattering length density calculator or the fitting engine, or if you need the 
latest unreleased version of bumps, you may want to install bumps from source.
Clone the bumps repository and install from source in your refl1d virtual environment::

    git clone https://github.com/bumps/bumps.git
    pip install -e ./bumps


Python Environment
------------------

If you are planning to contribute to the project, you will want to install
the package in development mode, including the dev dependencies::

    pip install -e .[dev]

    # or if you plan to develop the webview
    pip install -e .[dev,webview]

This will install the package in development mode, so that changes you make
to the source code will be reflected in the installed package.  It will also
install the development dependencies, which include the testing framework
and other tools used in the development process.

If you are not planning to develop the Vue TS webview client, you can now run the application with::

    refl1d-webview --port 8080

which will automatically open a browser window to the webview.

Javascript Environment
----------------------

If you are planning to develop the webview (client), you will need to install
a Javascript environment.

* `Node.js <https://nodejs.org/en/download/>`_ can be installed from the website, or using conda::

    conda install -c conda-forge nodejs

* `Bun <https://bun.sh/>`_ is a fast-performing drop-in replacement for npm, and is available on all platforms. 

Similar to the Python environment, you may want to install and link the ``bumps-webview-client`` in your Refl1d Javascript environment::

    cd /path/to/bumps/bumps/webview/client
    npm install # or bun install
    npm link    # or bun link

    cd /path/to/refl1d/refl1d/webview/client
    npm install                     # or bun install
    npm link bumps-webview-client   # or bun link bumps-webview-client

To build the client, run::

    cd /path/to/refl1d/refl1d/webview/client
    npm run build   # or bun build

If you are developing the client, you can run the client in development mode.
In this mode, any changes to client code are immediately reflected in a connected running client::

    npm run dev   # or bun run dev

This starts the client and shows the URL to connect to in the terminal (typically http://localhost:5173).

Now, you can start the Python webview server with::

    refl1d-webview --headless --port 8080

and point the client to the server with the `?server=localhost:8080` query string, e.g.

    http://localhost:5173/?server=localhost:8080
