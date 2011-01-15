.. _starting_intro:

####################
Refl1D - an overview
####################


Refl1D provides multiple ways of creating and adjusting the model,
and simultaneously showing the degree of fit between the 
experimental data and the theoretical data graphically. 
This package comes with an optimization algorithm which 
adjusts the variables of the model yielding the best fit to the data.

###########################
Installing the application
###########################

Refl1D |version| is provided as a |winexe|.  This installer walks through
the steps of setting the program up to run on your machine and provides
the sample data to be used in the tutorial.

Installers for other platforms are not yet available.

###########################
Building from source
###########################

Building the application from source requires some preparation.  

First you will need to set up your python environment.  We depend on
numerous external packages.  The versions listed below are a snapshot 
of a configuration that we are using. Both older or more recent versions 
may work.

Our base scientific python environment contains:

	Python 2.6  (not 3.x)
	Matplotlib 1.0.0
	Numpy 1.3	
	Scipy 0.7.0
	WxPython 2.8.11.0
	SetupTools 0.6c9
	gcc 3.4.4
	PyParsing 1.5.5
	Periodictable 1.3
	
To run tests you will need:

	Nose 0.10 - 1.0

To build the documentation you will need:

	Sphinx 1.0.4
	DocUtils 0.5
	Pyments 1.0
	Jinja2 2.2
	`MathJax <http://www.mathjax.org/download/>`_ 1.0.1 (for HTML)

You will also need to download the application |srczip|, or use the
code repository (see the contributing_ section for details).

Platform specific details for setting up your environment are given below.

Windows
-------

The `Python(X,Y) <http://pythonxy.org>` package contains most of the 
pieces required to build the application.  You can select "Full Install"
for convenience, or you can select "Custom Install" and make sure
the above packages are selected.  In particular, wx is not selected
by default.  Be sure to select py2exe as well, since you may want to
build a self contained release package.

The Python(x,y) package supplies a C/C++ compiler, but the package does 
not set it as the default compiler.  To do this you will need to create 
*C:\\Python26\\Lib\\distutils\\distutils.cfg* with the following content::

	[build]
	compiler=mingw32
Once python is prepared, you can install the periodic table package using
the Windows console.  To start the console, click the "Start" icon on your
task bar and select "Run...".  In the Run box, type "cmd".  Enter the
following in the console:

	python -m easy_install periodictable

This should install periodictable and pyparsing.

Next change to the directory containing the source.  This will be a command
like the following:

    cd "C:\Documents and Settings\<username>\My Documents\refl1d-src"
    
Now type the command to build and install refl1d:

    python setup.py install

Now change to your data directory:

	cd "C:\Documents and Settings\<username>\My Documents\data"
	
To run the program use:  *** Not yet implemented ***

	python -m refl1d.run -h

Linux
-----

Many linux distributions will provide the base required packages.  You
will need to refer to your distribution documentation for details.

On ubuntu you can use apt-get to install matplotlib, numpy, scipy, wx,
nose and sphinx.

From a terminal, change to the directory containing the source and type:

	python -m easy_install periodictable
	python setup.py install

This should install the reflfit file somewhere on your path.

To run the program use:

	reflfit	-h

OS/X
----

Building a useful python environment on OS/X is somewhat involved, and
this documentation will be expanded to provide more detail.

You will need to download python, numpy, scipy, wx and matplotlib
packages from their respective sites (use the links above). Setuptools 
will need to be installed by hand.

From a terminal, change to the directory containing the source and type:

	python -m easy_install periodictable nose sphinx
	python setup.py install

This should install the reflfit file somewhere on your path.

To run the program use:

	reflfit	-h


###########################
Windows Standalone Package
###########################

To build a windows standalone executable with py2exe you may first need
to create an empty file named
*C:\\Python26\\Lib\\numpy\\distutils\\tests\\__init__.py*.
Without this file, py2exe raises an error when it is searching for
the parts of the numpy package.  This may be fixed on recent versions
of numpy.

Next, please update the __version__ tag in refl1d/__init__.py to mark 
it as your own.  Now you can build the standalone executable using:

    python setup_py2exe

Copy the entire dist directory to the target computer.  The
file dist/refl1d.exe should now run.  If it fails with an error about
part of the installation missing, you may need the following:

	Visual C++ 2008 Redistributable Package (x86) 11/29/2007

This package, available as vcredist_x86.exe from the
`Microsoft Download Center <http://www.microsoft.com/downloads/>`_,
contains the Microsoft C++ runtime library needed to run the Python 
interpreter.  Be careful to select the version that corresponds to 
the one used to build Python --- different versions have different names.
For the Python 2.6 standard build, the file is 1.7 Mb and is 
dated 11/29/2007. We have placed a 
`copy <http://www.reflectometry.org/danse/download.php?file=vcredist_x86.exe>`_ 
on our website for your convenience.  Click on vcredist_x86.exe on the
target computer and the file dist/refl1d.exe should now run.

To build the Windows installer, even more effort is needed.  First,
move the above vcredist_x86.exe to the C:\Python26 directory so the
installer can find it.  Next download Inno Settup:

	`Inno Setup <http://www.jrsoftware.org/isdl.php>` 5.3.10 QuickStart Pack

When installing Inno Setup, be sure to choose the 'Install Inno Setup
Preprocessor' option.

Be sure to install sphinx if you have not already done so by running the
python xy installer again.

You should now be able to run the following from the top level python
directory:

	python master_builder.py
	
This creates the redistributable install refl1d-<version>.exe one level
up in the directory tree.
