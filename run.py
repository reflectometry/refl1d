#!/usr/bin/env python
"""
Build and run refl1d.

Usage:

./run.py [refl1d cli args]
"""

import os
import sys
from os.path import abspath, join as joinpath, dirname
import traceback
import warnings


# From mgab at https://stackoverflow.com/a/22376126/6195051
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def addpath(path):
    """
    Add a directory to the python path environment, and to the PYTHONPATH
    environment variable for subprocesses.
    """
    path = abspath(path)
    if "PYTHONPATH" in os.environ:
        PYTHONPATH = path + os.pathsep + os.environ["PYTHONPATH"]
    else:
        PYTHONPATH = path
    os.environ["PYTHONPATH"] = PYTHONPATH
    sys.path.insert(0, path)


from contextlib import contextmanager


@contextmanager
def cd(path):
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)


def prepare_environment():
    # sys.dont_write_bytecode = True

    # Make sure that we have a private version of mplconfig
    # mplconfig = joinpath(os.getcwd(), '.mplconfig')
    # os.environ['MPLCONFIGDIR'] = mplconfig
    # if not os.path.exists(mplconfig):
    #    os.mkdir(mplconfig)

    # import numpy; numpy.seterr(all='raise')
    refl1d_root = abspath(dirname(__file__))
    periodictable_root = abspath(joinpath(refl1d_root, "..", "periodictable"))
    bumps_root = abspath(joinpath(refl1d_root, "..", "bumps"))

    # add bumps and periodictable to the path
    addpath(periodictable_root)
    addpath(bumps_root)

    # Add the build dir to the system path
    addpath(refl1d_root)

    # Make sample data and models available
    os.environ["REFL1D_DATA"] = joinpath(refl1d_root, "doc", "examples")

    # print "\n".join(sys.path)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    prepare_environment()
    import refl1d.main

    refl1d.main.cli()
