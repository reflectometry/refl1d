#!/usr/bin/env python
# This program is in the public domain.
# Authors: Paul Kienzle and James Krycka

"""
This script starts the command line interface of the Refl1D Reflectometry
Modeler application to process the command just entered.
"""

# ========================== Path setup =====================================
# When this script is run interactively (i.e., from a Python command prompt),
# sys.path needs to be updated for some imports to work, namely 'from refl1d'
# and 'from dream'.  However, when this module is executed from a frozen image,
# sys.path will be automatically setup to include the full path of the frozen
# image and python will be able to perform the aforementioned imports.  Thus,
# for the interactive case, we will augment sys.path.  Assumptions:
#   <root> is the top-level directory of the package, it can have any name
#   script dir -> <root>/bin
#   'from refl1d' -> <root>/refl1d
#   'from dream' -> <root>/dream/dream
# Although <root> is currently named 'refl1d', it does not have an__init__.py.
# Likewise, <root>/dream does not have an __init__.py file.
#import os, sys
#if not hasattr(sys, 'frozen'):
#    path = os.path.realpath(__file__)
#    root = os.path.abspath(os.path.join(os.path.dirname(path), '..'))
#    sys.path.insert(0, root)
#    sys.path.insert(1, os.path.join(root, 'dream'))

# ========================== Start program ==================================
# Process the command line that has been entered.
if __name__ == "__main__":
    # This is necessary when running the application from a frozen image and
    # using the --parallel option.  Note that freeze_support() has no effect
    # when running from a python script (i.e., in a non-frozen environment).
    import multiprocessing
    multiprocessing.freeze_support()

    import refl1d.fitplugin
    import bumps.cli
    bumps.cli.install_plugin(refl1d.fitplugin)
    bumps.cli.main()
