#!/usr/bin/env python

# Copyright (C) 2006-2011, University of Maryland
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: James Krycka

"""
This script starts the graphical user interface of the Refl1D Reflectometry
Modeler application.
"""

#==============================================================================

import os
import sys

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
if not hasattr(sys, 'frozen'):
    path = os.path.realpath(__file__)
    root = os.path.abspath(os.path.join(os.path.dirname(path), '..'))
    sys.path.insert(0, root)
    sys.path.insert(1, os.path.join(root, 'dream'))

# ========================== Start program ====================================

if __name__ == "__main__":
    import refl1d.gui.gui_app
    refl1d.gui.gui_app.main()
