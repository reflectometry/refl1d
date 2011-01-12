# Copyright (C) 2006-2010, University of Maryland
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
This module implements the AuxiliaryPage class which constructs a notebook page
suitable for displaying matplotlib plots.
"""

#==============================================================================

import os
import sys
import time

import wx

import matplotlib

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure

# For use in the matplotlib toolbar.
from matplotlib.widgets import Slider, Button, RadioButtons

# Wx-Pylab magic for displaying plots within an application's window.
from matplotlib import _pylab_helpers
from matplotlib.backend_bases import FigureManagerBase

#from matplotlib import pyplot as plt
import pylab

import numpy

from common.utilities import get_appdir, log_time

from .wx_utils import StatusBarInfo

#==============================================================================

class AuxiliaryPage(wx.Panel):
    """
    This class constructs a notebook page for test purposes.
    """

    def __init__(self, parent, id=wx.ID_ANY, colour="", fignum=0, **kwargs):
        wx.Panel.__init__(self, parent, id=id, **kwargs)
        self.fignum=fignum
        self.SetBackgroundColour(colour)
        self.sbi = StatusBarInfo()

        # Create the display panel and initialize it.
        self.pan1 = wx.Panel(self, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.init_plot_panel()

        # Put the panel in a sizer attached to the main panel of the page.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.pan1, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)

        # Execute tests associated with the test tabs.
        if self.fignum == 2: test3()
        if self.fignum == 3: test4(self.figure)


    def init_plot_panel(self):
        """Initializes the main panel of the AuxiliaryPage."""

        # Instantiate a figure object that will contain our plots.
        self.figure = Figure()

        # Initialize the figure canvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self.pan1, wx.ID_ANY, self.figure)

        # Wx-Pylab magic ...
        # Make our canvas the active figure manager for pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).
        self.fm = FigureManagerBase(canvas, self.fignum)
        _pylab_helpers.Gcf.set_active(self.fm)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a vertical box sizer to manage the widgets in the main panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border=10)
        sizer.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=10)

        # Associate the sizer with its container.
        self.pan1.SetSizer(sizer)
        sizer.Fit(self.pan1)


    def OnActivePage(self):
        """This method is called when user selects (makes current) the page."""
        self.sbi.restore()

#==============================================================================

def test3():
    """
    Tests the ability to utilize code that uses the procedural interface
    to pylab to generate subplots.
    """

    pylab.suptitle("Test use of procedural interface to Pylab", fontsize=16)

    pylab.subplot(211)
    x = numpy.arange(0, 6, 0.01)
    y = numpy.sin(x**2)*numpy.exp(-x)
    pylab.xlabel("x-axis")
    pylab.ylabel("y-axis")
    pylab.title("First Plot")
    pylab.plot(x, y)

    pylab.subplot(212)
    x = numpy.arange(0, 8, 0.01)
    y = numpy.sin(x**2)*numpy.exp(-x) + 1
    pylab.xlabel("x-axis")
    pylab.ylabel("y-axis")
    pylab.title("Second Plot")
    pylab.plot(x, y)

    #pylab.show()


def test4(figure):
    """
    Tests the ability to utilize code that uses the object oriented interface
    to pylab to generate subplots.
    """

    axes = figure.add_subplot(311)
    x = numpy.arange(0, 6, 0.01)
    y = numpy.sin(x**2)*numpy.exp(-x)
    axes.plot(x, y)

    axes = figure.add_subplot(312)
    x = numpy.arange(0, 8, 0.01)
    y = numpy.sin(x**2)*numpy.exp(-x) + 1
    axes.plot(x, y)
    axes.set_ylabel("y-axis")

    axes = figure.add_subplot(313)
    x = numpy.arange(0, 4, 0.01)
    y = numpy.sin(x**2)*numpy.exp(-x) + 2
    axes.plot(x, y)
    axes.set_xlabel("x-axis")

    pylab.suptitle("Test use of object oriented interface to Pylab",
                   fontsize=16)
    pylab.subplots_adjust(hspace=0.35)
    #pylab.show()
