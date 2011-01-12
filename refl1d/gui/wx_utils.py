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
This module contains GUI utility functions and classes for the application.
"""

#==============================================================================

import os
import sys

import wx
from wx.lib import delayedresult

# Text string used to compare the string width in pixels for different fonts.
# This benchmark string has 273 characters, containing 92 distinct characters
# consisting of the lowercase alpha chars in the ratio used in an English
# Scrabble(TM) set, two sets of uppercase alpha chars, two sets of digits,
# special chars with multiples of commonly used ones, and many spaces to
# approximate spacing between words in sentences and labels.
BENCHMARK_TEXT =\
"aaaaaaaaa bb cc dddd eeeeeeeeeeee ff ggg hh iiiiiiiii j k llll mm "\
"nnnnnn oooooooo pp q rrrrrr ssss tttttt uuuu vv ww x yy z "\
"ABCD EFGH IJKL MNOP QRST UVW XYZ ABCD EFGH IJKL MNOP QRST UVW XYZ "\
"01234 56789 01234 56789 "\
"...... :::: ()()() \"\",,'' ++-- **//== {}[]<> ;|~\\_ ?!@#$%^&"

# The width and height in pixels of the test string using MS Windows default
# font "MS Shell Dlg 2" and a dpi of 96.
# Note: the MS Windows XP default font has the same width and height as Tahoma.
BENCHMARK_WIDTH = 1600
BENCHMARK_HEIGHT = 14

#==============================================================================

def choose_fontsize(fontname=None):
    """
    Determines the largest font size (in points) to use for a given font such
    that the rendered width of the benchmark string is less than or equal to
    101% of the rendered width of the string on a Windows XP computer using the
    Windows default font at 96 dpi.

    The width in pixels of a rendered string is affected by the choice of font,
    the point size of the font, and the resolution of the installed font as
    measured in dots-per-inch (aka points-per-inch).
    """

    frame = wx.Frame(parent=None, id=wx.ID_ANY, title="")
    if fontname is None:
        fontname = frame.GetFont().GetFaceName()
    max_width = BENCHMARK_WIDTH + BENCHMARK_WIDTH/100

    for fontsize in xrange(12, 5, -1):
        frame.SetFont(wx.Font(fontsize, wx.SWISS, wx.NORMAL, wx.NORMAL, False,
                              fontname))
        benchmark = wx.StaticText(frame, wx.ID_ANY, label="")
        w, h = benchmark.GetTextExtent(BENCHMARK_TEXT)
        benchmark.Destroy()
        if w <= max_width: break

    frame.Destroy()
    return fontsize


def display_fontsize(fontname=None, benchmark_text=BENCHMARK_TEXT,
                                    benchmark_width=BENCHMARK_WIDTH,
                                    benchmark_height=BENCHMARK_HEIGHT):
    """
    Displays the width in pixels of a benchmark text string for a given font
    at various point sizes when rendered on the application's output device
    (which implicitly takes into account the resolution in dpi of the font
    faces at the various point sizes).
    """

    # Create a temporary frame that we will soon destroy.
    frame = wx.Frame(parent=None, id=wx.ID_ANY, title="")

    # Set the fontname if one is given, otherwise use the system default font.
    # Get the font name even if we just set it in case the specified font is
    # not installed and the system chooses another one.
    if fontname is not None:
        frame.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL, False,
                              fontname))
    fontname = frame.GetFont().GetFaceName()

    x, y = wx.ClientDC(frame).GetPPI()
    print "*** Benchmark text width and height in pixels = %4d %2d"\
          %(benchmark_width, benchmark_height)
    print "*** Compare against %s font with dpi resolution of %d:"\
          %(fontname, x)

    for fontsize in xrange(12, 5, -1):
        frame.SetFont(wx.Font(fontsize, wx.SWISS, wx.NORMAL, wx.NORMAL, False,
                              fontname))
        benchmark = wx.StaticText(frame, wx.ID_ANY, label="")
        w, h = benchmark.GetTextExtent(benchmark_text)
        benchmark.Destroy()
        print "      For point size %2d, benchmark text w, h = %4d  %2d"\
              %(fontsize, w, h)

    frame.Destroy()


def popup_error_message(caption, message):
    """Displays an error message in a pop-up dialog box with an OK button."""

    msg = wx.MessageDialog(None, message, caption, style=wx.ICON_ERROR|wx.OK)
    msg.ShowModal()
    msg.Destroy()


def popup_warning_message(caption, message):
    """Displays a warning message in a pop-up dialog box with an OK button."""

    msg = wx.MessageDialog(None, message, caption, style=wx.ICON_WARNING|wx.OK)
    msg.ShowModal()
    msg.Destroy()


def popup_information_message(caption, message):
    """Displays an informational message in a pop-up with an OK button."""

    msg = wx.MessageDialog(None, message, caption,
                           style=wx.ICON_INFORMATION|wx.OK)
    msg.ShowModal()
    msg.Destroy()


def popup_question(caption, message):
    """Displays a question in a pop-up dialog box with YES and NO buttons."""

    msg = wx.MessageDialog(None, message, caption,
                           style=wx.ICON_QUESTION|wx.YES_NO)
    msg.ShowModal()
    msg.Destroy()

#==============================================================================

class StatusBarInfo():
    """This class writes, saves, and restores multi-field status bar text."""

    def __init__(self):
        frame = wx.FindWindowByName("AppFrame", parent=None)
        self.sb = frame.GetStatusBar()
        self.cnt = self.sb.GetFieldsCount()
        self.field = []
        for index in range(self.cnt):
            self.field.append("")


    def write(self, index=0, text=""):
        # Write text to the specified slot and save text locally.
        # Beware that if you use field 0, wxPython will likely overwite it.
        if index > self.cnt - 1:
            return
        self.sb.SetStatusText(text, index)
        self.field[index] = text


    def restore(self):
        # Restore saved text from fields 1 to n.
        # Note that wxPython updates field 0 with hints and other messages.
        for index in range(1, self.cnt):
            self.sb.SetStatusText(self.field[index], index)

#==============================================================================

class ExecuteInThread():
    """
    This class executes the specified function in a separate thread and calls a
    designated callback function when the execution completes.  Control is
    immediately given back to the caller of ExecuteInThread which can execute
    in parallel in the main thread.

    Note that wx.lib.delayedresult provides a simple interface to threading
    that does not include mechanism to stop the thread.
    """

    def __init__(self, callback, function, *args, **kwargs):
        if callback is None: callback = _callback
        #print "*** ExecuteInThread init:", callback, function, args, kwargs
        delayedresult.startWorker(consumer=callback, workerFn=function,
                                  wargs=args, wkwargs=kwargs)

    def _callback(self, delayedResult):
        '''
        jobID = delayedResult.getJobID()
        assert jobID == self.jobID
        try:
            result = delayedResult.get()
        except Exception, e:
            popup_error_message(self, "job %s raised exception: %s"%(jobID, e)
            return
        '''
        return

#==============================================================================

class WorkInProgress(wx.Panel):
    """
    This class implements a rotating 'work in progress' gauge.
    """

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)

        self.gauge = wx.Gauge(self, wx.ID_ANY, range=50, size=(250, 25))

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.TimerHandler)
        #self.count = 0

    def Start(self):
        self.timer.Start(100)

    def Stop(self):
        self.timer.Stop()

    def TimerHandler(self, event):
        #self.count += 1
        #print "*** count = ", self.count
        self.gauge.Pulse()
