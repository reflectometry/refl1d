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
This module contains utility functions and classes for the application.
"""

#==============================================================================

import os
import sys
import time

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


def get_appdir():
    """
    Returns the path of the directory that contains the application being run
    (i.e., the directory path of the executing python script or frozen image).
    Note that this path may be different than the current working directory.
    """

    if hasattr(sys, "frozen"):  # check for py2exe image
        path = sys.executable
    else:
        path = sys.argv[0]
    return os.path.dirname(os.path.abspath(path))


def get_rootdir(subdirlevel=0):
    """
    Returns the path of the root directory of the package from which the
    application is running (i.e., the path of the top-level directory of the
    python package or the directory containing the frozen image being run).
    If subdirlevel = 0 then the script being run is assumed to be located in
    the top-level directory of the package, otherwise n levels down.  For
    example, if the script is in <package>/bin, subdirlevel should be 1.
    """

    path = get_appdir()
    if hasattr(sys, "frozen"):  # check for py2exe image
        return path
    if subdirlevel > 0:
        for x in range(subdirlevel):
            path = os.path.abspath(os.path.join(path, '..'))
    return path


def get_rootdir_parent(subdirlevel=0):
    """
    Returns the path of the parent directory of the package from which the
    application is running (i.e., the path one level above the top-level
    directory of the package or returns None if a frozen image is being run).
    If subdirlevel = 0 then the script being run is assumed to be located in
    the top-level directory of the package, otherwise n levels down.  For
    example, if the script is in <package>/bin, subdirlevel should be 1.
    """

    if hasattr(sys, "frozen"):  # check for py2exe image
        return None
    return os.path.abspath(os.path.join(get_rootdir(subdirlevel), '..'))


def get_bitmap(filename, imgType=wx.BITMAP_TYPE_PNG):
    """
    Returns the bitmap from an image file (png, jpg, ico)
    """

    # TODO: This code was borrowed from KsRefl - pdir lookup needs fixing.
    imgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    if not os.path.exists(imgdir):  # for py2exe
        pdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        imgdir = os.path.join(pdir, 'images')

    fullname = os.path.join(imgdir, filename)

    return wx.BitmapFromImage(wx.Image(fullname, imgType).Scale(16, 16))


def popup_error_message(caption, message):
    """Displays an error message in a pop-up dialog box with an OK button."""

    msg = wx.MessageDialog(None, message, caption, style=wx.ICON_ERROR|wx.OK)
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


def popup_warning_message(caption, message):
    """Displays a warning message in a pop-up dialog box with an OK button."""

    msg = wx.MessageDialog(None, message, caption, style=wx.ICON_WARNING|wx.OK)
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

#==============================================================================

log_time_handle = None  # global variable for holding TimeStamp instance handle

def log_time(text=None, reset=False):
    """
    This is a convenience function for using the TimeStamp class from any
    module in the application for logging elapsed and delta time information.
    This data is prefixed by a timestamp and optionally suffixed by a comment.
    log_time maintains a single instance of TimeStamp during program execution.
    Example output from calls to log_time('...'):

    ==>     0.000s   0.000s  Starting application
    ==>     0.031s   0.031s  Starting to display the splash screen
    ==>     1.141s   1.110s  Starting to build the GUI on the frame
    ==>     1.422s   0.281s  Done initializing - entering the event loop
    """

    global log_time_handle
    if log_time_handle is None:
        log_time_handle = TimeStamp()
    if reset:
        log_time_handle.reset()
    log_time_handle.log_interval(text=text)


class TimeStamp():
    """
    This class provides timestamp, delta time, and elapsed time services for
    displaying wall clock time usage by the application.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        # Starts new timing interval.
        self.t0 = self.t1 = time.time()


    def gettime3(self):
        # Gets current time in timestamp, delta time, and elapsed time format.
        now = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        elapsed = now - self.t0
        delta = now - self.t1
        self.t1 = now
        return timestamp, delta, elapsed


    def gettime2(self):
        # Gets current time in delta time and elapsed time format.
        now = time.time()
        elapsed = now - self.t0
        delta = now - self.t1
        self.t1 = now
        return delta, elapsed


    def log_time_info(self, text=""):
        # Prints timestamp, delta time, elapsed time, and optional comment.
        t, d, e = self.gettime3()
        print "==> %s%9.3fs%9.3fs  %s" %(t, d, e, text)


    def log_timestamp(self, text=""):
        # Prints timestamp and optional comment.
        t, d, e = self.gettime3()
        print "==> %s  %s" %(t, text)


    def log_interval(self, text=""):
        # Prints elapsed time, delta time, and optional comment.
        d, e = self.gettime2()
        print "==>%9.3fs%9.3fs  %s" %(d, e, text)

#==============================================================================

if __name__ == '__main__':
    # Test the display_fontsize and choose_fontsize functions.
    app = wx.PySimpleApp()
    print "For Arial font:"
    display_fontsize(fontname="Arial")
    print "Calculated font size =", choose_fontsize(fontname="Arial")
    app.Destroy()

    # Test get_appdir(), get_rootdir(), and get_rootdir_parent() functions.
    print ""
    print "Assuming this script lives in the top-level directory of <package>:"
    print "  Application directory is:    ", get_appdir()
    print "  Package root directory is:   ", get_rootdir()
    print "  Parent of root directory is: ", get_rootdir_parent()
    print "Assuming this script lives 1 subdirectory level below <package>:"
    print "  Application directory is:    ", get_appdir()
    print "  Package root directory is:   ", get_rootdir(subdirlevel=1)
    print "  Parent of root directory is: ", get_rootdir_parent(subdirlevel=1)
    print "Assuming this script lives 2 subdirectory levels below <package>:"
    print "  Application directory is:    ", get_appdir()
    print "  Package root directory is:   ", get_rootdir(subdirlevel=2)
    print "  Parent of root directory is: ", get_rootdir_parent(subdirlevel=2)

    # Test the TimeStamp class and the convenience function.
    print ""
    log_time("Using log_time() function")
    print "Sleeping for 0.54 seconds ..."
    time.sleep(0.54)
    log_time("Using log_time() function")
    print "Sleeping for 0.83 seconds ..."
    time.sleep(0.83)
    log_time("Using log_time() function")
    print "Creating an instance of TimeStamp (as the second timing class)"
    ts = TimeStamp()
    print "Sleeping for 0.66 seconds ..."
    time.sleep(0.66)
    ts.log_time_info(text="Using log_time_info() method")
    ts.log_timestamp(text="Using log_timestamp() method")
    ts.log_interval(text="Using log_interval() method")
    print "Sleeping for 0.35 seconds ..."
    time.sleep(0.35)
    ts.log_interval(text="Using log_interval() method")
    print "Sleeping for 0.42 seconds ..."
    time.sleep(0.42)
    ts.log_interval(text="Using log_interval() method")
    print "Resetting the clock ..."
    ts.reset()
    ts.log_interval(text="Using log_interval() method")
    print "Sleeping for 0.33 seconds ..."
    time.sleep(0.33)
    ts.log_interval(text="Using log_interval() method")
    print "Switch back to the first timing class"
    log_time("Using log_time() function")
