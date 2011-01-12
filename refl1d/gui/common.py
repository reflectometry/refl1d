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
This module contains common utility functions and classes for the application.
"""

#==============================================================================

import os
import sys
import time

#==============================================================================

def get_appdir():
    """
    Returns the directory path of the main module of the application, i.e, the
    root directory from which the application was started.  Note that this may
    be different than the current working directory.
    """

    if hasattr(sys, "frozen"):  # check for py2exe image
        path = sys.executable
    else:
        path = sys.argv[0]
    return os.path.dirname(os.path.abspath(path))

#==============================================================================

log_time_handle = None  # global variable for holding TimeStamp instance handle

def log_time(text=None, reset=False):
    """
    This is a convenience function for using the TimeStamp class from any
    module in the application for logging elapsed and delta time information.
    This data is prefixed by a timestamp and optionally suffixed by a comment.
    log_time maintains a single instance of TimeStamp during program execution.
    Example output from calls to log_time('...'):

    ==>     0.000s   0.000s  Starting KsRefl
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
    # Test the TimeStamp class and the convenience function.
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
