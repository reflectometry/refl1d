import wx
import wx.lib.newevent
import time
from threading import *

EVT_RESULT_ID = 1

class ResultEvent(wx.PyEvent):

    def __init__(self, data):

        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

# Thread class that executes processing
class Worker(Thread):
    """Worker Thread Class."""
    def __init__(self, panel, problem, fn, pars=None, mapper = None,):
        """Init Worker Thread Class."""
        # base class initialization
        #Process.__init__(self)

        Thread.__init__(self)
        self.panel = panel
        self.problem = problem
        self.fitter = fn
        self.pars = pars
        self.mapper = mapper
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        self.start()

    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread. Simulation of
        # a long process (well, 10s here) as a simple loop - you will
        # need to structure your processing so that you periodically
        # peek at the abort variable
        self.fitter.mapper = self.mapper.start_mapper(self.problem, self.pars)
        best = self.fitter.fit()
        wx.PostEvent(self.panel, ResultEvent(best))
