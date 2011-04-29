import wx
import wx.lib.newevent
import time

from copy import deepcopy
from threading import Thread
from ..mystic import monitor
from ..fitters import FitDriver
from ..mapper import  MPMapper, SerialMapper

#==============================================================================

PROGRESS_DELAY = 5
IMPROVEMENT_DELAY = 5

(FitProgressEvent, EVT_FIT_PROGRESS) = wx.lib.newevent.NewEvent()
(FitImprovementEvent, EVT_FIT_IMPROVEMENT) = wx.lib.newevent.NewEvent()
(FitCompleteEvent, EVT_FIT_COMPLETE) = wx.lib.newevent.NewEvent()

# NOTE: GUIMonitor is running in a separate thread.  It should not touch the
# problem internals.
class GUIMonitor(monitor.TimedUpdate):
    def __init__(self, win, problem, progress=None, improvement=None):
        improvement = improvement if improvement else IMPROVEMENT_DELAY
        progress = progress if progress else PROGRESS_DELAY
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.win = win
        self.problem = problem

    def show_progress(self, history):
        evt = FitProgressEvent(problem=self.problem,
                               step=history.step[0],
                               value=history.value[0],
                               point=history.point[0])
        wx.PostEvent(self.win, evt)

    def show_improvement(self, history):
        evt = FitImprovementEvent(problem=self.problem,
                                  step=history.step[0],
                                  value=history.value[0],
                                  point=history.point[0])
        wx.PostEvent(self.win, evt)

#==============================================================================

class FitThread(Thread):
    """Run the fit in a separate thread from the GUI thread."""
    def __init__(self, win, problem=None,
                 fitter=None, options=None, mapper=None):
        # base class initialization
        #Process.__init__(self)

        Thread.__init__(self)
        self.win = win
        self.problem = problem
        self.fitter = fitter
        self.options = options
        self.mapper = mapper
        self.start() # Start it working.

    def run(self):
        # NOTE: Problem must be the original problem (not a copy) when used
        # inside the GUI monitor otherwise AppPanel will not be able to
        # recognize that it is the same problem when updating views.
        monitors = [GUIMonitor(self.win, self.problem)]
        # Serial mapper needs a copy of the problem because it is running in a
        # separate thread with shared memory.  However, the multiprocessing
        # mapper runs in a different process context.
        if True: # Multiprocessing parallel
            mapper = MPMapper
            problem = self.problem
        else:
            mapper = SerialMapper
            problem = deepcopy(self.problem)

        driver = FitDriver(self.fitter, problem=problem,
                           monitors=monitors, **self.options)

        self.fitter.mapper = mapper.start_mapper(problem, [])

        x,fx = driver.fit()
        evt = FitCompleteEvent(problem=problem,
                               point=x,
                               value=fx)
        wx.PostEvent(self.win, evt)
