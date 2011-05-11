from __future__ import with_statement

import numpy

import dream.views
from ..mystic import monitor
from ..util import coordinated_colors
from .plot_view import PlotView
from .signal import log_message

class UncertaintyView(PlotView):
    title = "Uncertainty"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            stats = dream.views.plot_vars(history)
            pylab.draw()
            log_message(dream.views.format_vars(stats))
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.state
        self.plot()

class CorrelationView(PlotView):
    title = "Correlations"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            dream.views.plot_corrmatrix(history)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.state
        self.plot()


class TraceView(PlotView):
    title = "Parameter Trace"
    def plot(self):
        if not self.plot_state: return
        history = self.plot_state
        import pylab
        with self.pylab_interface:
            dream.views.plot_trace(history)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.state
        self.plot()
