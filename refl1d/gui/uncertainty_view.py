from __future__ import with_statement

import numpy

import dream.views
from ..mystic import monitor
from ..util import coordinated_colors
from .. import errors
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
            # TODO: separate calculation of parameter uncertainty from plotting
            self.model.parameter_uncertainty = stats
            log_message(dream.views.format_vars(stats))
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.plot_state = event.uncertainty_state
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
        self.plot_state = event.uncertainty_state
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
        self.plot_state = event.uncertainty_state
        self.plot()

class ErrorView(PlotView):
    title = "Profile Uncertainty"
    def plot(self):
        if not self.plot_state: return
        import pylab
        with self.pylab_interface:
            pylab.clf()
            errors.show_distribution(*self.plot_state)
            pylab.draw()
    def OnFitProgress(self, event):
        if event.problem != self.model: return
        self.new_state(event.problem, event.uncertainty_state)
    def new_state(self, problem, state):
        # Should happen in a separate process
        self.plot_state = errors.calc_distribution_from_state(problem, state)
        self.plot()
