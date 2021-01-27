from __future__ import with_statement

import wx
# Can't seem to detect when notebook should be drawn on Mac
IS_MAC = (wx.Platform == '__WXMAC__')

from numpy import inf

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from bumps.gui.util import EmbeddedPylab
from bumps.fitproblem import MultiFitProblem

from refl1d.probe import Probe


# ------------------------------------------------------------------------
class DataView(wx.Panel):
    title = 'Reflectivity'
    default_size = (600,400)
    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # Instantiate a figure object that will contain our plots.
        figure = Figure(figsize=(1,1), dpi=72)

        # Initialize the figure canvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas an active figure manager for pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).  Use _activate_figure() to set.
        self.pylab_interface = EmbeddedPylab(canvas)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a vertical box sizer to manage the widgets in the main panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=0)
        sizer.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=0)

        # Associate the sizer with its container.
        self.SetSizer(sizer)
        sizer.Fit(self)

        self.view = Probe.view

        self._need_redraw = False
        self.Bind(wx.EVT_SHOW, self.OnShow)
        self._calculating = False
        self.toolbar = mpl_toolbar

    def menu(self):
        # Add 'View' menu to the menu bar and define its options.
        # Present y-axis plotting scales as radio buttons.
        # Grey out items that are not currently implemented.
        frame = wx.GetTopLevelParent(self)
        menu = wx.Menu()
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Fresnel",
                                          "Plot R/R_F")
        frame.Bind(wx.EVT_MENU, self.OnFresnel, _item)
        _item.Check(Probe.view == 'fresnel')
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "Log Fresnel",
                                          "Plot log R/R_F")
        frame.Bind(wx.EVT_MENU, self.OnLogFresnel, _item)
        _item.Check(Probe.view == 'logfresnel')
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "Li&near",
                                          "Plot linear R")
        frame.Bind(wx.EVT_MENU, self.OnLinear, _item)
        _item.Check(Probe.view == 'linear')
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Log",
                                          "Plot log R")
        frame.Bind(wx.EVT_MENU, self.OnLog, _item)
        _item.Check(Probe.view == 'log')
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Q4",
                                          "Plot R * Q^4")
        frame.Bind(wx.EVT_MENU, self.OnQ4, _item)
        _item.Check(Probe.view == 'q4')
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&SA",
                                          "Plot spin asymmetry")
        frame.Bind(wx.EVT_MENU, self.OnSA, _item)
        _item.Check(Probe.view == 'SA')

        menu.AppendSeparator()

        _item = menu.Append(wx.ID_ANY,
                                 "&Residuals",
                                 "Plot residuals (R_theory - R)/dR")
        frame.Bind(wx.EVT_MENU, self.OnResiduals, _item)
        menu.Enable(id=_item.GetId(), enable=True)

        return menu

    # ==== Views ====
    def OnLog(self, event):
        self.view = "log"
        self.redraw()

    def OnLinear(self, event):
        self.view = "linear"
        self.redraw()

    def OnFresnel(self, event):
        self.view = "fresnel"
        self.redraw()

    def OnLogFresnel(self, event):
        self.view = "logfresnel"
        self.redraw()

    def OnQ4(self, event):
        self.view = "q4"
        self.redraw()

    def OnSA(self, event):
        self.view = "SA"
        self.redraw()

    def OnResiduals(self, event):
        self.view = "residuals"
        self.redraw()

    # ==== Model view interface ===
    def OnShow(self, event):
        #print "theory show"
        if not event.Show: return
        #print "showing theory"
        if self._need_redraw:
            #print "-redraw"
            self.redraw()

    def get_state(self):
        return self.problem

    def set_state(self, state):
        self.set_model(state)

    def set_model(self, model):
        # print ">>>>>>> refl1d data set model"
        self.problem = model
        self.redraw(reset=True)

    def update_model(self, model):
        # print ">>>>>>> refl1d data update model"
        if self.problem == model:
            self.redraw()

    def update_parameters(self, model):
        # print ">>>>>>> refl1d data update parameters"
        if self.problem == model:
            self.redraw()
    # =============================

    def redraw(self, reset=False):
        # Hold off drawing until the tab is visible
        if not IS_MAC and not self.IsShown():
            self._need_redraw = True
            return
        #print "drawing theory"

        if self._calculating:
            # That means that I've entered the thread through a
            # wx.Yield for the currently executing redraw.  I need
            # to cancel the running thread and force it to start
            # the calculation over.
            self._cancel_calculate = True
            #print "canceling calculation"
            return

        self._need_redraw = False
        self._calculating = True

        # Calculate reflectivity
        #print "calling again"
        while True:
            #print "restarting"
            # We are restarting the calculation, so clear the reset flag
            self._cancel_calculate = False

            # Preform the calculation
            if isinstance(self.problem,MultiFitProblem):
                #print "n=",len(self.problem.models)
                for p in self.problem.models:
                    if hasattr(p.fitness, 'reflectivity'):
                        self._precalc(p.fitness)
                    #print "cancel",self._cancel_calculate,"reset",p.fitness.is_reset()
                        if p.fitness.is_reset() or self._cancel_calculate: break
                if self._cancel_calculate \
                    or self.problem.active_model.fitness.is_reset(): continue
            else:
                self._precalc(self.problem.fitness)
                if self._cancel_calculate \
                    or self.problem.fitness.is_reset(): continue

            # Redraw the canvas with newly calculated reflectivity
            with self.pylab_interface:
                ax = plt.gca()
                #print "reset",reset, ax.get_autoscalex_on(), ax.get_xlim()
                reset = reset or ax.get_autoscalex_on()
                range_x = ax.get_xlim()
                #print "composing"
                plt.clf() # clear the canvas
                #shift=20 if self.view == 'log' else 0
                shift=0
                if isinstance(self.problem,MultiFitProblem):
                    for _,p in enumerate(self.problem.models):
                        if hasattr(p.fitness, 'reflectivity'):
                            p.fitness.plot_reflectivity(view=self.view,
                                                        plot_shift=shift)
                            if self._cancel_calculate or p.fitness.is_reset(): break
                    if self._cancel_calculate \
                        or self.problem.active_model.fitness.is_reset(): continue
                else:
                    self.problem.fitness.plot_reflectivity(view=self.view,
                                                           plot_shift=shift)
                    if self._cancel_calculate \
                        or self.problem.fitness.is_reset(): continue

                try:
                    # If we can calculate chisq, then put it on the graph.
                    text = "chisq=" + self.problem.chisq_str()
                    plt.text(0.01, 0.01, text, transform=plt.gca().transAxes)
                except Exception:
                    pass
                #print "drawing"
                if not reset:
                    self.toolbar.push_current()
                    set_xrange(plt.gca(), range_x)
                    self.toolbar.push_current()
                plt.draw()
                #print "done drawing"
                break

        self._calculating = False

    def _precalc(self, fitness):
        # First calculate reflectivity
        fitness.reflectivity(resolution=False)
        #print "yield 1"
        wx.Yield()
        if self._cancel_calculate or fitness.is_reset(): return
        # Then calculate resolution
        fitness.reflectivity()
        #print "yield 2"
        wx.Yield()
        if self._cancel_calculate or fitness.is_reset(): return

def set_xrange(ax, range_x):
    miny,maxy = inf,-inf
    for L in ax.get_lines():
        x,y = L.get_data()
        idx = (x>range_x[0]) & (x<range_x[1])
        if idx.any():
            miny = min(miny,min(y[idx]))
            maxy = max(maxy,max(y[idx]))
    if miny < maxy:
        if ax.get_yscale() == 'linear':
            padding = 0.05*(maxy-miny)
            miny,maxy = miny-padding, maxy+padding
        else:
            miny,maxy = miny*0.95, maxy*1.05
    ax.set_xlim(range_x)
    ax.set_ylim(miny,maxy)
