from __future__ import with_statement

import wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure

import pylab

from refl1d.probe import Probe
from refl1d.fitproblem import MultiFitProblem

from ..gui.util import EmbeddedPylab

# ------------------------------------------------------------------------
class TheoryView(wx.Panel):
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

    def menu(self):
        # Add 'View' menu to the menu bar and define its options.
        # Present y-axis plotting scales as radio buttons.
        # Grey out items that are not currently implemented.
        frame = wx.GetTopLevelParent(self)
        menu = wx.Menu()
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Fresnel",
                                          "Plot y-axis in Fresnel scale")
        frame.Bind(wx.EVT_MENU, self.OnFresnel, _item)
        _item.Check(True)
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "Li&near",
                                          "Plot y-axis in linear scale")
        frame.Bind(wx.EVT_MENU, self.OnLinear, _item)
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Log",
                                          "Plot y-axis in log scale")
        frame.Bind(wx.EVT_MENU, self.OnLog, _item)
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&Q4",
                                          "Plot y-axis in Q4 scale")
        frame.Bind(wx.EVT_MENU, self.OnQ4, _item)
        _item = menu.AppendRadioItem(wx.ID_ANY,
                                          "&SA",
                                          "Spin Asymmetry")
        frame.Bind(wx.EVT_MENU, self.OnSA, _item)

        menu.AppendSeparator()

        _item = menu.Append(wx.ID_ANY,
                                 "&Residuals",
                                 "Show residuals on plot panel")
        frame.Bind(wx.EVT_MENU, self.OnResiduals, _item)
        menu.Enable(id=_item.GetId(), enable=True)

        return menu

    # ==== Views ====
    # TODO: can probably parameterize the view selection.
    def OnLog(self, event):
        self.view = "log"
        self.redraw()

    def OnLinear(self, event):
        self.view = "linear"
        self.redraw()

    def OnFresnel(self, event):
        self.view = "fresnel"
        self.redraw()

    def OnQ4(self, event):
        self.view = "q4"
        self.redraw()

    def OnSA(self, event):
        self.view = "SA"
        self.redraw()

    def OnResiduals(self, event):
        self.view = "residual"
        self.redraw()

    # ==== Model view interface ===
    def OnShow(self, event):
        if not event.Show: return
        #print "showing theory"
        if self._need_redraw:
            #print "-redraw"
            self.redraw()

    def set_model(self, model):
        self.problem = model
        self.redraw()

    def update_model(self, model):
        if self.problem == model: self.redraw()

    def update_parameters(self, model):
        if self.problem == model: self.redraw()
    # =============================

    def redraw(self):
        # Hold off drawing until the tab is visible
        if not self.IsShown():
            self._need_redraw = True
            return
        self._need_redraw = False

        # Redraw the canvas.
        with self.pylab_interface:
            pylab.clf() # clear the canvas
            if isinstance(self.problem,MultiFitProblem):
                for p in self.problem.models:
                    p.fitness.plot_reflectivity(view=self.view)
                    pylab.hold(True)
            else:
                self.problem.fitness.plot_reflectivity(view=self.view)
            try:
                # If we can't calculate chisq, then put it on the graph.
                pylab.text(0.01, 0.01, "chisq=%g" % self.problem.chisq(),
                           transform=pylab.gca().transAxes)
            except:
                pass
            pylab.draw()
