"""
A base panel to draw the profile
"""

import os

import numpy as np
import wx
from bumps.fitproblem import FitProblem
from bumps.gui import signal
from matplotlib.axes import Subplot
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.figure import Figure

from refl1d.experiment import MixedExperiment

# from .binder import pixel_to_data
from .interactor import BaseInteractor
from .profilei import ProfileInteractor
from .util import CopyImage

IS_MAC = wx.Platform == "__WXMAC__"


# ------------------------------------------------------------------------
class ModelView(wx.Panel):
    title = "Profile"
    default_size = (600, 400)

    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # Fig
        self.fig = Figure(
            figsize=(1, 1),
            dpi=75,
            facecolor="white",
            edgecolor="white",
        )
        # Canvas
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.fig.set_canvas(self.canvas)

        # Axes
        self.axes = self.fig.add_axes(Subplot(self.fig, 111))
        self.axes.set_autoscale_on(False)
        self.theta_axes = self.axes.twinx()
        self.theta_axes.set_autoscale_on(False)

        # Show toolbar or not?
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)
        self.toolbar.Show(True)

        # Create a figure manager to manage things
        self.figmgr = FigureManagerBase(self.canvas, 1)

        # Panel layout
        self.profile_selector_label = wx.StaticText(self, label="Sample")
        self.profile_selector = wx.Choice(self)
        self.profile_selector.Hide()
        self.profile_selector_label.Hide()
        self.Bind(wx.EVT_CHOICE, self.OnProfileSelect)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, border=2, flag=wx.LEFT | wx.TOP | wx.GROW)
        self.tbsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.tbsizer.Add(self.toolbar, 0, wx.ALIGN_CENTER_VERTICAL)
        self.tbsizer.AddSpacer(20)
        self.tbsizer.Add(self.profile_selector_label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.tbsizer.AddSpacer(5)
        self.tbsizer.Add(self.profile_selector, 0, wx.ALIGN_CENTER_VERTICAL)
        self.sizer.Add(self.tbsizer)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        # Status bar
        frame = self.GetTopLevelParent()
        self.statusbar = frame.GetStatusBar()
        if self.statusbar is None:
            self.statusbar = frame.CreateStatusBar()

        def status_update(msg):
            return self.statusbar.SetStatusText(msg)

        # Set the profile interactor
        self.profile = ProfileInteractor(self.axes, self.theta_axes, status_update=status_update)

        # Add context menu and keyboard support to canvas
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu)
        # self.canvas.Bind(wx.EVT_LEFT_DOWN, lambda evt: self.canvas.SetFocus())

        self.model = None
        self._need_interactors = self._need_redraw = False
        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnContextMenu(self, event):
        """
        Forward the context menu invocation to profile, if profile exists.
        """
        sx, sy = event.GetX(), event.GetY()
        # transform = self.axes.transData
        # data_x,data_y = pixel_to_data(transform, sx, self.fig.bbox.height-sy)

        popup = wx.Menu()
        item = popup.Append(wx.ID_ANY, "&Grid on/off", "Toggle grid lines")
        wx.EVT_MENU(self, item.GetId(), lambda _: (self.axes.grid(), self.fig.canvas.draw_idle()))
        item = popup.Append(wx.ID_ANY, "&Rescale", "Show entire profile")
        wx.EVT_MENU(self, item.GetId(), lambda _: (self.profile.reset_limits(), self.profile.draw_idle()))
        self.PopupMenu(popup, (sx, sy))
        return False

    def OnProfileSelect(self, event):
        self._set_profile(*self.profiles[event.GetInt()])

    # ==== Model view interface ===
    def OnShow(self, event):
        if not event.Show:
            return
        # print "showing profile"
        if self._need_redraw:
            self.redraw(reset_interactors=False, reset_limits=True)
        # event.Skip()

    def get_state(self):
        return self.model

    def set_state(self, state):
        self.set_model(state)

    def set_model(self, model):
        # print ">>>>>>> refl1d profile set model"
        self.model = model
        self.redraw(reset_interactors=True, reset_limits=True)

    def update_model(self, model):
        # print ">>>>>>> refl1d profile update model"
        if self.model == model:
            self.redraw(reset_interactors=False, reset_limits=True)

    def update_parameters(self, model):
        # print ">>>>>>> refl1d profile update parameters"
        if self.model == model:
            self.redraw(reset_interactors=False, reset_limits=False)

    def redraw(self, reset_interactors=False, reset_limits=False):
        if reset_interactors:
            self._need_interactors = True

        if not self.IsShown():
            self._need_redraw = True
            return

        if self._need_interactors:
            self._create_interactors()
            self._set_profile(*self.profiles[0])
            self._need_interactors = False

        self.profile.redraw(reset_limits=reset_limits)

    # =============================================
    def _create_interactors(self):
        self.profiles = []

        def add_profiles(name, exp, idx):
            if isinstance(exp, MixedExperiment):
                for i, p in enumerate(exp.parts):
                    self.profiles.append((name + chr(ord("a") + i), p, idx))
            else:
                self.profiles.append((name, exp, idx))

        if isinstance(self.model, FitProblem):
            for i, p in enumerate(self.model.models):
                if hasattr(p, "reflectivity"):
                    name = p.name
                    if not name:
                        name = "M%d" % (i + 1)
                    add_profiles(name, p, i)
        else:
            add_profiles("", self.model, -1)

        self.profile_selector.Clear()
        if len(self.profiles) > 1:
            self.profile_selector.AppendItems([k for k, _, _ in self.profiles])
            self.profile_selector_label.Show()
            self.profile_selector.Show()
            self.profile_selector.SetSelection(0)
        else:
            self.profile_selector_label.Hide()
            self.profile_selector.Hide()

    def _set_profile(self, name, experiment, idx):
        # Turn the model into a user interface
        # It is the responsibility of the party that is indicating
        # that a redraw is necessary to clear the precalculated
        # parts of the view; otherwise the theory function calculator
        # is going to be triggered twice.  This happens inside profile
        # before the profile is calculated.  Note that the profile
        # panel will receive its own signal, which will cause the
        # profile interactor to draw itself again.  We hope this isn't
        # too much of a problem.
        def signal_update():
            """Notify other views that the model has changed"""
            signal.update_parameters(model=self.model)

        def force_recalc():
            self.model.model_update()

        if isinstance(self.model, FitProblem):
            self.model.set_active_model(idx)
        self.profile.set_experiment(experiment, force_recalc=force_recalc, signal_update=signal_update)

    def onPrinterSetup(self, event=None):
        self.canvas.Printer_Setup(event=event)

    def onPrinterPreview(self, event=None):
        self.canvas.Printer_Preview(event=event)

    def onPrint(self, event=None):
        self.canvas.Printer_Print(event=event)

    def OnSaveFigureMenu(self, evt):
        """
        Save the current figure as an image file
        """
        dlg = wx.FileDialog(
            self,
            message="Save Figure As ...",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard="PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|All files (*.*)|*.*",
            style=wx.SAVE,
        )
        _val = dlg.ShowModal()
        if _val == wx.ID_CANCEL:
            return  # Do nothing
        if _val == wx.ID_OK:
            outfile = dlg.GetPath()

        dlg.Destroy()

        # Save
        self.fig.savefig(outfile)

    def GetToolBar(self):
        """
        backend_wx call this function. KEEP it
        """
        return None

    def GetTitle(self):
        """
        backend_wx calls this function.
        """
        return self.title

    def OnPanelFrameClose(self, evt):
        """
        On Close this Frame
        """
        self.Destroy()
        evt.Skip()

    def OnCopyFigureMenu(self, evt):
        """
        Copy the current figure
        """
        CopyImage(self.canvas)

    def CanShowContextMenu(self):
        return True

    def quit_on_error(self):
        np.seterr(all="raise")
        ProfileInteractor._debug = True
        BaseInteractor._debug = True
