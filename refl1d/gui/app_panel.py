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
# Author: James Krycka, Nikunj Patel

"""
This module implements the AppPanel class which creates the main panel on top
of the frame of the GUI for the Refl1D application.
"""

#==============================================================================
from __future__ import division
import os
import sys
import shutil
import logging
from copy import deepcopy

import wx
import wx.lib.newevent
from wx.lib.pubsub import Publisher as pub

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

from matplotlib.patches import Patch

# The Figure object is used to create backend-independent plot representations.
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

# For use in the matplotlib toolbar.
from matplotlib.widgets import Slider, Button, RadioButtons

# Wx-Pylab magic for displaying plots within an application's window.
from matplotlib import _pylab_helpers
from matplotlib.backend_bases import FigureManagerBase

import pylab

from refl1d.mystic import monitor, parameter
from refl1d.profileview.panel import ProfileView
from refl1d.probe import Probe

from .summary_view import SummaryView
from .fit_view import FitView
from .parameter_view import ParameterView
from .log_view import LogView
from .other_view import OtherView
from .fit_dialog import FitControl
from .gui_logic import load_problem, make_store
from .work_thread import Worker
from .util import nice
from .utilities import (get_appdir, get_bitmap, log_time,
                        popup_error_message, popup_warning_message,
                        StatusBarInfo, ExecuteInThread, WorkInProgress)

# Disable interactive mode so that plots are only updated on show() or draw().
# Note that the interactive function must be called before selecting a backend
# or importing pyplot, otherwise it will have no effect.

matplotlib.interactive(False)

# Specify the backend to use for plotting and import backend dependent classes.
# Note that this must be done before importing pyplot to have an effect.

#from .images import getOpenBitmap

# File selection strings.
PYTHON_FILES = "Script files (*.py)|*.py"
REFL_FILES = "Refl files (*.refl)|*.refl"
DATA_FILES = "Data files (*.dat)|*.dat"
TEXT_FILES = "Text files (*.txt)|*.txt"
ALL_FILES = "All files (*.*)|*.*"

# Custom colors.
PALE_GREEN = "#C8FFC8"
PALE_BLUE  = "#E8E8FF"
WINDOW_BKGD_COLOUR = "#ECE9D8"
PALE_YELLOW = "#FFFFB0"

#==============================================================================
IMPROVEMENT_DELAY = 5

EVT_RESULT_ID = 1

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)


class GUIMonitor(monitor.TimedUpdate):
    def __init__(self, problem, progress=1, improvement=None):
        improvement = improvement if improvement else IMPROVEMENT_DELAY
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = problem

    def show_progress(self, history):
        temp = "  "
        chisq_rounded = nice(history.value[0])
        wx.CallAfter(pub.sendMessage, "update",
            "step  " + str(history.step[0])+temp + "chisq  " + str(chisq_rounded))

    def show_improvement(self, history):
        self.problem.setp(history.point[0])
        out = parameter.summarize(self.problem.parameters)
        wx.CallAfter(pub.sendMessage, "update_plot", out)


class AppPanel(wx.Panel):
    """
    This class creates the main panel of the frame and builds the GUI for the
    application on it.
    """

    def __init__(self, frame, id=wx.ID_ANY, style=wx.RAISED_BORDER,
                 name="AppPanel"
                ):
        # Create a panel on the frame.  This will be the only child panel of
        # the frame and it inherits its size from the frame which is useful
        # during resize operations (as it provides a minimal size to sizers).

        wx.Panel.__init__(self, parent=frame, id=id, style=style, name=name)

        self.SetBackgroundColour("WHITE")
        self.frame = frame

        # Modify the tool bar.
        self.modify_toolbar()

        # Reconfigure the status bar.
        self.modify_statusbar([-34, -50, -16, -16])

        # Split the panel into left and right halves.
        self.split_panel()

        # Modify the menu bar.
        self.modify_menubar()

        # Create a PubSub receiver.
        pub.subscribe(self.OnUpdateDisplay, "update")
        pub.subscribe(self.OnUpdatePlot, "update_plot")
        EVT_RESULT(self,self.OnFitResult)

        self.worker = None   #worker for fitting job

    def modify_menubar(self):
        """
        Adds items to the menu bar, menus, and menu options.
        The menu bar should already have a simple File menu and a Help menu.
        """
        frame = self.frame
        mb = frame.GetMenuBar()

        # Add items to the "File" menu (prepending them in reverse order).
        # Grey out items that are not currently implemented.
        file_menu = mb.GetMenu(0)
        file_menu.PrependSeparator()

        _item = file_menu.Prepend(wx.ID_ANY,
                                  "&Import",
                                  "Import script to define model")
        frame.Bind(wx.EVT_MENU, self.OnImportScript, _item)

        file_menu.PrependSeparator()

        _item = file_menu.Prepend(wx.ID_SAVEAS,
                                  "Save&As",
                                  "Save model as another name")
        frame.Bind(wx.EVT_MENU, self.OnSaveAsModel, _item)
        file_menu.Enable(id=wx.ID_SAVEAS, enable=False)
        _item = file_menu.Prepend(wx.ID_SAVE,
                                  "&Save",
                                  "Save model")
        frame.Bind(wx.EVT_MENU, self.OnSaveModel, _item)
        file_menu.Enable(id=wx.ID_SAVE, enable=False)
        _item = file_menu.Prepend(wx.ID_OPEN,
                                  "&Open",
                                  "Open existing model")
        frame.Bind(wx.EVT_MENU, self.OnOpenModel, _item)
        file_menu.Enable(id=wx.ID_OPEN, enable=False)
        _item = file_menu.Prepend(wx.ID_NEW,
                                  "&New",
                                  "Create new model")
        frame.Bind(wx.EVT_MENU, self.OnNewModel, _item)
        file_menu.Enable(id=wx.ID_NEW, enable=False)

        # Add 'View' menu to the menu bar and define its options.
        # Present y-axis plotting scales as radio buttons.
        # Grey out items that are not currently implemented.
        view_menu = wx.Menu()
        _item = view_menu.AppendRadioItem(wx.ID_ANY,
                                          "&Fresnel",
                                          "Plot y-axis in Fresnel scale")
        frame.Bind(wx.EVT_MENU, self.OnFresnel, _item)
        _item.Check(True)
        _item = view_menu.AppendRadioItem(wx.ID_ANY,
                                          "Li&near",
                                          "Plot y-axis in linear scale")
        frame.Bind(wx.EVT_MENU, self.OnLinear, _item)
        _item = view_menu.AppendRadioItem(wx.ID_ANY,
                                          "&Log",
                                          "Plot y-axis in log scale")
        frame.Bind(wx.EVT_MENU, self.OnLog, _item)
        _item = view_menu.AppendRadioItem(wx.ID_ANY,
                                          "&Q4",
                                          "Plot y-axis in Q4 scale")
        frame.Bind(wx.EVT_MENU, self.OnQ4, _item)

        view_menu.AppendSeparator()

        _item = view_menu.Append(wx.ID_ANY,
                                 "&Residuals",
                                 "Show residuals on plot panel")
        frame.Bind(wx.EVT_MENU, self.OnResiduals, _item)
        view_menu.Enable(id=_item.GetId(), enable=False)

        mb.Insert(1, view_menu, "&View")

        # Add 'Fitting' menu to the menu bar and define its options.
        # Grey out items that are not currently implemented.
        fit_menu = wx.Menu()

        _item = fit_menu.Append(wx.ID_ANY,
                                "&Start Fit",
                                "Start fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnStartFit, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        _item = fit_menu.Append(wx.ID_ANY,
                                "&Stop Fit",
                                "Stop fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnStopFit, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        _item = fit_menu.Append(wx.ID_ANY,
                                "Fit &Options ...",
                                "Edit fitting options")
        frame.Bind(wx.EVT_MENU, self.OnFitOptions, _item)

        mb.Insert(2, fit_menu, "&Fitting")

        # Add 'Advanced' menu to the menu bar and define its options.
        adv_menu = wx.Menu()

        _item = adv_menu.AppendRadioItem(wx.ID_ANY,
                               "&Top-Bottom",
                               "Display plot and view panels top to bottom")
        frame.Bind(wx.EVT_MENU, self.OnSplitHorizontal, _item)
        _item.Check(True)
        _item = adv_menu.AppendRadioItem(wx.ID_ANY,
                               "&Left-Right",
                               "Display plot and view panels left to right")
        frame.Bind(wx.EVT_MENU, self.OnSplitVertical, _item)
        self.vert_sash_pos = 0  # set sash to center on first vertical split

        adv_menu.AppendSeparator()

        _item = adv_menu.Append(wx.ID_ANY,
                                "&Swap Panels",
                                "Switch positions of plot and view panels")
        frame.Bind(wx.EVT_MENU, self.OnSwapPanels, _item)

        mb.Insert(3, adv_menu, "&Advanced")

    def modify_toolbar(self):
        """Populates the tool bar."""
        frame = self.frame
        tb = frame.GetToolBar()

        script_bmp = get_bitmap("import_script.png", wx.BITMAP_TYPE_PNG)
        start_bmp = get_bitmap("start_fit.png", wx.BITMAP_TYPE_PNG)
        stop_bmp = get_bitmap("stop_fit.png", wx.BITMAP_TYPE_PNG)

        _tool = tb.AddSimpleTool(wx.ID_ANY, script_bmp,
                                 "Import Script",
                                 "Load model from script")
        frame.Bind(wx.EVT_TOOL, self.OnImportScript, _tool)

        tb.AddSeparator()

        _tool = tb.AddSimpleTool(wx.ID_ANY, start_bmp,
                                 "Start Fit",
                                 "Start fitting operation")
        frame.Bind(wx.EVT_TOOL, self.OnStartFit, _tool)
        _tool = tb.AddSimpleTool(wx.ID_ANY, stop_bmp,
                                 "Stop Fit",
                                 "Stop fitting operation")
        frame.Bind(wx.EVT_TOOL, self.OnStopFit, _tool)

        tb.Realize()
        frame.SetToolBar(tb)

    def modify_statusbar(self, subbars):
        """Divides the status bar into multiple segments."""

        self.sb = self.frame.GetStatusBar()
        self.sb.SetFieldsCount(len(subbars))
        self.sb.SetStatusWidths(subbars)

    def split_panel(self):
        """Splits panel into a top panel and a bottom panel."""

        # Split the panel into two pieces.
        self.sp = sp = wx.SplitterWindow(self, style=wx.SP_3D|wx.SP_LIVE_UPDATE)
        sp.SetMinimumPaneSize(100)

        self.pan1 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan1.SetBackgroundColour("WHITE")

        self.pan2 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan2.SetBackgroundColour("WHITE")

        sp.SplitHorizontally(self.pan1, self.pan2)
        sp.SetSashGravity(0.5)  # on resize expand/contract panels equally

        # Initialize the panels.
        self.init_top_panel()
        self.init_bottom_panel()

        # Put the splitter in a sizer attached to the main panel of the page.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sp, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)

    def init_top_panel(self):
        # Instantiate a figure object that will contain our plots.
        figure = Figure(figsize=(1,1), dpi=72)

        # Initialize the figure canvas, mapping the figure object to the plot
        # engine backend.
        canvas = FigureCanvas(self.pan1, wx.ID_ANY, figure)

        # Wx-Pylab magic ...
        # Make our canvas the active figure manager for pylab so that when
        # pylab plotting statements are executed they will operate on our
        # canvas and not create a new frame and canvas for display purposes.
        # This technique allows this application to execute code that uses
        # pylab stataments to generate plots and embed these plots in our
        # application window(s).

        self.fignum = 0
        self.fm = FigureManagerBase(canvas, self.fignum)

        # Instantiate the matplotlib navigation toolbar and explicitly show it.
        mpl_toolbar = Toolbar(canvas)
        mpl_toolbar.Realize()

        # Create a progress bar to be displayed during a lengthy computation.
        #self.progress_gauge = WorkInProgress(self.pan1)
        #self.progress_gauge.Show(False)

        # Create a vertical box sizer to manage the widgets in the main panel.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=0)
        sizer.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=0)

        # Associate the sizer with its container.
        self.pan1.SetSizer(sizer)
        sizer.Fit(self.pan1)

    def init_bottom_panel(self):
        nb = self.notebook = wx.Notebook(self.pan2, wx.ID_ANY,
                             style=wx.NB_TOP|wx.NB_FIXEDWIDTH|wx.NB_NOPAGETHEME)
        nb.SetTabSize((100,20))  # works on Windows but not on Linux

        # Create page windows as children of the notebook.
        self.page0 = ProfileView(nb)
        self.page1 = ParameterView(nb)
        self.page2 = SummaryView(nb)
        self.page3 = LogView(nb)
        #self.page4 = OtherView(nb)

        # Add the pages to the notebook with a label to show on the tab.
        nb.AddPage(self.page0, "Profile")
        nb.AddPage(self.page1, "Parameters")
        nb.AddPage(self.page2, "Summary")
        nb.AddPage(self.page3, "Log")
        #nb.AddPage(self.page4, "Dummy")

        self.pan2.sizer = wx.BoxSizer(wx.VERTICAL)
        self.pan2.sizer.Add(nb, 1, wx.EXPAND)
        self.pan2.SetSizer(self.pan2.sizer)
        self.pan2.SetAutoLayout(True)
        self.pan2.sizer.Fit(self.pan2)

        # Make sure the first page is the active one.
        # Note that SetSelection generates a page change event only if the
        # page changes and ChangeSelection does not generate an event.  Thus
        # we force a page change event so that the status bar is properly set
        # on startup.

        nb.ChangeSelection(0)
        nb.SendPageChangedEvent(0, 0)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanged)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.OnPageChanging)

    def OnPageChanged(self, event):
        old = event.GetOldSelection()
        new = event.GetSelection()
        sel = self.notebook.GetSelection()
        event.Skip()

    def OnPageChanging(self, event):
        old = event.GetOldSelection()
        new = event.GetSelection()
        sel = self.notebook.GetSelection()
        event.Skip()

    def OnNewModel(self, event):
        print "Clicked on new model ..." # not implemented

    def OnOpenModel(self, event):
        print "Clicked on open model ..." # not implemented

    def OnSaveModel(self, event):
        print "Clicked on save model ..." # not implemented

    def OnSaveAsModel(self, event):
        print "Clicked on save as model ..." # not implemented

    def OnImportScript(self, event):
        # Load the script which will contain model defination and data.
        dlg = wx.FileDialog(self,
                            message="Select Script File",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=(PYTHON_FILES+"|"+ALL_FILES),
                            style=wx.OPEN|wx.CHANGE_DIR)

        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            file_path = dlg.GetPath()
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        dir,file = os.path.split(file_path)
        os.chdir(dir)
        self.args = [file, "T1"]
        self.problem = load_problem(self.args)
        self.redraw(self.problem)

        # Send new model (problem) loaded message to all interested panels.
        pub.sendMessage("initial_model", self.problem)

        # Recieving message to start a fit operation.
        #pub.subscribe(self.OnFit, "fit")
        pub.subscribe(self.OnFit, "fit_option")
        # Recieving parameter update message from parameter tab
        # This will trigger on_para_change method to update all the views of
        # model (profile tab, summary tab and the canvas will be redrawn with
        # new model parameters).
        pub.subscribe(self.OnUpdateModel, "update_model")
        pub.subscribe(self.OnUpdateParameters, "update_parameters")

    def OnResiduals(self, event):
        print "Clicked on residuals ..." # not implemented

    def OnStartFit(self, event):
        print "Clicked on start fit ..." # not implemented

    def OnStopFit(self, event):
        print "Clicked on stop fit ..." # not implemented

    def OnFitOptions(self, event):
        fit_dlg = FitControl(self, -1, "Fit Control")

    def OnFit(self, event):
        from .main import FitOpts, FitProxy, SerialMapper
        from refl1d.fitter import RLFit, DEFit, BFGSFit, AmoebaFit, SnobFit

        opts = FitOpts(self.args)
        FITTERS = dict(dream=None, rl=RLFit,
                   de=DEFit, newton=BFGSFit, amoeba=AmoebaFit, snobfit=SnobFit)

        self.sb.SetStatusText("Fit status: Running", 3)
        monitor = GUIMonitor(self.problem)

        options = event.data
        algorithm = options["algo"]

        self.fitter = FitProxy(FITTERS[algorithm],
                               problem=self.problem, monitor=monitor,
                               options=options)
        mapper = SerialMapper

        Probe.view = opts.plot

        make_store(self.problem,opts)

        self.pan1.Layout()

        # Start a new thread worker and give fit problem to the worker.
        self.worker = Worker(self, self.problem, fn=self.fitter,
                                   pars=opts.args, mapper=mapper)

    '''
    def OnFit(self, event):
        """
        On recieving a fit message, start a fit of the model to the data.
        """
        # TODO: Need to put options on fit panel.
        from .main import FitOpts, FitProxy, SerialMapper
        from refl1d.fitter import RLFit, DEFit, BFGSFit, AmoebaFit, SnobFit
        from refl1d.probe import Probe

        self.sb.SetStatusText('Fit status: Running', 3)
        monitor = GUIMonitor(self.problem)
        opts = FitOpts(self.args)

        FITTERS = dict(dream=None, rl=RLFit,
                   de=DEFit, newton=BFGSFit, amoeba=AmoebaFit, snobfit=SnobFit)

        self.fitter = FitProxy(FITTERS[opts.fit],
                               problem=self.problem, monitor=monitor,opts=opts,)
        mapper = SerialMapper

        Probe.view = opts.plot

        make_store(self.problem,opts)
        self.pan1.Layout()

        #self.temp = copy.deepcopy(self.problem)
        # Start a new thread worker and give fit problem to the worker.
        self.worker = Worker(self, self.problem, fn=self.fitter,
                                   pars=opts.args, mapper=mapper)
    '''

    def OnFitResult(self, event):
        self.sb.SetStatusText("Fit status: Complete", 3)
        pub.sendMessage("fit_complete")
        if event.data is None:
            # Thread aborted (using our convention of None return)
            print "Computation failed/aborted"
        else:
            self.remember_best(self.fitter, self.problem, event.data)

    def remember_best(self, fitter, problem, best):
        fitter.save(problem.output)

        try:
            problem.save(problem.output, best)
        except:
            pass
        sys.stdout = open(problem.output+".out", "w")

        self.pan1.Layout()
        self.redraw(problem)

    def OnSplitHorizontal(self, event):
        # Place panels in Top-Bottom orientation.
        # Note that this event does not occur if user chooses same orientation.
        self.vert_sash_pos = self.sp.GetSashPosition()
        self.sp.SetSplitMode(wx.SPLIT_HORIZONTAL)
        self.sp.SetSashPosition(position=self.horz_sash_pos, redraw=False)
        self.sp.SizeWindows()
        self.sp.Refresh(eraseBackground=False)

    def OnSplitVertical(self, event):
        # Place panels in Left-Right orientation.
        # Note that this event does not occur if user chooses same orientation.
        self.horz_sash_pos = self.sp.GetSashPosition()
        self.sp.SetSplitMode(wx.SPLIT_VERTICAL)
        self.sp.SetSashPosition(position=self.vert_sash_pos, redraw=False)
        self.sp.SizeWindows()
        self.sp.Refresh(eraseBackground=False)

    def OnSwapPanels(self, event):
        win1 = self.sp.GetWindow1()
        win2 = self.sp.GetWindow2()
        self.sp.ReplaceWindow(winOld=win1, winNew=win2)
        self.sp.ReplaceWindow(winOld=win2, winNew=win1)
        sash_pos = -self.sp.GetSashPosition()  # set sash to keep panel sizes
        self.sp.SetSashPosition(position=sash_pos, redraw=False)
        self.sp.Refresh(eraseBackground=False)

    def OnUpdateModel(self, event):
        # Update the profile tab and redraw the canvas with new values.
        self.problem.fitness.update()
        self.redraw(self.problem)

    def OnUpdateParameters(self, event):
        self.redraw(self.problem)

    def OnUpdateDisplay(self, msg):
        """
        Receives fit update messages from the thread and redirects
        the update messages to the log view tab for display.
        """
        pub.sendMessage("log", msg.data)

    def OnUpdatePlot(self, d):
        """
        Receives data from thread and update the plot
        get the model fittable parameter and send message all views to update
        itself
        """
        pub.sendMessage("update_parameters", self.problem)
        self.redraw(self.problem)

    def OnFresnel(self, event):
        Probe.view = "fresnel"
        self.redraw(self.problem)

    def OnLinear(self, event):
        Probe.view = "linear"
        self.redraw(self.problem)

    def OnQ4(self, event):
        Probe.view = "q4"
        self.redraw(self.problem)

    def OnLog(self, event):
        Probe.view = "log"
        self.redraw(self.problem)

    def redraw(self, model):
        # Redraw the canvas.
        pylab.clf() #### clear the canvas
        self._activate_figure()
        model.show()
        model.fitness.plot_reflectivity()
        pylab.text(0.01, 0.01, "chisq=%g" % model.chisq(),
                   transform=pylab.gca().transAxes)
        pylab.draw()

    def _activate_figure(self):
        _pylab_helpers.Gcf.set_active(self.fm)
