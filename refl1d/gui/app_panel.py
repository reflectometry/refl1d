# Copyright (C) 2006-2011, University of Maryland
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
import copy

import wx
import wx.lib.newevent

# This is a workaround for a py2exe problem when using pubsub from wxPython ...
# wxPython 2.8.11.0 supports both the pubsub V1 and V3 APIs (V1 is the default;
# there is no V2), whereas 2.8.9.x and 2.8.10.1 offer only the V1 API. Since
# we want to build with older versions of wxPython we will use the V1 API.
# However, py2exe does not find pubsub when 2.8.11.0 is used unless there is
# an explicit import of setupv1.  This is not a problem with older versions
# of wxPython.  The import of setupv1 should only be done once because it
# establishes the interface for all subsequent pubsub imports.
try:
    from wx.lib.pubsub import setupv1  # necessary for py2exe with 2.8.11.0
except:
    pass
from wx.lib.pubsub import Publisher as pub  # when using the V1 API
# from wx.lib.pubsub import pub  # in the future when using the V3 API

# If we are running from an image built by py2exe, keep the frozen environment
# self contained by having matplotlib use a private directory instead of using
# .matplotlib under the user's home directory for storing shared data files
# such as fontList.cache.  Note that a Windows installer/uninstaller such as
# Inno Setup should explicitly delete this private directory on uninstall.
if hasattr(sys, 'frozen'):
    mplconfigdir = os.path.join(sys.prefix, '.matplotlib')
    if not os.path.exists(mplconfigdir):
        os.mkdir(mplconfigdir)
    os.environ['MPLCONFIGDIR'] = mplconfigdir
import matplotlib

# Disable interactive mode so that plots are only updated on show() or draw().
# Note that the interactive function must be called before selecting a backend
# or importing pyplot, otherwise it will have no effect.

matplotlib.interactive(False)

# Specify the backend to use for plotting and import backend dependent classes.
# Note that this must be done before importing pyplot to have an effect.
matplotlib.use('WXAgg')

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
from refl1d.cli import FITTERS, FitOpts, FitProxy, SerialMapper, load_problem
from refl1d.cli import getopts

from .summary_view import SummaryView
from .fit_view import FitView
from .parameter_view import ParameterView
from .log_view import LogView
from .fit_dialog import FitControl
from .work_thread import Worker
from .util import nice
from .utilities import (get_appdir, get_bitmap, log_time,
                        popup_error_message, popup_warning_message,
                        StatusBarInfo, ExecuteInThread, WorkInProgress)

# File selection strings.
PYTHON_FILES = "Script files (*.py)|*.py"
REFL_FILES = "Refl files (*.refl)|*.refl"
DATA_FILES = "Data files (*.dat)|*.dat"
TEXT_FILES = "Text files (*.txt)|*.txt"
ALL_FILES = "All files (*.*)|*.*"

# Custom colors.
WINDOW_BKGD_COLOUR = "#ECE9D8"

#==============================================================================

EVT_RESULT_ID = 1

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)

#==============================================================================

IMPROVEMENT_DELAY = 5

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

#==============================================================================

class FitParams():

    opts = getopts()
    opts.pop_rl = 0.5
    default_fitter = opts.fit

    FITTER_DESC = dict(dream="Dream", rl="Random Lines", pt="Parallel Tempering",
                       ps="Particle Swarm", de="Differential Evolution",
                       newton="Quasi-Newton", amoeba="Amoeba", snobfit="SnobFit")

    default_fitter_desc = FITTER_DESC[default_fitter]

    FITTER_PARAMS = dict(
                  amoeba =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Starts:", opts.starts, opts.starts, "int"),
                    ],
                  de =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                    ],
                  dream =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Burn:", opts.burn, opts.burn, "int"),
                    ],
                  newton =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                      ("Burn:", opts.burn, opts.burn, "int"),
                      ("Min Temperature:", opts.Tmin, opts.Tmin, "float"),
                      ("Max Temperature:", opts.Tmax, opts.Tmax, "float"),
                    ],
                  ps =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                    ],
                  pt =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                      ("Burn:", opts.burn, opts.burn, "int"),
                      ("Min Temperature:", opts.Tmin, opts.Tmin, "float"),
                      ("Max Temperature:", opts.Tmax, opts.Tmax, "float"),
                    ],
                  rl =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Starts:", 1, 1, "int"),
                      ("Population:", opts.pop_rl, opts.pop_rl, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                    ],
                  snobfit =
                    [
                      ("Steps:", opts.steps, opts.steps, "int"),
                      ("Population:", opts.pop, opts.pop, "float"),
                      ("Crossover Ratio:", opts.CR, opts.CR, "float"),
                    ],
                 )

    fitter_info = {}
    for ftr in FITTERS:
        fitter_info[ftr] = [
                            FITTER_DESC[ftr],
                            FITTERS[ftr],
                            FITTER_PARAMS[ftr]
                           ]

#==============================================================================

class AppPanel(wx.Panel):
    """
    This class builds the GUI for the application on a panel and attaches it
    to the frame.
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

        # Split the panel into top and bottem halves.
        self.split_panel()

        # Modify the menu bar.
        self.modify_menubar()

        # Create list of available fitters and their parameters.
        self.init_fitter_info()

        # Create a PubSub receiver.
        pub.subscribe(self.OnUpdateDisplay, "update")
        pub.subscribe(self.OnUpdatePlot, "update_plot")
        EVT_RESULT(self, self.OnFitResult)
        self.view = "fresnel"  # default view for the plot
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
        view_menu.Enable(id=_item.GetId(), enable=True)

        mb.Insert(1, view_menu, "&View")

        # Add 'Fitting' menu to the menu bar and define its options.
        # Items are initially greyed out, but will be enabled after a script
        # is loaded.
        fit_menu = self.fit_menu = wx.Menu()

        _item = fit_menu.Append(wx.ID_ANY,
                                "&Start Fit",
                                "Start fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnStartFit, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_start = _item

        _item = fit_menu.Append(wx.ID_ANY,
                                "&Stop Fit",
                                "Stop fitting operation")
        frame.Bind(wx.EVT_MENU, self.OnStopFit, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_stop = _item

        _item = fit_menu.Append(wx.ID_ANY,
                                "Fit &Options ...",
                                "Edit fitting options")
        frame.Bind(wx.EVT_MENU, self.OnFitOptions, _item)
        fit_menu.Enable(id=_item.GetId(), enable=False)
        self.fit_menu_options = _item

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
        tb = self.tb = frame.GetToolBar()

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
        tb.EnableTool(_tool.GetId(), False)
        self.tb_start = _tool

        _tool = tb.AddSimpleTool(wx.ID_ANY, stop_bmp,
                                 "Stop Fit",
                                 "Stop fitting operation")
        frame.Bind(wx.EVT_TOOL, self.OnStopFit, _tool)
        tb.EnableTool(_tool.GetId(), False)
        self.tb_stop = _tool

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

        sp.SplitHorizontally(self.pan1, self.pan2, sashPosition=0)
        sp.SetSashGravity(0.5)  # on resize expand/contract panels equally

        # Initialize the panels.
        self.init_top_panel()
        self.init_bottom_panel()

        # Put the splitter in a sizer attached to the main panel of the page.
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(sp, 1, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)

        # Workaround: For some unknown reason, the sash is not placed in the
        # middle of the enclosing panel.  As a workaround, we reset it here.
        sp.SetSashPosition(position=0, redraw=False)

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

    def init_fitter_info(self):
        fit_params = FitParams()
        #for k, v in sorted(fit_params.fitter_info.iteritems()):
        #   print "key =", k, "value =", v

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
        pub.subscribe(self.OnStartFit, "start_fit")

        # Recieving parameter update message from parameter tab
        # This will trigger on_para_change method to update all the views of
        # model (profile tab, summary tab and the canvas will be redrawn with
        # new model parameters).
        pub.subscribe(self.OnUpdateModel, "update_model")
        pub.subscribe(self.OnUpdateParameters, "update_parameters")

        # Enable appropriate menu items.
        self.fit_menu.Enable(id=self.fit_menu_start.GetId(), enable=True)
        #self.fit_menu.Enable(id=self.fit_menu_stop.GetId(), enable=True)
        self.fit_menu.Enable(id=self.fit_menu_options.GetId(), enable=True)

        # Enable appropriate toolbar items.
        self.tb.EnableTool(id=self.tb_start.GetId(), enable=True)
        #self.tb.EnableTool(id=self.tb_stop.GetId(), enable=True)

    def OnResiduals(self, event):
        self.view = "residual"
        self.redraw(self.problem)

    def OnFitOptions(self, event):
        plist ={}
        for k in FITTERS:
            value = FitParams.fitter_info[k]
            plist[value[0]] = value[2]
        #print "****** plist =\n", plist

        # Pass in the frame object as the parent window so that the dialog box
        # will inherit font info from it instead of using system defaults.
        frame = wx.FindWindowByName("AppFrame")
        fit_dlg = FitControl(parent=frame, id=wx.ID_ANY, title="Fit Control",
                             plist=plist,
                             default_algo=FitParams.default_fitter_desc)

        start_fit_flag = False
        if fit_dlg.ShowModal() == wx.ID_OK:
            curr_algo, results, start_fit_flag = fit_dlg.get_results()
            #print 'results', curr_algo, results, start_fit_flag
            FitParams.default_fitter_desc = curr_algo
            for kk, vv in FitParams.FITTER_DESC.iteritems():
                if vv == curr_algo:
                    FitParams.default_fitter = kk
                    break

            for ret_k, ret_v in results.iteritems():
                #print "ret_k, ret_v =", ret_k, ret_v
                for kk, vv in FitParams.FITTER_DESC.iteritems():
                    #print "*** kk, vv =", kk, vv
                    if ret_k == vv:
                        #print "Found it", ret_k, kk
                        break
                value = FitParams.fitter_info[kk]
                vlist = value[2]
                #print "++++ kk, vlist =", kk, vlist
                for i, p in enumerate(ret_v):
                    #print "*** i, p =", i, p
                    param = list(vlist[i])
                    param[2] = p
                    vlist[i] = tuple(param)
                value[2] = vlist
                FitParams.fitter_info[kk] = value

        fit_dlg.Destroy()
        if start_fit_flag:
            OnStartFit(event)

    def OnStartFit(self, event):
        self.problem_copy = copy.deepcopy(self.problem)
        monitors = [GUIMonitor(self.problem)]

        # FIXME: Temporary very ugly code ...
        x = FitParams.fitter_info[FitParams.default_fitter]

        opts = FitOpts(self.args)
        opts.starts = 1
        for p in x[2]:
            if p[0] == "Steps:":
                opts.steps = int(p[2])
            elif p[0] == "Population:":
                opts.pop = float(p[2])
            elif p[0] == "Crossover Ratio:":
                opts.CR = float(p[2])
            elif p[0] == "Burn:":
                opts.burn = int(p[2])
            elif p[0] == "Min Temperature:":
                opts.Tmin = float(p[2])
            elif p[0] == "Max Temperature:":
                opts.Tmax = float(p[2])
            elif p[0] == "Starts:":
                opts.starts = int(p[2])

        self.fitter = FitProxy(FITTERS[FitParams.default_fitter],
                               problem=self.problem_copy, monitors=monitors,
                               options=opts)
        mapper = SerialMapper
        self.pan1.Layout()

        # Start a new thread worker and give fit problem to the worker.
        self.worker = Worker(self, self.problem_copy, fn=self.fitter,
                                   pars=opts.args, mapper=mapper)

        self.sb.SetStatusText("Fit status: Running", 3)

    def OnStopFit(self, event):
        print "Clicked on stop fit ..." # not implemented

    def OnFitResult(self, event):
        self.redraw(self.problem) # redraw the plot last time with fitted chsiq

        pub.sendMessage("fit_complete")
        if event.data is None:
            # Thread aborted (using our convention of None return)
            print "Computation failed/aborted"
        else:
            pass
            #self.remember_best(self.fitter, self.problem, event.data)

        self.sb.SetStatusText("Fit status: Complete", 3)

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
        self.view = "fresnel"
        self.redraw(self.problem)

    def OnLinear(self, event):
        self.view = "linear"
        self.redraw(self.problem)

    def OnQ4(self, event):
        self.view = "q4"
        self.redraw(self.problem)

    def OnLog(self, event):
        self.view = "log"
        self.redraw(self.problem)

    def redraw(self, model):
        # Redraw the canvas.
        pylab.clf() # clear the canvas
        self._activate_figure()
        model.fitness.plot_reflectivity(view=self.view)
        pylab.text(0.01, 0.01, "chisq=%g" % model.chisq(),
                   transform=pylab.gca().transAxes)
        pylab.draw()

    def _activate_figure(self):
        _pylab_helpers.Gcf.set_active(self.fm)
