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
######### systen imports ############################
from __future__ import division
import os
import sys
import shutil
import copy
import wx
import logging
from wx.lib.pubsub import Publisher as pub
from wx.lib.pubsub import Publisher
import wx.lib.newevent

############### matplotlib imports #####################
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

from .summary_view import SummaryView
from .fit_view import FitView
from .parameter_view import ParameterView
from .log_view import LogView
from .other_view import OtherView

from refl1d.mystic import monitor, parameter
from .gui_logic import load_problem, make_store
from .work_thread import Worker
from .util import nice
from .utilities import (get_appdir, log_time,
                        popup_error_message, popup_warning_message,
                        StatusBarInfo, ExecuteInThread, WorkInProgress)

# Disable interactive mode so that plots are only updated on show() or draw().
# Note that the interactive function must be called before selecting a backend
# or importing pyplot, otherwise it will have no effect.

matplotlib.interactive(False)

# Specify the backend to use for plotting and import backend dependent classes.
# Note that this must be done before importing pyplot to have an effect.

from copy import deepcopy
from .images import getOpenBitmap
from .auxiliary_page import AuxiliaryPage


#### File selection
PYTHON_FILES = "py files (*.py)|*.py"
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

EVT_RESULT_ID = 1

def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)


class GUIMonitor(monitor.TimedUpdate):
    def __init__(self, problem, progress=1, improvement=15):
        monitor.TimedUpdate.__init__(self, progress=progress,
                                     improvement=improvement)
        self.problem = problem

    def show_progress(self, history):
        temp = "  "
        chisq_rounded = nice(history.value[0])
        wx.CallAfter(Publisher().sendMessage, "update", "step  " + str(history.step[0])+temp + "chisq  " + str(chisq_rounded))
        
    def show_improvement(self, history):
        self.problem.setp(history.point[0])
        out = parameter.summarize(self.problem.parameters)
        wx.CallAfter(Publisher().sendMessage, "update_plot", out)


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

        # create a pubsub receiver
        Publisher().subscribe(self.OnUpdateDisplay, "update")
        Publisher().subscribe(self.OnUpdatePlot, "update_plot")
        EVT_RESULT(self,self.OnFitResult)

        self.worker = None   #worker for fitting job

    def modify_menubar(self):
        """Adds items to the menu bar, menus, and menu options."""
        frame = self.frame
        mb = frame.GetMenuBar()
        file_menu = mb.GetMenu(0)
        file_menu.PrependSeparator()

        _item = file_menu.Prepend(wx.ID_ANY, "&Import",
                                             "Import a script file")
        frame.Bind(wx.EVT_MENU, self.OnLoadScript, _item)
        _item = file_menu.Prepend(wx.ID_ANY, "Save&As",
                                        "Save a script file in specified location")
        _item = file_menu.Prepend(wx.ID_ANY, "&Save",
                                             "Save a script file")
        _item = file_menu.Prepend(wx.ID_ANY, "&Open",
                                             "Open a script file")
        _item = file_menu.Prepend(wx.ID_ANY, "&New",
                                             "Create a new script file")

        ############View menu#########################
        view_menu = wx.Menu()
        _item = view_menu.Append(wx.ID_ANY, "&Theory",
                                            "view theory")
        _item = view_menu.Append(wx.ID_ANY, "&Panel",
                                            "view panel")
        mb.Insert(1, view_menu, "&View")


    def modify_toolbar(self):
        """Populates the tool bar."""
        tb = self.frame.GetToolBar()
        tb.Realize()
        self.frame.SetToolBar(tb)


    def modify_statusbar(self, subbars):
        """Divides the status bar into multiple segments."""

        self.sb = self.frame.GetStatusBar()
        self.sb.SetFieldsCount(len(subbars))
        self.sb.SetStatusWidths(subbars)

    def split_panel(self):
        """Creates separate left and right panels."""

        # Split the panel into two pieces.
        sp = wx.SplitterWindow(self, style=wx.SP_3D|wx.SP_LIVE_UPDATE)
        sp.SetMinimumPaneSize(100)

        self.pan1 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan1.SetBackgroundColour("WHITE")

        self.pan2 = wx.Panel(sp, wx.ID_ANY, style=wx.SUNKEN_BORDER)
        self.pan2.SetBackgroundColour("WHITE")


        sp.SplitHorizontally(self.pan1,self.pan2)

        # Initialize the left and right panels.
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
        #sizer.Add(hbox1_sizer, 0, wx.EXPAND|wx.ALL, border=10)
        sizer.Add(canvas, 1, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
        sizer.Add(mpl_toolbar, 0, wx.EXPAND|wx.ALL, border=10)

        # Associate the sizer with its container.
        self.pan1.SetSizer(sizer)
        sizer.Fit(self.pan1)

    def _activate_figure(self):
        _pylab_helpers.Gcf.set_active(self.fm)

    def init_bottom_panel(self):
        nb = self.notebook = wx.Notebook(self.pan2, wx.ID_ANY,
                             style=wx.NB_TOP|wx.NB_FIXEDWIDTH|wx.NB_NOPAGETHEME)
        nb.SetTabSize((100,20))  # works on Windows but not on Linux

        # Create page windows as children of the notebook.
        from refl1d.profileview.panel import ProfileView
        self.page0 = ProfileView(nb)
        self.page1 = ParameterView(nb)
        self.page2 = SummaryView(nb)
        self.page3 = LogView(nb)
        self.page4 = FitView(nb)

        # Add the pages to the notebook with a label to show on the tab.
        nb.AddPage(self.page0, "Profile")
        nb.AddPage(self.page1, "Parameters")
        nb.AddPage(self.page2, "Summary")
        nb.AddPage(self.page3, "Log")
        nb.AddPage(self.page4, "Fit")

        self.pan2.sizer = wx.BoxSizer(wx.VERTICAL)
        self.pan2.sizer.Add(nb, 1, wx.EXPAND)
        self.pan2.SetSizer(self.pan2.sizer)
        self.pan2.SetAutoLayout(1)
        self.pan2.sizer.Fit(self.pan2)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnPageChanged)

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

    def OnLoadScript(self, event):
        # Load the script which will contain model defination and data.
        dlg = wx.FileDialog(self,
                            message="Select Script File",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard=(PYTHON_FILES+"|"+REFL_FILES+"|"+DATA_FILES+"|"+
                                      TEXT_FILES+"|"+ALL_FILES),
                            style=wx.OPEN|wx.MULTIPLE|wx.CHANGE_DIR)

        # Wait for user to close the dialog.
        sts = dlg.ShowModal()
        if sts == wx.ID_OK:
            file_path = dlg.GetPath()
        dlg.Destroy()
        if sts == wx.ID_CANCEL:
            return  # Do nothing

        dir,file = os.path.split(file_path)
        os.chdir(dir)
        self.args = [file, 'T1']
        self.problem = load_problem(self.args)
        self.view(self.problem)

        # send new model (problem) loaded message to all listening tabs and panel
        pub.sendMessage("initial_model", self.problem)

        # recieving fit message from fit tab
        pub.subscribe(self.OnFit, "fit")

        # recieving parameter update message from parameter tab
        # this will trigger on_para_change method to update all the views of
        # model (profile tab, summary tab and the canvas will be redrawn will
        # new model parameters)
        pub.subscribe(self.OnUpdateModel, "update_model")
        pub.subscribe(self.OnUpdateParameters, "update_parameters")


    def OnUpdateModel(self, event):
        # update the profile tab and redraw the canvas with new values
        self.problem.fitness.update()
        self.view(self.problem)

    def OnUpdateParameters(self, event):
        self.view(self.problem)

    def view(self, model):
        # redraws the canvas
        pylab.clf() #### clear the canvas
        self._activate_figure()
        model.show()
        model.fitness.plot_reflectivity()
        pylab.text(0, 0, 'chisq=%g' % model.chisq(),
                   transform=pylab.gca().transAxes)        
        pylab.draw()
        

    def OnFit(self, event):
        """
        On recieving fit message this event is triggered to
        fit the data and model
        """
        # TODO: need to put options on fit panel
        from .main import FitOpts, FitProxy, SerialMapper
        from refl1d.fitter import RLFit, DEFit, BFGSFit, AmoebaFit, SnobFit
        from refl1d.probe import Probe

        self.sb.SetStatusText('Fit status: Running', 3)
        moniter = GUIMonitor(self.problem)
        opts = FitOpts(self.args)

        FITTERS = dict(dream=None, rl=RLFit,
                   de=DEFit, newton=BFGSFit, amoeba=AmoebaFit, snobfit=SnobFit)

        self.fitter = FitProxy(FITTERS[opts.fit],
                               problem=self.problem, moniter=moniter,opts=opts,)
        mapper = SerialMapper

        Probe.view = opts.plot

        make_store(self.problem,opts)
        self.pan1.Layout()
        
        #self.temp = copy.deepcopy(self.problem)
        # start a new thread worker and give fit problem to the worker
        self.worker = Worker(self, self.problem, fn = self.fitter,
                                       pars = opts.args,
                                       mapper = mapper)

    def OnFitResult(self, event):
        self.sb.SetStatusText('Fit status: Complete', 3)
        pub.sendMessage("fit_complete")
        if event.data is None:
            # Thread aborted (using our convention of None return)
            print 'Computation failed/aborted'
        else:
            self.remember_best(self.fitter, self.problem, event.data)


    def remember_best(self,fitter, problem, best):

        fitter.save(problem.output)

        try:
            problem.save(problem.output, best)
        except:
            pass
        sys.stdout = open(problem.output+".out","w")

        #self.progress_gauge.Stop()
        #self.progress_gauge.Show(False)
        self.pan1.Layout()

        self.view(problem)

    def OnUpdateDisplay(self, msg):
        """
        Receives fit update messages from the thread
        and redirects the update messages to log tab for dispaly
        """
        pub.sendMessage("log", msg.data)

    def OnUpdatePlot(self, d):
        """
        Receives data from thread and update the plot
        get the model fittable parameter and send message all views to update
        itself
        """
        pub.sendMessage("update_parameters", self.problem)
        self.view(self.problem)
        
    
