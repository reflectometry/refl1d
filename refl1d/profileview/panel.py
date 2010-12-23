"""
A base panel to draw the profile
"""

import os
import wx
import numpy
from matplotlib.figure import Figure
from matplotlib.axes   import Subplot
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import FigureManager
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg

from .auipanel import AuiPanel
from .util import CopyImage
from .profile import ProfileInteractor
from .interactor import BaseInteractor
from .listener import Listener

# ------------------------------------------------------------------------
class ProfileView(AuiPanel):

    def __init__( self,
                  parent,
                  size=wx.DefaultSize
                  ):
        super(ProfileView, self).__init__(parent, id=-1, size=size )

        # This make sure we can communicate between different panels.
        self.parent = parent

        # Fig
        self.fig = Figure( figsize   = (1,1),
                           dpi       = 75,
                           facecolor = 'white',
                           edgecolor = 'white',
                           )
        # Canvas
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.fig.set_canvas(self.canvas)
        self.fig.add_axes( Subplot(self.fig, 111) )

        # Axes
        self.axes = self.fig.get_axes()[0]
        self.axes.set_autoscale_on(False)


        # Show toolbar or not?
        self.toolbar = NavigationToolbar2WxAgg( self.canvas )
        self.toolbar.Show(True)

        # Create a figure manager to manage things
        self.figmgr = FigureManager( self.canvas, 1, self )

        self.sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer.Add( self.canvas,1, border=2, flag= wx.LEFT|wx.TOP|wx.GROW)
        self.sizer.Add(self.toolbar)
        self.SetSizer( self.sizer)
        self.Fit()
        self.listener = Listener()

    def SetProfile(self, experiment):
        """Initialize model by profile."""

        # Turn the model into a user interface
        self.profile = ProfileInteractor(self.axes,
                                         experiment,
                                         self.listener)
        self.profile.update_markers()
        self.profile.update_profile()
        self.profile.reset_limits()
        self.profile.draw_idle()

    def onPrinterSetup(self,event=None):
        self.canvas.Printer_Setup(event=event)

    def onPrinterPreview(self,event=None):
        self.canvas.Printer_Preview(event=event)

    def onPrint(self,event=None):
        self.canvas.Printer_Print(event=event)


    def OnSaveFigureMenu(self, evt ):
        """
        Save the current figure as an image file
        """
        dlg = wx.FileDialog(self,
                       message="Save Figure As ...",
                       defaultDir=os.getcwd(),
                       defaultFile="",
                       wildcard="PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|All files (*.*)|*.*",
                       style=wx.SAVE
                      )

        _val = dlg.ShowModal()
        if  _val == wx.ID_CANCEL:  return  #Do nothing
        if  _val == wx.ID_OK:
            outfile = dlg.GetPath()

        dlg.Destroy()

        # Save
        self.fig.savefig( outfile )


    def GetToolBar(self):
        """
        backend_wx call this function. KEEP it
        """
        return None

    def OnPanelFrameClose(self, evt):
        """
        On Close this Frame
        """
        self.Destroy()
        evt.Skip()


    def OnCopyFigureMenu(self, evt ):
        """
        Copy the current figure
        """
        CopyImage(self.canvas)

    def CanShowContextMenu(self):
        return True
        
    def quit_on_error(self):
        numpy.seterr(all='raise')
        ProfileInteractor._debug = True
        BaseInteractor._debug = True
