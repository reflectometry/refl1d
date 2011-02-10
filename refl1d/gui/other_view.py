import wx
import sys

import  wx.lib.scrolledpanel as scrolled
from wx.lib.pubsub import Publisher as pub
import wx.lib.newevent

from refl1d.mystic.parameter import Parameter, BaseParameter
from refl1d.profileview.panel import ProfileView


# Other non-implemented tabs
# This tabs will be not implemented in this release

class OtherView(scrolled.ScrolledPanel):
    def __init__(self, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)

        INTRO_TEXT = "Not Implemented Yet....Under Construction:"

        self.intro_text = INTRO_TEXT
        self.log_text = wx.StaticText(self, wx.ID_ANY, label=INTRO_TEXT)

        # Create a horizontal box sizer to hold the title and progress bar.
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(self.log_text, 0, wx.ALIGN_CENTER_VERTICAL)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self.progress.Clear()
        vsizer.Add(sizer1, 0, wx.EXPAND|wx.ALL, border=10)
        vsizer.Add(self.progress, 1, wx.EXPAND)
        self.SetSizer(vsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()
