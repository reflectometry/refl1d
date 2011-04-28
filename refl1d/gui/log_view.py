import wx
import sys

import wx.lib.scrolledpanel as scrolled

from .util import subscribe

class LogView(scrolled.ScrolledPanel):
    def __init__(self, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        _ = """
        self.intro_text = "Fitting Progress Log:"
        self.log_text = wx.StaticText(self, wx.ID_ANY, label=INTRO_TEXT)

        # Create a horizontal box sizer to hold the title and progress bar.
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(self.log_text, 0, wx.ALIGN_CENTER_VERTICAL)
        vsizer.Add(sizer1, 0, wx.EXPAND|wx.ALL, border=0)
        """

        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self.progress.Clear()

        vsizer.Add(self.progress, 1, wx.EXPAND)

        self.SetSizer(vsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()

        subscribe(self.Onlog, "log")


    def Onlog(self, message):
        self.progress.AppendText(message)
        self.progress.AppendText('\n')
