import wx
import sys

import wx.lib.scrolledpanel as scrolled

# Global log info
# We have this separate from the view because (a) it is a good idea to
# separate model and view, and (b) because our method for 'reparenting'
# a view to a top-level window involves recreating the view from scratch,
# so no information can be stored within the view.
LOG_INFO = []

class LogView(scrolled.ScrolledPanel):
    title = 'Log'
    default_size = (600,400)
    def __init__(self, *args, **kw):
        scrolled.ScrolledPanel.__init__(self, *args, **kw)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self._redraw()

        vsizer.Add(self.progress, 1, wx.EXPAND)

        self.SetSizer(vsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()

        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnShow(self, event):
        if not event.Show: return
        #print "showing log"
        if self._need_redraw:
            #print "-redraw"
            self._redraw()

    def log_message(self, message):
        if len(LOG_INFO) > 1000:
            del LOG_INFO[:-1000]
        LOG_INFO.append(message)
        self._redraw()

    def _redraw(self):
        if not self.IsShown():
            self._need_redraw = True
        else:
            self._need_redraw = False
            self.progress.Clear()
            self.progress.AppendText("\n".join(LOG_INFO))
