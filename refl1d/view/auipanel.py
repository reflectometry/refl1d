import wx

class AuiPanel(wx.Panel):
    """ A base panel to support AUI managerment. """
    def __init__(self,
                 parent,
                 id    = -1,
                 pos   = wx.DefaultPosition,
                 size  = wx.DefaultSize,
                 style = wx.TAB_TRAVERSAL,
                 name  = ''
                 ):
        """ Constructor"""
        super(AuiPanel, self).__init__(parent=parent, id=id, pos=pos,
                                       size=size, style=style, name=name )
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDClick)


    def OnLeftDClick(self, event):
        self.RestorePerspective()
        event.Skip()


    def SetPerspective(self, perspective):
        """ Set the perspective to restore itself in AUI """
        self._perspective = perspective


    def GetPerspective(self):
        """ Get the perspective to restore itself in AUI """
        return self._perspective


    def RestorePerspective(self):
        """ Restore itself in AUI """
        try:
            mgr = self.GetParent().GetOwnerManager()

            mgr.LoadPerspective(self.GetPerspective())
            all_panes = mgr.GetAllPanes()
            for pane in range(len(all_panes)):
                all_panes[pane].Show()
            mgr.Update()

        except:
            pass

        return

    def ShowErrorMsg(self, msg, title):
        msg = wx.MessageDialog(self, msg, title, wx.ICON_ERROR|wx.OK)
        msg.ShowModal()
        msg.Destroy()
