import wx

class FolderDialog(wx.Dialog):
    def __init__(self, parent, id, title):
        wx.Dialog.__init__(self, parent, id, title, size=(450, 200))
        
        panel = wx.Panel(self, -1)
        
        folder_label = wx.StaticText(panel, wx.ID_ANY, label="Folder Name:")
        folder_text = wx.TextCtrl(panel, wx.ID_ANY, value="",style=wx.TE_RIGHT)
        
        location_label = wx.StaticText(panel, wx.ID_ANY, label="Save in:")
        location_text = wx.TextCtrl(panel, wx.ID_ANY, value="",style=wx.TE_RIGHT)
        
        x, y = location_text.GetSizeTuple()
        bt = wx.Button(panel, wx.ID_ANY, "...", size=(30, y))
        
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(folder_label, 0, border=5,
                        flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)
                        
        hbox1.Add(folder_text, 0, border=5,
                        flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)
        
        extra_space1 = wx.BoxSizer(wx.HORIZONTAL)
        extra_space1.Add((5,5), 1)
         
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(location_label, 0, border=5,
                        flag=wx.EXPAND|wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)
                        
        hbox2.Add(location_text,0,border=5,flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT )
        hbox2.Add(bt, 0)
        
        extra_space2 = wx.BoxSizer(wx.HORIZONTAL)
        extra_space2.Add((5,5), 1) 
        
        ok_button = wx.Button(panel, wx.ID_ANY, "Ok")
        close_button = wx.Button(panel, wx.ID_ANY, "Close")
                
        # Create a horizontal box sizer for the buttons.
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add((10,5), 1)  # stretchable whitespace
        hbox3.Add(ok_button, 0)
        hbox3.Add((3,3), 1)  # stretchable whitespace
        hbox3.Add(close_button, 0)
        hbox3.Add((10,5), 1)
        vbox_sizer = wx.BoxSizer(wx.VERTICAL)
        vbox_sizer.Add(hbox1, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT,
                        border=10)
        vbox_sizer.Add(extra_space1, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT,
                        border=10)
        
        vbox_sizer.Add(hbox2, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT,
                        border=10)
        vbox_sizer.Add(extra_space2, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT,
                        border=10)
        
        vbox_sizer.Add(hbox3, 0, wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT,
                        border=10)
                
        panel.SetSizer(vbox_sizer)
        vbox_sizer.Fit(panel)

        
