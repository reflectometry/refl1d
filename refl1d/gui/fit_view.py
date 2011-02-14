import wx
import sys

from wx.lib.pubsub import Publisher as pub
import wx.lib.newevent

class FitView(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent

        extra_space1 = wx.BoxSizer(wx.HORIZONTAL)
        extra_space1.Add((5,5), 1)

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        label1 = wx.StaticText(self, 1, label="Store Folder:")

        self.store_file = wx.TextCtrl(self, 2, value="",style=wx.TE_RIGHT)
        sizer1.Add(label1, 0, border=5,
                        flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)
        sizer1.Add(self.store_file, 1, wx.EXPAND|wx.RIGHT, border=10)

        # Create the Compute button.
        self.btn_fit = wx.Button(self, wx.ID_ANY, "Fit")
        self.btn_fit.SetToolTip(wx.ToolTip("click to start fit"))
        self.Bind(wx.EVT_BUTTON, self.OnFit, self.btn_fit)
        
        pub.subscribe(self.OnFitComplete, "fit_complete")
        
        # Create a horizontal box sizer for the buttons.
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add((5,5), 1)  # stretchable whitespace
        sizer2.Add((2,2), 1)  # stretchable whitespace
        sizer2.Add(self.btn_fit, 0)
        sizer2.Add((5,5), 1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(extra_space1, 0, wx.ALL,5)
        sizer.Add(sizer1, 0, wx.ALL, 5)
        sizer.Add(extra_space1, 0, wx.ALL,5)
        sizer.Add(sizer2, 0, wx.ALL, 5)

        self.SetSizer(sizer)

    def OnFit(self, event):
        btnLabel = self.btn_fit.GetLabel()
        if btnLabel == "Fit":
            self.btn_fit.SetLabel("Stop")
            self.store = self.store_file.GetValue()
            ################LOGIC######################
            # send fit event message to panel with
            # all required data to fit
            # the panel will listen to event and start
            # the fit.
            ###########################################
            pub.sendMessage("fit", str(self.store))

        else:
            print 'stop logic goes here'
            self.btn_fit.SetLabel("Fit")
            pass


    def OnFitComplete(self, event):
        self.btn_fit.SetLabel("Fit")
       
    
