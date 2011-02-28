#!/usr/bin/python

import wx

from util import Validator

class FitControl(wx.Dialog):
    def __init__(self, parent, id, title):
        wx.Dialog.__init__(self, parent, id, title, size=(300, 575))

        vbox_top = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self, -1)

        vbox = wx.BoxSizer(wx.VERTICAL)

        vbox.Add((-1, 15))

        # panel1

        panel1 = wx.Panel(panel, -1)

        vbox1 = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        save_label = wx.StaticText(panel1, -1, 'Save location: ')
        self.save = wx.TextCtrl(panel1, -1, 'T1', size=(200, -1), style=wx.TE_RIGHT)
        self.overwrite = wx.CheckBox(panel1, wx.ID_ANY, 'overwrite')
        hbox1.Add(save_label, 0, wx.ALIGN_CENTER_VERTICAL)
        hbox1.Add(self.save)
        vbox1.Add(hbox1)
        vbox1.Add(self.overwrite)

        panel1.SetSizer(vbox1)
        vbox.Add(panel1, 0, wx.BOTTOM | wx.TOP, 9)

        vbox.Add((-1, 15))

        # panel2

        panel2 = wx.Panel(panel, -1)
        hbox2 = wx.BoxSizer(wx.VERTICAL)

        static_box1 = wx.StaticBox(panel2, -1, 'Fit Algorithms')

        self.ameoba_radio = wx.RadioButton(panel2, -1, 'Ameoba')
        self.de_radio = wx.RadioButton(panel2, -1, 'DE')
        self.de_radio.SetValue(True)
        self.dream_radio = wx.RadioButton(panel2, -1, 'Dream')
        self.pt_radio = wx.RadioButton(panel2, -1, 'Parallel Tempering')
        self.rl_radio = wx.RadioButton(panel2, -1, 'Random Lines')

        # radio button event to enable/disable other options based on 
        # algorithm selected        
        self.Bind(wx.EVT_RADIOBUTTON, self.OnDe, id=self.de_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnAmeoba, id=self.ameoba_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnDream, id=self.dream_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnPt, id=self.pt_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRl, id=self.rl_radio.GetId())


        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.VERTICAL)
        fit_hsizer.Add(self.ameoba_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.de_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.dream_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.pt_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.rl_radio, 0, wx.ALL, 5)

        hbox2.Add(fit_hsizer, 1, wx.RIGHT, 5)
        panel2.SetSizer(hbox2)
        vbox.Add(panel2, 0)

        vbox.Add((-1, 15))
        
        
        # panel4

        panel4 = wx.Panel(panel, -1)
        static_box3 = wx.StaticBox(panel4, -1, 'Other Options')
        other_hsizer = wx.StaticBoxSizer(static_box3, orient=wx.HORIZONTAL)
        vbox4 = wx.BoxSizer(wx.VERTICAL)
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer6 = wx.BoxSizer(wx.HORIZONTAL)
          
        burn_label = wx.StaticText(panel4, -1, 'Burn:                       ')
        self.burn = wx.TextCtrl(panel4, -1, '0', size=(100, -1),
                           style=wx.TE_RIGHT, validator=Validator("no-alpha"))
                           
        stepsize_label = wx.StaticText(panel4, -1, 'Step size:               ')
        self.stepsize = wx.TextCtrl(panel4, -1, '1000', size=(100, -1), 
                           style=wx.TE_RIGHT, validator=Validator("no-alpha"))
                                                      
        tmin_label = wx.StaticText(panel4,  -1, 'T min:                      ')
        self.tmin = wx.TextCtrl(panel4, -1, '0.1',  size=(100, -1),
                       style=wx.TE_RIGHT, validator=Validator("no-alpha"))
        tmax_label = wx.StaticText(panel4,  -1, 'T max:                     ')
        self.tmax = wx.TextCtrl(panel4, -1, '10',  size=(100, -1),
                       style=wx.TE_RIGHT, validator=Validator("no-alpha"))
        crossover_label = wx.StaticText(panel4,  -1, 'Crossover Ratio:  ')
        self.crossover = wx.TextCtrl(panel4, -1, '0.9', size=(100, -1), 
                           style=wx.TE_RIGHT, validator=Validator("no-alpha"))
        pop_label = wx.StaticText(panel4, -1, 'Population:             ')
        self.pop = wx.TextCtrl(panel4, -1, '10', size=(100, -1), 
                      style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        # make following option disable as 'de' algorithm is choosen by default
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)                      

        sizer1.Add(burn_label, 0, wx.ALL, 5)
        sizer1.Add(self.burn, 0, wx.ALL, 5)
        
        sizer2.Add(stepsize_label, 0, wx.ALL, 5)
        sizer2.Add(self.stepsize, 0, wx.ALL, 5)

        sizer3.Add(crossover_label, 0, wx.ALL, 5)
        sizer3.Add(self.crossover, 0, wx.ALL, 5)
        
        sizer4.Add(pop_label, 0, wx.ALL, 5)
        sizer4.Add(self.pop, 0, wx.ALL, 5)

        sizer5.Add(tmin_label, 0, wx.ALL, 5)
        sizer5.Add(self.tmin, 0, wx.ALL, 5)

        sizer6.Add(tmax_label, 0, wx.ALL, 5)
        sizer6.Add(self.tmax, 0, wx.ALL, 5)
        
        vbox4.Add(sizer1)
        vbox4.Add(sizer2)
        vbox4.Add(sizer3)
        vbox4.Add(sizer4)
        vbox4.Add(sizer5)
        vbox4.Add(sizer6)

        other_hsizer.Add(vbox4, 0, wx.TOP, 4)

        panel4.SetSizer(other_hsizer)
        vbox.Add(panel4, 0, wx.BOTTOM, 15)

        vbox.Add((-1, 15))

        # panel5

        panel5 = wx.Panel(panel, -1)
        sizer5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer5.Add((145, -1), 1, wx.EXPAND | wx.ALIGN_RIGHT)

        self.ok_btn = wx.Button(panel5, -1, 'Fit', size=(50, -1))
        self.ok_btn.SetDefault()
        self.cancel_btn = wx.Button(panel5, -1, 'Cancel', size=(50, -1))
        
        self.Bind(wx.EVT_BUTTON, self.OnFit, self.ok_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, self.cancel_btn)
        
        sizer5.Add(self.ok_btn)
        sizer5.Add(self.cancel_btn)

        panel5.SetSizer(sizer5)
        vbox.Add(panel5, 1, wx.BOTTOM, 9)

        vbox_top.Add(vbox, 1, wx.LEFT, 5)

        panel.SetSizer(vbox_top)
        vbox_top.Fit(panel)

        self.Centre()
        self.ShowModal()
        #self.Destroy()
        
    def OnDe(self, event):
        
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)
        self.crossover.Enable(True)
        self.pop.Enable(True)
        
    def OnAmeoba(self, event):
        
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)
        self.crossover.Enable(False)
        self.pop.Enable(False)
        
    
    def OnDream(self, event):
        
        self.pop.Enable(True) 
        self.burn.Enable(True)       
        self.tmin.Enable(False)
        self.tmax.Enable(False)
        self.crossover.Enable(False)
                     
    def OnRl(self, event):
        
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)
                
    def OnPt(self, event):
        
        self.pop.Enable(True) 
        self.burn.Enable(True)       
        self.tmin.Enable(True)
        self.tmax.Enable(True)
        self.crossover.Enable(True)       
                        
    def OnFit(self, event):
        
        print 'fit button pressed' 
        self.Destroy()   
        
    def OnCancel(self, event):
        print 'cancel button pressed' 
        self.Destroy()   
        
if __name__=="__main__":
    app = wx.App()
    FitControl(None, -1, 'Fit Control')
    app.MainLoop()
