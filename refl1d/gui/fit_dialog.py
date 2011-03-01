#!/usr/bin/python
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
# Author: Nikunj Patel and James Krycka

"""
This module implements the FitControl class which presents a pop-up dialog box
for the user to control fitting options.
"""

#==============================================================================

import wx

from util import Validator
from wx.lib.pubsub import Publisher as pub

class FitControl(wx.Dialog):
    """
    FitControl lets the user set fitting options from a pop-up a dialog box.
    """

    def __init__(self,
                 parent = None,
                 id     = wx.ID_ANY,
                 title  = "Fit Dialog",
                 pos    = wx.DefaultPosition,
                 size   = wx.DefaultSize, # dialog box size will be calculated
                 style  = wx.DEFAULT_DIALOG_STYLE,
                 name   = ""
                ):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Section 1

        vbox1 = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        save_label = wx.StaticText(self, -1, "Save location: ")
        self.save = wx.TextCtrl(self, -1, "T1", style=wx.TE_RIGHT)
        self.overwrite = wx.CheckBox(self, wx.ID_ANY, " Overwrite")
        hbox1.Add(save_label, 0, wx.ALIGN_CENTER_VERTICAL)
        hbox1.Add(self.save, 1, wx.EXPAND)

        vbox.Add(hbox1, 0, wx.ALL|wx.EXPAND, 10)
        vbox.Add(self.overwrite, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        # Section 2

        panel2 = wx.Panel(self, -1)

        static_box1 = wx.StaticBox(panel2, -1, "Fit Algorithms")

        self.amoeba_radio = wx.RadioButton(panel2, -1, "Amoeba")
        self.de_radio = wx.RadioButton(panel2, -1, "DE")
        self.de_radio.SetValue(True)
        self.dream_radio = wx.RadioButton(panel2, -1, "Dream")
        self.pt_radio = wx.RadioButton(panel2, -1, "Parallel Tempering")
        self.rl_radio = wx.RadioButton(panel2, -1, "Random Lines")

        # Radio button events to enable/disable other options based on the
        # algorithm selected.
        self.Bind(wx.EVT_RADIOBUTTON, self.OnDE, id=self.de_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnAmoeba, id=self.amoeba_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnDream, id=self.dream_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnPT, id=self.pt_radio.GetId())
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRL, id=self.rl_radio.GetId())

        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.VERTICAL)
        fit_hsizer.Add(self.amoeba_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.de_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.dream_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.pt_radio, 0, wx.ALL, 5)
        fit_hsizer.Add(self.rl_radio, 0, wx.ALL, 5)

        panel2.SetSizer(fit_hsizer)
        vbox.Add(panel2, 0, wx.ALL, 10)

        # Section 3

        panel3 = wx.Panel(self, -1)
        static_box3 = wx.StaticBox(panel3, -1, "Fitting Options")

        opts_hsizer = wx.StaticBoxSizer(static_box3, orient=wx.HORIZONTAL)
        vbox3 = wx.BoxSizer(wx.VERTICAL)
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer6 = wx.BoxSizer(wx.HORIZONTAL)

        step_label = wx.StaticText(panel3, -1, "Step Size:", size=(90, -1))
        self.stepsize = wx.TextCtrl(panel3, -1, "1000", size=(100, -1),
                           style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        pop_label = wx.StaticText(panel3, -1, "Population:", size=(90, -1))
        self.pop = wx.TextCtrl(panel3, -1, "10", size=(100, -1),
                      style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        co_label = wx.StaticText(panel3, -1, "Crossover Ratio:", size=(90, -1))
        self.crossover = wx.TextCtrl(panel3, -1, "0.9", size=(100, -1),
                            style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        burn_label = wx.StaticText(panel3, -1, "Burn:", size=(90, -1))
        self.burn = wx.TextCtrl(panel3, -1, "0", size=(100, -1),
                       style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        tmin_label = wx.StaticText(panel3, -1, "T min:", size=(90, -1))
        self.tmin = wx.TextCtrl(panel3, -1, "0.1",  size=(100, -1),
                       style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        tmax_label = wx.StaticText(panel3, -1, "T max:", size=(90, -1))
        self.tmax = wx.TextCtrl(panel3, -1, "10",  size=(100, -1),
                       style=wx.TE_RIGHT, validator=Validator("no-alpha"))

        self.OnDE(event=None) # Set fit options for default 'DE' algorithm

        sizer1.Add(step_label, 0, wx.ALL, 5)
        sizer1.Add(self.stepsize, 0, wx.ALL, 5)

        sizer2.Add(pop_label, 0, wx.ALL, 5)
        sizer2.Add(self.pop, 0, wx.ALL, 5)

        sizer3.Add(co_label, 0, wx.ALL, 5)
        sizer3.Add(self.crossover, 0, wx.ALL, 5)

        sizer4.Add(burn_label, 0, wx.ALL, 5)
        sizer4.Add(self.burn, 0, wx.ALL, 5)

        sizer5.Add(tmin_label, 0, wx.ALL, 5)
        sizer5.Add(self.tmin, 0, wx.ALL, 5)

        sizer6.Add(tmax_label, 0, wx.ALL, 5)
        sizer6.Add(self.tmax, 0, wx.ALL, 5)

        vbox3.Add(sizer1)
        vbox3.Add(sizer2)
        vbox3.Add(sizer3)
        vbox3.Add(sizer4)
        vbox3.Add(sizer5)
        vbox3.Add(sizer6)

        opts_hsizer.Add(vbox3, 0, wx.TOP, 5)

        panel3.SetSizer(opts_hsizer)
        vbox.Add(panel3, 0, wx.ALL, 10)

        # Section 4

        panel4 = wx.Panel(self, -1)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Create the button controls (Fit and Cancel) and bind their events.
        self.fit_btn = wx.Button(panel4, -1, "Fit")
        self.fit_btn.SetDefault()
        self.cancel_btn = wx.Button(panel4, -1, "Cancel")

        self.Bind(wx.EVT_BUTTON, self.OnFit, self.fit_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, self.cancel_btn)

        btn_sizer.Add((10,-1), 1)  # stretchable whitespace
        btn_sizer.Add(self.fit_btn)
        btn_sizer.Add((10,-1), 0)  # non-stretchable whitespace
        btn_sizer.Add(self.cancel_btn, 0)

        panel4.SetSizer(btn_sizer)
        vbox.Add(panel4, 0, wx.EXPAND|wx.ALL, 10)

        self.SetSizer(vbox)
        vbox.Fit(self)

        self.Centre()
        self.ShowModal()

    def OnAmoeba(self, event):
        self.pop.Enable(False)
        self.crossover.Enable(False)
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)

    def OnDE(self, event):
        self.pop.Enable(True)
        self.crossover.Enable(True)
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)

    def OnDream(self, event):
        self.pop.Enable(True)
        self.crossover.Enable(False)
        self.burn.Enable(True)
        self.tmin.Enable(False)
        self.tmax.Enable(False)

    def OnPT(self, event):
        self.pop.Enable(True)
        self.crossover.Enable(True)
        self.burn.Enable(True)
        self.tmin.Enable(True)
        self.tmax.Enable(True)

    def OnRL(self, event):
        self.pop.Enable(True)
        self.crossover.Enable(True)
        self.burn.Enable(False)
        self.tmin.Enable(False)
        self.tmax.Enable(False)

    def OnFit(self, event):
        # send fit options based on algorithm selected
        if self.de_radio.GetValue():
            # de algorithm is selected, send all fit options related to de
            algo = 'de'
            steps = self.stepsize.GetValue()
            pop = self.pop.GetValue()
            cross = self.crossover.GetValue()
            fit_option = dict(steps=steps, pop=pop,cross=cross, algo=algo,)
            
            # send fit options to main panel to start fit
            pub.sendMessage("fit_option", fit_option)
            self.Destroy()
        
        if self.amoeba_radio.GetValue():
            # ameoba algorithm is selected, send all fit options related 
            # to ameoba
            algo = 'amoeba'
            steps = self.stepsize.GetValue()
            fit_option = dict(steps=steps, algo=algo,)
            
            # send fit options to main panel to start fit
            pub.sendMessage("fit_option", fit_option)
            self.Destroy()
            
        if self.dream_radio.GetValue():
            # dream algorithm is selected, send all fit options related to dream
            algo = 'dream'
            steps = self.stepsize.GetValue()
            burn = self.burn.GetValue()
            pop = self.pop.GetValue()
            fit_option = dict(steps=steps, pop=pop, burn=burn, algo=algo,)
            
            # send fit options to main panel to start fit
            pub.sendMessage("fit_option", fit_option)
            self.Destroy()
        
        if self.pt_radio.GetValue():
            # parallel temparing algorithm is selected, send all fit options 
            # related to parallel temparing
            algo = 'pt'
            steps = self.stepsize.GetValue()
            pop = self.pop.GetValue()
            tmin = self.tmin.GetValue()
            tmax = self.tmax.GetValue()
            burn = self.burn.GetValue()
            cross = self.crossover.GetValue()
            fit_option = dict(steps=steps, tmin=tmin, tmax=tmax, burn=burn,
                                      pop=pop, cross=cross, algo=algo,)
            
            # send fit options to main panel to start fit
            pub.sendMessage("fit_option", fit_option)
            self.Destroy()    
            
        if self.rl_radio.GetValue():
            # random lines algorithm is selected, send all fit options related 
            # to random lines
            algo = 'rl'
            steps = self.stepsize.GetValue()
            pop = self.pop.GetValue()
            cross = self.crossover.GetValue()
            fit_option = dict(steps=steps, pop=pop, cross=cross, algo=algo,)
            
            # send fit options to main panel to start fit
            pub.sendMessage("fit_option", fit_option)
            self.Destroy()    

    def OnCancel(self, event):
        # exit the fit control dialog box
        self.Destroy()

if __name__=="__main__":
    app = wx.App()
    FitControl(None, -1, 'Fit Control')
    app.MainLoop()
