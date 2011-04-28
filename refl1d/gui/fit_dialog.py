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

from wx.lib.scrolledpanel import ScrolledPanel

from .input_list import InputListPanel
from .util import Validator

class FitControl(wx.Dialog):
    """
    FitControl lets the user set fitting options from a pop-up a dialog box.
    """
    def __init__(self,
                 parent = None,
                 id     = wx.ID_ANY,
                 title  = "Fit Parameters",
                 pos    = wx.DefaultPosition,
                 size   = wx.DefaultSize, # dialog box size will be calculated
                 style  = wx.DEFAULT_DIALOG_STYLE,
                 name   = "",
                 plist  = None,
                 default_algo = None,
                 fontsize = None
                ):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)

        self.para_list = plist
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.default_algo = default_algo

        # Set the font for this window and all child windows (widgets) from the
        # parent window, or from the system defaults if no parent is given.
        # A dialog box does not inherit font info from its parent, so we will
        # explicitly get it from the parent and apply it to the dialog box.
        if parent is not None:
            font = parent.GetFont()
            self.SetFont(font)

        # If the caller specifies a font size, override the default value.
        if fontsize is not None:
            font = self.GetFont()
            font.SetPointSize(fontsize)
            self.SetFont(font)

        # Section 1
        self.panel1 = wx.Panel(self, -1)
        static_box1 = wx.StaticBox(self.panel1, -1, "Fit Algorithms")

        self.algorithm = []

        if isinstance(self.para_list, dict) and self.para_list != {}:
            for algo in sorted(self.para_list.keys()):
                self.algorithm.append(algo)

        self.radio_list = []
        rows = (len(self.algorithm)+1)/2

        flexsizer = wx.FlexGridSizer(rows, 2, hgap=20, vgap=10)

        for algo in self.algorithm:
            self.radio = wx.RadioButton(self.panel1, -1, algo)
            self.radio_list.append(self.radio)
            self.Bind(wx.EVT_RADIOBUTTON, self.OnRadio, id=self.radio.GetId())
            flexsizer.Add(self.radio, 0, 0)

        fit_hsizer = wx.StaticBoxSizer(static_box1, orient=wx.VERTICAL)
        fit_hsizer.Add(flexsizer, 0, wx.ALL, 5)

        self.panel1.SetSizer(fit_hsizer)
        self.vbox.Add(self.panel1, 0, wx.ALL, 10)

        # Section 2
        # Create list of all panels for later use in hiding and showing panels.
        self.panel_list = []

        for idx, algo in enumerate(self.algorithm):
            parameters = self.para_list[algo]
            self.algo_panel = AlgorithmParameter(self, parameters, algo)
            self.panel_list.append(self.algo_panel)
            self.vbox.Add(self.algo_panel, 1, wx.EXPAND|wx.ALL, 10)
            self.algo_panel.Hide()

            if algo == self.default_algo:
                self.radio_list[idx].SetValue(True)
                self.panel_list[idx].Show()

        # Section 3
        # Create the button controls (Fit, OK, Cancel) and bind their events.
        #fit_btn = wx.Button(self, wx.ID_ANY, "Fit")
        ok_btn = wx.Button(self, wx.ID_OK, "OK")
        ok_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")

        #self.Bind(wx.EVT_BUTTON, self.OnFit, fit_btn)
        self.start_fit_flag = False
        self.Bind(wx.EVT_BUTTON, self.OnOk, ok_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        # Create the button sizer that will put the buttons in a row, right
        # justified, and with a fixed amount of space between them.  This
        # emulates the Windows convention for placing a set of buttons at the
        # bottom right of the window.
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add((10,20), 1)  # stretchable whitespace
        #btn_sizer.Add(fit_btn, 0)
        #btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
        btn_sizer.Add(ok_btn, 0)
        btn_sizer.Add((10,20), 0)  # non-stretchable whitespace
        btn_sizer.Add(cancel_btn, 0)

        # Add the button sizer to the main sizer.
        self.vbox.Add(btn_sizer, 0, border=10,
                      flag=wx.EXPAND|wx.TOP|wx.BOTTOM|wx.RIGHT)

        # Finalize the sizer and establish the dimensions of the dialog box.
        # The minimum width is explicitly set because the sizer is not able to
        # take into consideration the width of the enclosing frame's title.
        self.SetSizer(self.vbox)
        #self.vbox.SetMinSize((size[0], -1))
        self.vbox.Fit(self)

        self.Centre()

    def OnRadio(self, event):

        radio = event.GetEventObject()

        for btn_idx, btn_instance in enumerate(self.radio_list):
            if radio is btn_instance:
                break

        for panel in self.panel_list:
            if panel.IsShown():
                panel.Hide()
                self.panel_list[btn_idx].Show()
                self.vbox.Layout()
                break

    def OnFit(self, event):
        self.start_fit_flag = True
        #pub.sendMessage("start_fit")

    def OnOk(self, event):
        event.Skip()

    def OnCancel(self, event):
        event.Skip()

    def get_results(self):
        self.fit_option={}
        for algo_idx, algo in enumerate(self.algorithm):
            result = self.panel_list[algo_idx].fit_param.GetResults()
            self.fit_option[algo] = result
            if self.radio_list[algo_idx].GetValue():
                active_algo = algo

        flag = self.start_fit_flag
        self.start_fit_flag = False
        return active_algo, self.fit_option, flag


class AlgorithmParameter(wx.Panel):

    def __init__(self, parent, parameters, algo):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        fields = []
        sbox = wx.StaticBox(self, wx.ID_ANY, algo+" Fitting Parameters")

        for parameter in parameters:
            label, default_value, curr_value, datatype = parameter
            sub_list = [label, curr_value, datatype, 'RE', None]
            fields.append(sub_list)

        self.fit_param = InputListPanel(parent=self, itemlist=fields,
                                        align=True, size=(-1,220))

        sbox_sizer = wx.StaticBoxSizer(sbox, wx.VERTICAL)
        sbox_sizer.Add(self.fit_param, 1, wx.EXPAND|wx.ALL, 5)
        self.SetSizer(sbox_sizer)
        sbox_sizer.Fit(self)


if __name__=="__main__":
    app = wx.App()
    FitControl(None, -1, 'Fit Control')
    app.MainLoop()
