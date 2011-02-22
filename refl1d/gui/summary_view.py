from __future__ import division
import wx

import  wx.lib.scrolledpanel as scrolled
from wx.lib.pubsub import Publisher as pub

from .util import nice

class SummaryView(scrolled.ScrolledPanel):
    def __init__(self, parent):
        """
        Model view showing summary of fit (only fittable parameters)
        """
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        self.parent = parent

        # set new model
        pub.subscribe(self.OnInitialModel, "initial_model")
        # change model structure message
        pub.subscribe(self.OnUpdateModel, "update_model")
        # change model parameter message
        pub.subscribe(self.OnUpdateParameters, "update_parameters")

        # event for showing notebook tab when it is clicked
        self.Bind(wx.EVT_SHOW, self.OnShow)

        # Keep track of whether the view needs to be redrawn
        self._reset_model = False
        self._reset_parameters = False

        self.SetAutoLayout(1)
        self.SetupScrolling()

    # ============= Signal bindings =========================
    def OnInitialModel(self, event):
        self.set_model(event.data)

    def OnUpdateModel(self, event):
        if self.model == event.data:
            self.update_model()

    def OnUpdateParameters(self, event):
        if self.model == event.data:
            self.update_parameters()
        else:
            print 'model and data donot match'

    def OnShow(self, event):
        if self._reset_model:
           self.update_model()
        elif self._reset_parameters:
           self.update_parameters()

    # ============ Operations on the model  ===============

    def set_model(self, model):
        self.model = model
        self.update_model()

    def update_model(self):
        #TODO not implemented but need to figure how to hide/show notebook tab
        #if not self.IsShown():
            #print "parameter tab is hidden"
            #self._reset_model = True
            #return

        self._reset_model = False
        self._reset_parameters = False

        bagSizer = wx.GridBagSizer(hgap=3, vgap=5)
        
        # Not an efficient way to making label align with the values in
        # GridBagSizer
        self.layer_label = wx.StaticText(self, wx.ID_ANY, 'Layer Name                                                          Value                     Low Range                        High Range')
        #self.slider_label = wx.StaticText(self, wx.ID_ANY, '            ')
        #self.value_label = wx.StaticText(self, wx.ID_ANY, 'Value')
        #self.low_label = wx.StaticText(self, wx.ID_ANY, 'Low Range')
        #self.high_label = wx.StaticText(self, wx.ID_ANY, 'High Range')

        line = wx.StaticLine(self, -1 )

        bagSizer.Add(self.layer_label, pos=(1,0),
                     flag=wx.LEFT, border=5)
        #bagSizer.Add(self.slider_label, pos=(1,1),
                     #flag=wx.LEFT, border=55)
        #bagSizer.Add(self.value_label, pos=(1,2),
                     #flag=wx.RIGHT, border=35)
        #bagSizer.Add(self.low_label, pos=(1,3),
                     #flag=wx.LEFT, border=55)
        #bagSizer.Add(self.high_label, (1,4))

        bagSizer.Add(line, pos=(2,0),flag=wx.EXPAND|wx.LEFT|wx.RIGHT, border=5)

        self.output = []

        for p in sorted(self.model.parameters, cmp=lambda x,y: cmp(x.name,y.name)):
            self.output.append(ParameterSummary(self, p, self.model))

        for index, item in enumerate(self.output):
            bagSizer.Add(item, pos = (index+3,0))

        self.SetSizerAndFit(bagSizer)
        

    def update_parameters(self):
        if not self.IsShown():
            self._reset_parameters = True
            return
        self._reset_parameters = False

        for p in self.output:
            p.update_slider()


class ParameterSummary(wx.Panel):
    def __init__(self, parent, parameter, model):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.parameter = parameter
        self.model = model

        self.low, self.high = (v for v in self.parameter.bounds.limits)

        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.layer_name = wx.StaticText(self, wx.ID_ANY,
                          str(self.parameter.name), style=wx.TE_LEFT )
        self.slider = wx.Slider(self, -1, 0, 0, 100,
                      size=(100, 8), style=wx.SL_AUTOTICKS|wx.SL_HORIZONTAL )
        self.slider.SetThumbLength(3)
        self.value = wx.StaticText(self, wx.ID_ANY, str(self.parameter.value),
                     style=wx.TE_LEFT )
        self.min_range = wx.StaticText(self, wx.ID_ANY, str(self.low),
                         style=wx.TE_LEFT )
        self.max_range = wx.StaticText(self, wx.ID_ANY, str(self.high),
                         style=wx.TE_LEFT )

        # add static box and slider to sizer

        text_hbox.Add(self.layer_name,1, wx.LEFT,1 )
        text_hbox.Add(self.slider, 1, wx.EXPAND|wx.LEFT,20)
        text_hbox.Add(self.value, 1, wx.EXPAND|wx.LEFT,20)
        text_hbox.Add(self.min_range, 1, wx.LEFT,1)
        text_hbox.Add(self.max_range, 1, wx.LEFT,1)

        self.SetSizer(text_hbox)

        self.slider.Bind(wx.EVT_SCROLL, self.OnScroll)
        self.update_slider()

    def update_slider(self):
        slider_pos = int(self.parameter.bounds.get01(self.parameter.value)*100)
        # May need the following if get01 doesn't protect against values out
        # of range
        #slider_pos = min(max(slider_pos,0),100)
        self.slider.SetValue(slider_pos)
        self.value.SetLabel(str(nice(self.parameter.value)))

    def OnScroll(self, event):
        value = self.slider.GetValue()
        new_value  = self.parameter.bounds.put01(value/100)
        self.parameter.value = new_value
        self.value.SetLabel(str(nice(new_value)))
        pub.sendMessage("update_parameters", self.model)

