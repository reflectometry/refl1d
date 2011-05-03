import wx
from wx.lib.scrolledpanel import ScrolledPanel

from ..gui.input_list import ItemListValidator


class LayerEditorDialog(wx.Dialog):
    """
    Select layer type and edit layer parameters interactively.
    """
    def __init__(self,
                 parent = None,
                 id = wx.ID_ANY,
                 title = "Layer Editor",
                 pos = wx.DefaultPosition,
                 size = wx.DefaultSize,
                 style = wx.DEFAULT_FRAME_STYLE, #|wx.THICK_FRAME|wx.RESIZE_BORDER,
                 name = "",
                 stack = None,
                 layer_num = 0):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)
        if parent is not None: self.SetFont(parent.GetFont())

        self.layer_id = wx.StaticText(self)
        self.total_thickness = wx.StaticText(self)

        self.next_button = wx.Button(self, wx.ID_ANY, "Next")
        self.prev_button = wx.Button(self, wx.ID_ANY, "Prev")
        self.ins_button = wx.Button(self, wx.ID_ANY, "Insert")
        self.del_button = wx.Button(self, wx.ID_ANY, "Delete")

        self.layer_name = wx.TextCtrl(self, wx.ID_ANY, size=(250,-1))

        self.selector_label = wx.StaticText(self, label="Type")
        self.selector = wx.Choice(self)

        self.thickness_label = wx.StaticText(self, label="Thickness")
        self.thickness = wx.TextCtrl(self, wx.ID_ANY, size=(100,-1))
        self.interface_label = wx.StaticText(self, label="Interface")
        self.interface = wx.TextCtrl(self, wx.ID_ANY, size=(100,-1))

        #self.dimension_label = wx.StaticText(self, label="size")
        #self.dimension = wx.TextCtrl(self, wx.ID_ANY,
        #                             value=str(dimension),
        #                             validator=ItemListValidator('int', True))

        # Generic bindings
        self.Bind(wx.EVT_BUTTON, self.OnNext, self.next_button)
        self.Bind(wx.EVT_BUTTON, self.OnPrev, self.prev_button)
        self.Bind(wx.EVT_BUTTON, self.OnInsert, self.ins_button)
        self.Bind(wx.EVT_BUTTON, self.OnDelete, self.del_button)
        self.Bind(wx.EVT_TEXT, self.OnValidate)
        #self.Bind(wx.EVT_COMBO_BOX, self.OnValidate)
        self.selector.Bind(wx.EVT_CHOICE, self.OnLayerType)

        # Layout
        outer = wx.BoxSizer(wx.VERTICAL)

        # Top line
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        bbox = wx.GridSizer(1,4)
        bbox.Add(self.next_button,1,wx.EXPAND)
        bbox.Add(self.prev_button,1,wx.EXPAND)
        bbox.Add(self.ins_button,1,wx.EXPAND)
        bbox.Add(self.del_button,1,wx.EXPAND)

        hbox.Add(self.layer_id, 0, wx.EXPAND)
        hbox.Add((0,0),1,wx.EXPAND)
        hbox.Add(bbox,0, wx.EXPAND)
        outer.Add(hbox, 0, wx.EXPAND)

        # Second, third line
        outer.Add(self.total_thickness, 0, wx.EXPAND)
        outer.Add(self.layer_name, 0, wx.EXPAND)

        # Fourth line
        bbox = wx.FlexGridSizer(3,2)
        bbox.AddMany([
            self.selector_label, (self.selector, 1, wx.EXPAND),
            self.thickness_label, (self.thickness, 1, wx.EXPAND),
            self.interface_label, (self.interface, 1, wx.EXPAND),
            ])
        outer.Add(bbox, 0, wx.EXPAND)
        self.SetSizer(outer)
        outer.Fit(self)

        # Set initial layer
        self.layer = None
        self.layer_num = layer_num
        self.set_stack(stack)

    def OnNext(self, event):
        self.set_layer(self.layer_num+1)
    def OnPrev(self, event):
        self.set_layer(self.layer_num-1)
    def OnDelete(self, event):
        del self.stack[self.layer_num]
        self.set_layer(self.layer_num)
    def OnInsert(self, event):
        slab = Slab(SLD(rho=1.0,name="layer"), thickness=100, interface=5)
        self.stack.insert(self.layer_num, slab)
        self.set_layer(self.layer_num)

    def OnLayerType(self, event):
        pass

    def set_stack(self, stack):
        if stack == None: return
        self.stack = stack
        self.set_layer(self.layer_num)

    def set_layer(self, layer_num):
        # Make sure the layer is valid
        N = len(self.stack)
        if layer_num < 0: layer_num = 0
        elif layer_num >= N: layer_num = N-1
        self.layer_num = layer_num

        # Enable navigation and control buttons
        self.next_button.Enable(layer_num < N-1)
        self.prev_button.Enable(layer_num > 0)
        self.del_button.Enable(N > 1)
        self.layer_id.SetLabel("Layer %d"%layer_num)
        self.total_thickness.SetLabel("Sample thickness %g"
                                      % self.stack.thickness.value)

        # Null case: no stack
        if N == 0: return # Should never happen...

        # Update the layer contents
        if self.layer != self.stack[layer_num]:
            self.layer = self.stack[layer_num]
            self.draw_layer()

    def draw_layer(self):
        if self.layer is None: return
        self.layer_name.SetValue(self.layer.name)
        self.thickness.SetValue("%g"%self.layer.thickness.value)
        self.interface.SetValue("%g"%self.layer.interface.value)
        #self.selector.SetValue(self.layer.__class__.__name__)

    def OnValidate(self, event):
        ctrl = event.GetEventObject()
        if ctrl:
            validator = ctrl.GetValidator()
            if validator:
                validator.Validate(ctrl)
        event.Skip()

class SlabEditor(ScrolledPanel):

    def __init__(self,
                 parent,
                 id       = wx.ID_ANY,
                 pos      = wx.DefaultPosition,
                 size     = wx.DefaultSize,
                 style    = wx.TAB_TRAVERSAL,
                 name     = "",
                 itemlist = [],
                 align    = False,
                 fontsize = None
                ):
        ScrolledPanel.__init__(self, parent, id, pos, size, style, name)

def main(stack):
    class App(wx.App):
        def OnInit(self):
            dia = LayerEditorDialog(None, -1, "simpledialog.py")
            dia.set_stack(stack)
            dia.ShowModal()
            dia.Destroy()
            return True
    app = App(0)
    app.MainLoop()

if __name__ == "__main__":
    from refl1d.names import *
    stack = silicon | air
    main(stack)
