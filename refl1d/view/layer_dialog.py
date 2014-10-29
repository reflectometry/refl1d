import wx

from numpy import inf

from bumps.gui.input_list import InputListPanel

from refl1d.names import Slab, FreeLayer, FreeInterface, PolymerBrush
from refl1d.material import SLD

# name, type, description, *args
# type: 'string'
# type: 'material' (see below)
# type: 'parameter' with units and limits (see below)
# type: '[parameter]' is a vector of parameters with common properties

# 'material' is a scatterer, which is one of:
#
#   SLD, with scattering length density rho + irho
#   Material, with compound and natural density, which can be fit
#      by density, natural density, relative density or cell volume.
#      The compound may be a chemical formula, or it may be multiple
#      formulas, with a count for each.
#   Mixture, which is a set of scatterers with a base making up the
#      bulk and % making up the rest.  The % is by mass or by volume.
#      If by mass, each material needs a density.  The scatterers can
#      be SLDs, materials or other mixtures.
#
# The material name defaults to the chemical formula, if available.
#
# Users must be able to select a material from another layer, with
# fittable parameters automatically tied to that other layer, and a
# visual indication that this is so.
#
# A database of scatterers used in various experiments would be useful.
#
# The periodictable class can be used to lookup expected SLD values
# as in the SANS SLD calculator.
#
# 'parameter' has name, value, and can be fixed, fitted or computed.
#
# Fitted parameters have a range, which can be value pm v, value pm v%,
# value pm (-v1,v2), value pm (-v1,+v2)%, or simply [v1,v2], where value
# is the current parameter value.  Maybe we can do this in the input
# box for value.
#
# Computed values need a formula, which can be any math expression,
# with a popup box to select parameter Pn from a table of Pn:name:path.
#
# Parameters may be copied directly from other fields with the same units,
# presumably using a combobox to select the parameter.
#
# Some of these operations will require more information than just the
# current stack.

# These belong with the layer definition, so that users can add their own
# layer types more easily.  I will use monkey patching until the details
# of the interface are worked out.
Slab.name = 'slab'
Slab.description = 'layer with uniform density and roughness on top'
Slab.fields = (
    ('name','string','slab name (defaults to material name)'),
    ('material','material','slab material'),
    ('thickness','parameter','slab thickness','A',(0,inf)),
    ('interface','parameter',
     'rms roughness between this slab and the next','A',(0,inf)),
)

PolymerBrush.name = 'brush'
PolymerBrush.description = 'polymer brush in a solvent'
PolymerBrush.fields = (
    ('name','string','brush name (defaults to polymer name)'),
    ('polymer','material','polymer composition'),
    ('solvent','material','solvent composition'),
    ('thickness','parameter','brush+solvent thickness','A',(0,inf)),
    ('interface','parameter',
     'rms roughness between the solvent and the next layer','A',(0,inf)),
    ('base_vf','parameter',
     'volume fraction of the polymer brush at the base','%',(0,100)),
    ('base','parameter','thickness of the base region','A',(0,inf)),
    ('length','parameter','thickness of the thinning region','A',(0,inf)),
    ('power','parameter','rate of brush thinning','',(-inf,inf)),
    ('sigma','parameter','rms roughness within the brush','A',(0,inf)),
)

FreeLayer.name = 'freeform'
FreeLayer.description = 'freeform layer using monotonic splines for good control'
FreeLayer.fields = (
    ('name','string','layer name'),
    ('thickness','string','total layer thickness','A',(0,inf)),
    # The following need to be links to neighbouring layer parameters; easy
    # enough with slabs, but somewhat messy with something like a tethered
    # polymer.  Perhaps we can define below and above fields for all the
    # layer types which act as scatterers to make this happen, but for now
    # just assume it is attached to a slab.  Insert/Delete are going to
    # have to update
    ('below','material','material below the layer'),
    ('above','material','material above the layer'),
    # Need to indicate that these are three coordinated vectors
    ('z','[parameter]','control point location','A',(0,1)),
    ('rho','[parameter]',
     'scattering length density','1e-6 inv A^2',(-inf,inf)),
    ('irho','[parameter]',
     'complex scattering length density','1e-6 inv A^2',(0,inf)),
)

FreeInterface.name = 'blend'
FreeInterface.description = 'blend between two materials using a monotonic spline'
FreeInterface.fields = (
    ('name','string','layer name'),
    ('below','material','material below the layer'),
    ('above','material','material above the layer'),
    ('thickness','parameter','total layer thickness','A',(0,inf)),
    ('interface','parameter',
     'rms roughness between "above" and the next layer', 'A', (0,inf)),
    # Need to indicate that these are two coordinated vectors
    ('dz','[parameter]','relative segment size','',(0,inf)),
    ('dp','[parameter]','relative step height','',(0,inf)),
)

LAYERS = (Slab, FreeInterface, FreeLayer, PolymerBrush)


class LayerEditorDialog(wx.Dialog):
    """
    Select layer type and edit layer parameters interactively.
    """
    def __init__(self,
                 parent = None,
                 id = wx.ID_ANY, #@ReservedAssignment
                 title = "Layer Editor",
                 pos = wx.DefaultPosition,
                 size = (-1, 400),
                 style = wx.DEFAULT_FRAME_STYLE, #|wx.THICK_FRAME|wx.RESIZE_BORDER,
                 name = "",
                 stack = None,
                 layer_num = 0):
        wx.Dialog.__init__(self, parent, id, title, pos, size, style, name)
        if parent is not None: self.SetFont(parent.GetFont())

        self.next_button = wx.Button(self, wx.ID_ANY, "Next", style=wx.BU_EXACTFIT)
        self.prev_button = wx.Button(self, wx.ID_ANY, "Prev", style=wx.BU_EXACTFIT)
        self.ins_button = wx.Button(self, wx.ID_ANY, "Insert", style=wx.BU_EXACTFIT)
        self.del_button = wx.Button(self, wx.ID_ANY, "Delete", style=wx.BU_EXACTFIT)

        self.layer_id = wx.StaticText(self, size=(-1,-1))
        self.selector = wx.Choice(self)
        self.fields = InputListPanel(self)

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

        bbox = wx.GridSizer(rows=1, cols=4, vgap=0, hgap=5)
        bbox.Add(self.next_button, 1, wx.EXPAND)
        bbox.Add(self.prev_button, 1, wx.EXPAND)
        bbox.Add(self.ins_button, 1, wx.EXPAND)
        bbox.Add(self.del_button, 1, wx.EXPAND)
        outer.Add(bbox, 0, wx.ALIGN_LEFT|wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.layer_id, 0,
                 wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=5)
        hbox.Add(self.selector,0, wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL)
        hbox.Add((0,0), 1)
        outer.Add(hbox, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        outer.Add(self.fields, 1, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        outer.Add((-1,10))

        dx,dy = outer.GetMinSize()
        outer.SetMinSize((dx,dy+100))
        outer.Fit(self)
        self.SetSizer(outer)

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
        if stack is None: return
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
        self.layer_id.SetLabel("Layer %d:"%layer_num)

        # Null case: no stack
        if N == 0: return # Should never happen...

        # Update the layer contents
        if self.layer != self.stack[layer_num]:
            self.layer = self.stack[layer_num]
            self.draw_layer()

    def draw_layer(self):
        return
        if self.layer is None: return
        self.name_label = wx.StaticText(self, label="Name")
        self.name = wx.TextCtrl(self, wx.ID_ANY, size=(250,-1))

        self.thickness_label = wx.StaticText(self, label="Thickness")
        self.thickness = wx.TextCtrl(self, wx.ID_ANY, size=(100,-1))
        self.interface_label = wx.StaticText(self, label="Interface")
        self.interface = wx.TextCtrl(self, wx.ID_ANY, size=(100,-1))

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

def main(stack):
    class App(wx.App):
        def OnInit(self):
            dia = LayerEditorDialog(None, -1, "Mock-up")
            dia.set_stack(stack)
            dia.ShowModal()
            dia.Destroy()
            return True

    app = App(0)
    app.MainLoop()

if __name__ == "__main__":
    from refl1d.names import silicon, air
    stack = silicon | air
    main(stack)
