import wx
import sys

import wx.gizmos as gizmos
from wx.lib.pubsub import Publisher as pub

from refl1d.mystic.parameter import Parameter, BaseParameter
from refl1d.profileview.panel import ProfileView

class ParameterView(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        #sizers
        vbox = wx.BoxSizer(wx.VERTICAL)
        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        # flag to set True if updation of parameter happened locally (this means
        # this view has been updated by the user). So we do not need to redraw 
        # the whole tree but we need to refresh the tree with new updated values.
        # As some other values may depend upon updated values.
        self.update_local = False
        
        self.tree = gizmos.TreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        | wx.TR_HAS_BUTTONS
                                        | wx.TR_TWIST_BUTTONS
                                        | wx.TR_ROW_LINES
                                        | wx.TR_COLUMN_LINES
                                        | wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                       )

        # create some columns
        self.tree.AddColumn("Model")
        self.tree.AddColumn("Fixed Value")
        self.tree.AddColumn("Layer Name")
        self.tree.AddColumn("Min Range")
        self.tree.AddColumn("Max Range")
        self.tree.AddColumn("Fittable")

        # Align the textctrl box with treelistctrl
        self.tree.SetMainColumn(0) # the one with the tree in it...
        self.tree.SetColumnWidth(0, 210)
        self.tree.SetColumnWidth(1, 185)
        self.tree.SetColumnWidth(2, 195)
        self.tree.SetColumnWidth(3, 195)
        self.tree.SetColumnWidth(4, 195)

        # making all colunm editable except first column
        self.tree.SetColumnEditable(0, False)
        self.tree.SetColumnEditable(1, True)
        self.tree.SetColumnEditable(2, True)
        self.tree.SetColumnEditable(3, True)
        self.tree.SetColumnEditable(4, True)

        # set new model
        pub.subscribe(self.OnInitialModel, "initial_model")
        # change model structure message
        pub.subscribe(self.OnUpdateModel, "update_model")
        # change model parameter message
        pub.subscribe(self.OnUpdateParameters, "update_parameters")
        
        self.tree.GetMainWindow().Bind(wx.EVT_RIGHT_UP, self.OnRightUp)
        self.tree.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit)
        self.tree.Bind(wx.EVT_TREE_ITEM_GETTOOLTIP,self.OnTreeTooltip)
        wx.EVT_MOTION(self.tree, self.OnMouseMotion) 
        
        vbox.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.SetAutoLayout(1)
        
    
    def OnTreeTooltip(self, event):
         print 'in tool tip'
         itemtext = self.tree.GetItemText(event.GetItem())
         event.SetToolTip("This is a ToolTip for %s!" % itemtext)
         event.Skip() 
         
    def OnMouseMotion(self, event):
        pos = event.GetPosition()
        item, flags, col = self.tree.HitTest(pos)
        """
        if wx.TREE_HITTEST_ONITEMLABEL:
            print 'in if'
            self.tree.SetToolTipString("tool tip")
        else:
            print 'in else'
            self.tree.SetToolTipString("")
        """
        event.Skip() 


    # ============= Signal bindings =========================

    def OnInitialModel(self, event):
        self.set_model(event.data)

    def OnUpdateModel(self, event):
        if self.model == event.data:
            # Delete the prevoius tree (if any)
            self.tree.DeleteAllItems()
            self.update_model()

    def OnUpdateParameters(self, event):
        if self.model == event.data:
            self.update_parameters()    
        else:
            print 'model donot match'

    # ============ Operations on the model  ===============

    def set_model(self, model):
        self.model = model
        self.update_model()

    def update_model(self):
        parameters = self.model.model_parameters()
        # Add a root node
        self.root = self.tree.AddRoot("Model")
        # Add nodes from our data set
        self.add_tree_nodes(self.root, parameters)
        self.update_tree_nodes()
        self.tree.ExpandAll(self.root)
        
    def update_parameters(self):
        if not self.update_local:
            self.tree.DeleteAllItems()
            self.update_model()
        else:
            # we need to refresh the tree with only updated values
            # TODO not full implemented yet
            self.update_tree_nodes()
            self.tree.Update()
            self.update_local = False
        

    def add_tree_nodes(self, branch, nodes):
        if isinstance(nodes,dict) and nodes != {}:
            for k in sorted(nodes.keys()):
                child = self.tree.AppendItem(branch, k)
                self.add_tree_nodes(child,nodes[k])
        elif ( ( isinstance(nodes, tuple) and nodes != () ) or
              ( isinstance(nodes, list) and nodes != [] ) ):
            for i,v in enumerate(nodes):
                child = self.tree.AppendItem(branch, '[%d]'%i)
                self.add_tree_nodes(child,v)

        elif isinstance(nodes, BaseParameter):
            self.tree.SetItemPyData(branch, nodes)
        
    def update_tree_nodes(self):
        node = self.tree.GetRootItem()
        while node.IsOk():
            self.set_leaf(node)
            node = self.tree.GetNext(node)
                    
    def set_leaf(self, branch):
        par = self.tree.GetItemPyData(branch)
        if par is None: return
        
        # for checking wheather the parameters are fittable or not.
        if par.fittable == True:
            fittable = 'Yes'
            low, high = (str(v) for v in par.bounds.limits)
        else:
            fittable = 'No'
            low, high = '', ''

        self.tree.SetItemText(branch, str(par.value), 1)
        self.tree.SetItemText(branch, str(par.name), 2)
        self.tree.SetItemText(branch, low, 3)
        self.tree.SetItemText(branch, high, 4)
        self.tree.SetItemText(branch, fittable, 5)


    def OnRightUp(self, evt):
        pos = evt.GetPosition()
        item, flags, col = self.tree.HitTest(pos)

    def OnEndEdit(self, evt):
        item = self.tree.GetSelection()
        self.node_object = self.tree.GetItemPyData(evt.GetItem())

        # Not an efficient way of updating values of Parameters
        # but its hard to find out which column changed during edit
        # operation. May be fixed in Future.
        wx.CallAfter(self.get_new_value, item, 1)
        wx.CallAfter(self.get_new_name, item, 2)
        wx.CallAfter(self.get_new_min, item, 3)
        wx.CallAfter(self.get_new_max, item, 4)

    def get_new_value(self, item, column):
        new_value = self.tree.GetItemText(item, column)

        # send update message to other tabs/panels only if parameter value
        # is updated
        if new_value != str(self.node_object.value):
            self.node_object.clip_set(float(new_value))
            self.update_local = True
            pub.sendMessage("update_parameters", self.model)

    def get_new_name(self, item, column):
        new_name = self.tree.GetItemText(item, column)

        # send update message to other tabs/panels only if parameter name
        # is updated
        if new_name != str(self.node_object.name):
            self.node_object.name = new_name
            self.update_local = True
            pub.sendMessage("update_parameters", self.model)


    def get_new_min(self, item, column):
        low = float(self.tree.GetItemText(item, column))
        high = self.node_object.bounds.limits[1]

        # send update message to other tabs/panels only if parameter min range
        # value is updated
        if low != self.node_object.bounds.limits[0]:
            self.node_object.range(low, high)
            self.update_local = True
            pub.sendMessage("update_parameters", self.model)

    def get_new_max(self, item, column):
        low = self.node_object.bounds.limits[0]
        high = float(self.tree.GetItemText(item, column))

        # send update message to other tabs/panels only if parameter max range
        # value is updated
        if high != self.node_object.bounds.limits[1]:
            self.node_object.range(low, high)
            self.update_local = True
            pub.sendMessage("update_parameters", self.model)

