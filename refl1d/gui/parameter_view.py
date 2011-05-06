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
# Author: Nikunj Patel

"""
This module implements the Parameter View panel.
"""

#==============================================================================

import wx
import sys

import wx.gizmos as gizmos

from ..mystic.parameter import BaseParameter
from .util import nice, publish, subscribe


class ParameterView(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        #sizers
        vbox = wx.BoxSizer(wx.VERTICAL)
        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.tree = gizmos.TreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        | wx.TR_HAS_BUTTONS
                                        | wx.TR_TWIST_BUTTONS
                                        | wx.TR_ROW_LINES
                                        | wx.TR_COLUMN_LINES
                                        | wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                       )

        # Create columns.
        self.tree.AddColumn("Model")
        self.tree.AddColumn("Parameter")
        self.tree.AddColumn("Value")
        self.tree.AddColumn("Minimum")
        self.tree.AddColumn("Maximum")
        self.tree.AddColumn("Fit?")

        # Align the textctrl box with treelistctrl.
        self.tree.SetMainColumn(0) # the one with the tree in it...
        self.tree.SetColumnWidth(0, 200)
        self.tree.SetColumnWidth(1, 160)
        self.tree.SetColumnWidth(2, 80)
        self.tree.SetColumnWidth(3, 80)
        self.tree.SetColumnWidth(4, 80)
        self.tree.SetColumnWidth(5, 50)

        # Determine which colunms are editable.
        self.tree.SetColumnEditable(0, False)
        self.tree.SetColumnEditable(1, False)
        self.tree.SetColumnEditable(2, True)
        self.tree.SetColumnEditable(3, True)
        self.tree.SetColumnEditable(4, True)
        self.tree.SetColumnEditable(5, False)

        subscribe(self.OnModelChange, "model.change")
        subscribe(self.OnModelUpdate, "model.update")

        self.tree.GetMainWindow().Bind(wx.EVT_RIGHT_UP, self.OnRightUp)
        self.tree.Bind(wx.EVT_TREE_END_LABEL_EDIT, self.OnEndEdit)
        '''
        self.tree.Bind(wx.EVT_TREE_ITEM_GETTOOLTIP,self.OnTreeTooltip)
        wx.EVT_MOTION(self.tree, self.OnMouseMotion)
        '''

        vbox.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.SetAutoLayout(True)

    # ============= Signal bindings =========================

    def OnModelChange(self, model):
        if self.model == model:
            self.update_model()

    def OnModelUpdate(self, model):
        if self.model == model:
            self.update_parameters()

    '''
    def OnTreeTooltip(self, event):
         itemtext = self.tree.GetItemText(event.GetItem())
         event.SetToolTip("This is a ToolTip for %s!" % itemtext)
         event.Skip()

    def OnMouseMotion(self, event):
        pos = event.GetPosition()
        item, flags, col = self.tree.HitTest(pos)

        if wx.TREE_HITTEST_ONITEMLABEL:
            self.tree.SetToolTipString("tool tip")
        else:
            self.tree.SetToolTipString("")

        event.Skip()
    '''

    # ============ Operations on the model  ===============

    def set_model(self, model):
        self.model = model
        self.update_model()

    def update_model(self):
        # Delete the previous tree (if any).
        self.tree.DeleteAllItems()
        parameters = self.model.model_parameters()
        # Add a root node.
        self.root = self.tree.AddRoot("Model")
        # Add nodes from our data set .
        self.add_tree_nodes(self.root, parameters)
        self.update_tree_nodes()
        self.tree.ExpandAll(self.root)

    def update_parameters(self):
        self.update_tree_nodes()

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

        if par.fittable:
            if par.fixed:
                fitting_parameter = 'No'
                low, high = '', ''
            else:
                fitting_parameter = 'Yes'
                low, high = (str(v) for v in par.bounds.limits)
        else:
            fitting_parameter = ''
            low, high = '', ''

        self.tree.SetItemText(branch, str(par.name), 1)
        self.tree.SetItemText(branch, str(nice(par.value)), 2)
        self.tree.SetItemText(branch, low, 3)
        self.tree.SetItemText(branch, high, 4)
        self.tree.SetItemText(branch, fitting_parameter, 5)

    def OnRightUp(self, evt):
        pos = evt.GetPosition()
        branch, flags, column = self.tree.HitTest(pos)
        if column == 5:
            par = self.tree.GetItemPyData(branch)
            if par is None: return

            if par.fittable:
                fitting_parameter = self.tree.GetItemText(branch, column)
                if fitting_parameter == 'No':
                    par.fixed = False
                    fitting_parameter = 'Yes'
                    low, high = (str(v) for v in par.bounds.limits)
                elif fitting_parameter == 'Yes':
                    par.fixed = True
                    fitting_parameter = 'No'
                    low, high = '', ''

                self.tree.SetItemText(branch, low, 3)
                self.tree.SetItemText(branch, high, 4)
                self.tree.SetItemText(branch, fitting_parameter, 5)

                self.model.model_reset()  # force recalc on constraint change
                publish("model.change", model=self.model)

    def OnEndEdit(self, evt):
        item = self.tree.GetSelection()
        self.node_object = self.tree.GetItemPyData(evt.GetItem())
        # TODO: Not an efficient way of updating values of Parameters
        # but it is hard to find out which column changed during edit
        # operation. This may be fixed in the future.
        wx.CallAfter(self.get_new_name, item, 1)
        wx.CallAfter(self.get_new_value, item, 2)
        wx.CallAfter(self.get_new_min, item, 3)
        wx.CallAfter(self.get_new_max, item, 4)

    def get_new_value(self, item, column):
        new_value = self.tree.GetItemText(item, column)

        # Send update message to other tabs/panels only if parameter value
        # is updated .
        if new_value != str(self.node_object.value):
            self.node_object.clip_set(float(new_value))
            self.model.model_update()  # force recalc when value changes
            publish("model.update", model=self.model)

    def get_new_name(self, item, column):
        new_name = self.tree.GetItemText(item, column)

        # Send update message to other tabs/panels only if parameter name
        # is updated.
        if new_name != str(self.node_object.name):
            self.node_object.name = new_name
            publish("model.update", model=self.model)

    def get_new_min(self, item, column):
        low = self.tree.GetItemText(item, column)
        if low == '': return
        low = float(low)
        high = self.node_object.bounds.limits[1]

        # Send update message to other tabs/panels only if parameter min range
        # value is updated.
        if low != self.node_object.bounds.limits[0]:
            self.node_object.range(low, high)
            publish("model.update", model=self.model)

    def get_new_max(self, item, column):
        high = self.tree.GetItemText(item, column)
        if high == '': return
        low = self.node_object.bounds.limits[0]
        high = float(high)
        # Send update message to other tabs/panels only if parameter max range
        # value is updated.
        if high != self.node_object.bounds.limits[1]:
            self.node_object.range(low, high)
            publish("model.update", model=self.model)
