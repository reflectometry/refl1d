"""
A base panel to draw the profile
"""

import os
import wx

import wx.lib.filebrowsebutton as filebrowse

import numpy
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import FigureManager
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg

from bumps.fitproblem import MultiFitProblem
from bumps.gui import signal

from refl1d.experiment import MixedExperiment
from refl1d.material import SLD
from refl1d.model import Slab

# from .binder import pixel_to_data
from .util import CopyImage
from .profilei import ProfileInteractor
from .interactor import BaseInteractor

IS_MAC = (wx.Platform == '__WXMAC__')

# ------------------------------------------------------------------------
class ModelView(wx.Panel):
    title = 'Profile'
    default_size = (600, 400)
    def __init__(self, *args, **kw):
        wx.Panel.__init__(self, *args, **kw)

        # Fig
        self.fig = Figure(figsize=(1, 1),
                          dpi=75,
                          facecolor='white',
                          edgecolor='white',
        )
        # Canvas
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.fig.set_canvas(self.canvas)

        # Axes
        self.axes = self.fig.add_axes(Subplot(self.fig, 111))
        self.axes.set_autoscale_on(False)
        self.theta_axes = self.axes.twinx()
        self.theta_axes.set_autoscale_on(False)

        # Show toolbar or not?
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)
        self.toolbar.Show(True)

        # Create a figure manager to manage things
        self.figmgr = FigureManager(self.canvas, 1, self)

        # Panel layout
        self.profile_selector_label = wx.StaticText(self, label="Sample")
        self.profile_selector = wx.Choice(self)
        self.profile_selector.Hide()
        self.profile_selector_label.Hide()
        self.Bind(wx.EVT_CHOICE, self.OnProfileSelect)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add( self.canvas,1, border=2, flag= wx.LEFT|wx.TOP|wx.GROW)
        self.tbsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.tbsizer.Add(self.toolbar, 0, wx.ALIGN_CENTER_VERTICAL)
        self.tbsizer.AddSpacer(20)
        self.tbsizer.Add(self.profile_selector_label,
                         0, wx.ALIGN_CENTER_VERTICAL)
        self.tbsizer.AddSpacer(5)
        self.tbsizer.Add(self.profile_selector,
                         0, wx.ALIGN_CENTER_VERTICAL)
        self.sizer.Add(self.tbsizer)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

        # Status bar
        frame = self.GetTopLevelParent()
        self.statusbar = frame.GetStatusBar()
        if self.statusbar == None:
            self.statusbar = frame.CreateStatusBar()
        status_update = lambda msg: self.statusbar.SetStatusText(msg)

        # Set the profile interactor
        self.profile = ProfileInteractor(self.axes, self.theta_axes,
                                         status_update=status_update)

        # Add context menu and keyboard support to canvas
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu)
        #self.canvas.Bind(wx.EVT_LEFT_DOWN, lambda evt: self.canvas.SetFocus())

        self.model = None
        self._need_set_model = self._need_redraw = False
        self.Bind(wx.EVT_SHOW, self.OnShow)

    def OnContextMenu(self, event):
        """
        Forward the context menu invocation to profile, if profile exists.
        """
        sx, sy = event.GetX(), event.GetY()
        #transform = self.axes.transData
        #data_x,data_y = pixel_to_data(transform, sx, self.fig.bbox.height-sy)
        #print "data_x: %s \rdata_y: %s " %(data_x,data_y)
        popup = wx.Menu()
        item = popup.Append(wx.ID_ANY, '&Grid on/off', 'Toggle grid lines')
        wx.EVT_MENU(self, item.GetId(),
                    lambda _: (self.axes.grid(), self.fig.canvas.draw_idle()))
        item = popup.Append(wx.ID_ANY, '&Rescale', 'Show entire profile')
        wx.EVT_MENU(self, item.GetId(),
                    lambda _: (self.profile.reset_limits(), self.profile.draw_idle()))

        item = popup.Append(wx.ID_ANY, "&Insert Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnInsert)
        item = popup.Append(wx.ID_ANY, "&Remove Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnRemove)
        item = popup.Append(wx.ID_ANY, "&Edit Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnEdit)
        item = popup.Append(wx.ID_ANY, "&Experiment")
        wx.EVT_MENU(self, item.GetId(), self.OnExperiment)
        self.PopupMenu(popup, (sx, sy))
        return False

    def OnInsert(self, event):
        #  Will not attempt to insert Layer before the current first layer
        if self.profile.layer_num != 0:
            a = LayerDialog(self, profile=self.profile, insert=True)
            a.ShowModal()
        else:
            signal.log_message("You cannot insert a layer before the first, current layer!")

    def OnRemove(self, event):
        a = LayerDialog(self, profile=self.profile)
        a.OnDelLayer(event)

    def OnEdit(self, event):
        self.layerobject = self.profile.experiment.sample[self.profile.layer_num]
        if type(self.layerobject) == Slab: #Only edit slabs
            a = LayerDialog(self,
                            profile=self.profile,
                            material=self.layerobject.material,
                            thickness=self.layerobject.thickness,
                            interface=self.layerobject.interface,
                            insert=False)
        elif type(self,layerobject) ==  1:
            print
        else:
            a = LayerDialog(self, profile=self.profile)
        a.ShowModal()

    def OnExperiment(self, event):
        self.experimentLayerObject = ExperimentDialog(experiment=self.profile.experiment, profile=self.profile)
        if self.experimentLayerObject.ShowModal() == wx.ID_OK:
            self.profile.signal_update()

    def OnProfileSelect(self, event):
        self.set_profile(*self.profiles[event.GetInt()])

    # ==== Model view interface ===
    def OnShow(self, event):
        if not event.Show: return
        #print "showing profile"
        if self._need_set_model:
            #print "-set model"
            self.set_model(self.model)
        elif self._need_redraw:
            #print "-redraw"
            self.profile.redraw()
        event.Skip()
    def get_state(self):
        return self.model
    def set_state(self, state):
        self.set_model(state)

    def set_model(self, model):
        self.model = model
        if not self.IsShown():
            self._need_set_model = True
        else:
            self._need_set_model = self._need_redraw = False
            self._set_model()

    def update_model(self, model):
        #print "profile update model"
        if self.model != model: return

        if not IS_MAC and not self.IsShown():
            self._need_set_model = True
        else:
            self._need_set_model = self._need_redraw = False
            self._set_model()

    def update_parameters(self, model):
        #print "profile update parameters"
        if self.model != model: return

        if not IS_MAC and not self.IsShown():
            self._need_redraw = True
        else:
            self._need_redraw = False
            self.profile.redraw()
    # =============================================

    def _set_model(self):
        """Initialize model by profile."""
        self.profiles = []
        def add_profiles(name, exp, idx):
            if isinstance(exp, MixedExperiment):
                for i, p in enumerate(exp.parts):
                    self.profiles.append((name + chr(ord("a") + i), p, idx))
            else:
                self.profiles.append((name, exp, idx))
        if isinstance(self.model, MultiFitProblem):
            for i, p in enumerate(self.model.models):
                if hasattr(p.fitness, "reflectivity"):
                    name = p.fitness.name
                    if not name: name = "M%d" % (i + 1)
                    add_profiles(name, p.fitness, i)
        else:
            add_profiles("", self.model.fitness, -1)

        self.profile_selector.Clear()
        if len(self.profiles) > 1:
            self.profile_selector.AppendItems([k for k, _, _ in self.profiles])
            self.profile_selector_label.Show()
            self.profile_selector.Show()
            self.profile_selector.SetSelection(0)
        else:
            self.profile_selector_label.Hide()
            self.profile_selector.Hide()

        self.set_profile(*self.profiles[0])

        # update the figure
        self.profile.redraw(reset_limits=True)


    def set_profile(self, name, experiment, idx):
        # Turn the model into a user interface
        # It is the responsibility of the party that is indicating
        # that a redraw is necessary to clear the precalculated
        # parts of the view; otherwise the theory function calculator
        # is going to be triggered twice.  This happens inside profile
        # before the profile is calculated.  Note that the profile
        # panel will receive its own signal, which will cause the
        # profile interactor to draw itself again.  We hope this isn't
        # too much of a problem.
        def signal_update():
            """Notify other views that the model has changed"""
            signal.update_parameters(model=self.model)
        def force_recalc():
            self.model.model_update()
        if isinstance(self.model, MultiFitProblem):
            self.model.set_active_model(idx)
        self.profile.set_experiment(experiment,
                                    force_recalc=force_recalc,
                                    signal_update=signal_update)

    def onPrinterSetup(self, event=None):
        self.canvas.Printer_Setup(event=event)

    def onPrinterPreview(self, event=None):
        self.canvas.Printer_Preview(event=event)

    def onPrint(self, event=None):
        self.canvas.Printer_Print(event=event)

    def OnSaveFigureMenu(self, evt):
        """
        Save the current figure as an image file
        """
        dlg = wx.FileDialog(self,
                            message="Save Figure As ...",
                            defaultDir=os.getcwd(),
                            defaultFile="",
                            wildcard="PNG files (*.png)|*.png|BMP files (*.bmp)|*.bmp|All files (*.*)|*.*",
                            style=wx.SAVE
        )
        _val = dlg.ShowModal()
        if _val == wx.ID_CANCEL:  return  #Do nothing
        if _val == wx.ID_OK:
            outfile = dlg.GetPath()
        dlg.Destroy()

        # Save
        self.fig.savefig(outfile)

    def GetToolBar(self):
        """
        backend_wx call this function. KEEP it
        """
        return self.toolbar

    def OnPanelFrameClose(self, evt):
        """
        On Close this Frame
        """
        self.Destroy()
        evt.Skip()

    def OnCopyFigureMenu(self, evt):
        """
        Copy the current figure
        """
        CopyImage(self.canvas)

    def CanShowContextMenu(self):
        return True

    def quit_on_error(self):
        numpy.seterr(all='raise')
        ProfileInteractor._debug = True
        BaseInteractor._debug = True

##===============================Validator class========================================###
#TODO: Add a validator to make sure that the user does not input unexpected/illegal values#
##===============================Validator class========================================###


class LayerDialog(wx.Dialog):
    '''
    Creates a dialog in which the user can edit or insert a layer into the current stack.
    '''
    def __init__(self,
                 parent=None,
                 id=wx.ID_ANY,
                 title="Layer",
                 pos=wx.DefaultPosition,
                 size=wx.DefaultSize,
                 style=wx.DEFAULT_DIALOG_STYLE,
                 profile=None,
                 material=None,
                 thickness=10.0,
                 interface=0.0,
                 insert=False):  # if this is true, the dialog will 'insert' on OK. If false, OK will cause an edit
        wx.Dialog.__init__(self, parent, id, title, pos, size, style)
        self.profile = profile

        if material != None:
            material_name = material.name
            try:
                rho = material.rho.value
            except AttributeError:
                rho = 0.
            try:
                irho = material.irho.value
            except AttributeError:
                irho = 0.
            try:
                thickness = thickness.value
            except AttributeError:
                thickness = 0.
            try:
                interface = interface.value
            except AtributeError:
                interface = 0.
        else:
            material_name = "Default Name"
            rho = 0.0
            irho = 0.0
            thickness = 10.0
            interface = 0.0

        # Sizers
        whole_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.GridBagSizer(3, 4)
        button_sizer = wx.StdDialogButtonSizer()

        # Content
        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, "Material Name:"), (1, 0), (1, 1))
        self.material_name = wx.TextCtrl(self, -1, str(material_name))
        content_sizer.Add(self.material_name, (1, 1), (1, 1))
        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, "rho:"), (2, 0), (1, 1))
        self.rho = wx.TextCtrl(self, -1, str(rho))
        content_sizer.Add(self.rho, (2, 1), (1, 1))
        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, "irho:"), (3, 0), (1, 1))
        self.irho = wx.TextCtrl(self, -1, str(irho))
        content_sizer.Add(self.irho, (3, 1), (1, 1))
        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, "Thickness:"), (1, 3), (1, 1))
        self.thickness = wx.TextCtrl(self, -1, str(thickness))
        content_sizer.Add(self.thickness, (1, 4), (1, 1))
        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, "Interface:"), (2, 3), (1, 1))
        self.interface = wx.TextCtrl(self, -1, str(interface))
        content_sizer.Add(self.interface, (2, 4), (1, 1))

        #Buttons
        ok_btn = wx.Button(self, wx.ID_OK, "Ok")
        ok_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")

        #Bind: check to see whether or not this is a insert dialog or an edit dialog.
        if insert:
            self.Bind(wx.EVT_BUTTON, self.OnInsertLayer, ok_btn)
        else:
            self.Bind(wx.EVT_BUTTON, self.OnEditLayer, ok_btn)

        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        button_sizer.AddButton(ok_btn)
        button_sizer.AddButton(cancel_btn)
        button_sizer.Realize()

        #Size it all in the whole_sizer
        whole_sizer.Add(content_sizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL,5)
        whole_sizer.Add(button_sizer, 0, wx.ALIGN_BOTTOM|wx.ALIGN_RIGHT|wx.ALL, 5)
        self.SetSizer(whole_sizer)
        self.Fit()

    def __GetSlabObject(self):
        """
        :returns the Slab Object that the should be edited or inserted into the layer.
        """
        # Strings values of the parameters
        namestr = str(self.material_name.GetValue())
        rhoft = float(self.rho.GetValue())
        irhoft = float(self.irho.GetValue())
        thicknessft = float(self.thickness.GetValue())
        interfaceft = float(self.interface.GetValue())
        # make an SLD object
        sldObject = SLD(name=namestr, rho=rhoft, irho=irhoft)
        #make a slab object
        slabObject = Slab(material=sldObject,
                          thickness=thicknessft,
                          interface=interfaceft,
                          name=None,
                          magnetism=None)
        return slabObject

    def __GetMagneticObject(self):
        """
        :return: the magnetic Object that should be edited or inserted into the layer
        """

    # Events for the buttons
    def OnDelLayer(self, event): #TODO debug: Sometimes you get an index out of bounds error
        """
        Deletes the layer the user clicked on.
        Cannot delete layer when there are only two inside of the stack.
        """
        del self.profile.experiment.sample[self.profile.layer_num]
        #if we are on the last layer, decrement layer_num
        if self.profile.layer_num == len(self.profile.experiment.sample):
            self.profile.layer_num -= 1
        self.ResetView()

    def OnInsertLayer(self, event):
        """
        Inserts a layer at the index before the mouse click in the stack.
        Cannot insert a layer before the substrate (first layer)
        """
        slabObject = self.__GetSlabObject()
        self.profile.experiment.sample.insert(self.profile.layer_num, slabObject)
        self.ResetView()
        self.Destroy()

    def OnCancel(self, event):
        """
        Exit the Layer Dialog.
        Dont do anything.
        """
        self.Destroy()

    def OnEditLayer(self, event):
        """
        Get the values of the current Slab object and place them into the parameters
        """
        slabObject = self.__GetSlabObject()
        self.profile.experiment.sample[self.profile.layer_num] = slabObject
        self.ResetView()

    def ResetView(self):
        """
        Reset the profile view so that changes take effect
        immediately after an event (i.e. delete, insert, or edit layer)
        """
        self.profile._find_layer_boundaries()
        self.profile.thickness_interactor.reset_markers()
        self.profile.update()
        self.Destroy()


REFL_FILES = "Refl files (*.refl)|*.refl"


class ExperimentDialog(wx.Dialog):
    """
     Create a Dialog for editing the current experiment and adding data.
     The user can edit the roughness limit for layers, the dz, the dA, and enable/disable
     step interfaces.
     The user can also edit the dRoR and the dQoQ
    """
    def __init__(self,
                 parent=None,
                 id=wx.ID_ANY,
                 title="Experiment...",
                 pos=wx.DefaultPosition,
                 size=wx.DefaultSize,
                 style=wx.DEFAULT_DIALOG_STYLE,
                 experiment=None,
                 profile=None):
        # Defualt Values
        self.experiment = experiment
        self.profile = profile
        roughness_limit = self.experiment.roughness_limit
        dz = self.experiment.dz
        dA = 0 if self.experiment.dA == None else self.experiment.dA

        step_interfaces = "Yes" if self.experiment.step_interfaces else "No"
        try:
            dRoR = self.experiment.probe.dRoR
        except AttributeError:
            dRoR = 0.
        try:
            dQoQ = self.experiment.probe.dQoQ
        except AttributeError:
            dQoQ = 0.
        self.current_file = os.getcwd()

        #  Create a dialog
        wx.Dialog.__init__(self, parent, id, title, pos, size, style)

        #sizers
        whole_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.GridBagSizer(6, 2)
        button_sizer = wx.StdDialogButtonSizer()

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " File Entry: "), (0, 0), (1, 1))
        self.file_text = wx.TextCtrl(self, wx.ID_ANY, self.current_file, size=(299, -1))
        content_sizer.Add(self.file_text, (0, 1), (1, 1))
        open_btn = wx.Button(self, wx.ID_OPEN, "Open", size=(-1, -1))
        content_sizer.Add(open_btn, (0, 2), (1, 1))
        self.Bind(wx.EVT_BUTTON, self.OnExperimentOpen, open_btn)

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " dRoR: "), (1, 0), (1, 1))
        self.dRoR = wx.TextCtrl(self, -1, str(dRoR))
        content_sizer.Add(self.dRoR, (1, 1), (1, 1))

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " dQoQ: "), (2, 0), (1, 1))
        self.dQoQ = wx.TextCtrl(self, -1, str(dQoQ))
        content_sizer.Add(self.dQoQ, (2, 1), (1, 1))

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " Use Step Interface: "), (3, 0), (1, 1))
        self.step_interfaces = wx.ComboBox(self, wx.ID_ANY, value=step_interfaces, pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           choices=['Yes', 'No'], style=wx.CB_DROPDOWN, name=wx.ComboBoxNameStr)
        content_sizer.Add(self.step_interfaces, (3, 1), (1, 1))

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " Roughness Limit: "), (4, 0), (1, 1))
        self.roughness_limit = wx.TextCtrl(self, -1, str(roughness_limit))
        content_sizer.Add(self.roughness_limit, (4, 1), (1, 1))

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " dA: "), (5, 0), (1, 1))
        self.dA = wx.TextCtrl(self, -1, str(dA))
        content_sizer.Add(self.dA, (5, 1), (1, 1))

        content_sizer.Add(wx.StaticText(self, wx.ID_ANY, " dz: "), (6, 0), (1, 1))
        self.dz = wx.TextCtrl(self, -1, str(dz))
        content_sizer.Add(self.dz, (6, 1), (1, 1))

        ok_btn = wx.Button(self, wx.ID_OK, "Ok")
        ok_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")

        #Binding
        self.Bind(wx.EVT_BUTTON, self.OnOk, ok_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        button_sizer.AddButton(ok_btn)
        button_sizer.AddButton(cancel_btn)
        button_sizer.Realize()

        #Size it all in the whole_sizer
        whole_sizer.Add(content_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        whole_sizer.Add(button_sizer, 0, wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT | wx.ALL, 5)
        self.SetSizer(whole_sizer)
        self.Fit()

    def OnOk(self, event):
        """
        Get user changes (input) and reset the experiment to accomodate the changes.
        :param event:
        :return:
        """
        # Get the values from the user input
        step_interfaces = (str(self.step_interfaces.GetValue()) == "Yes")
        roughness_limit = float(self.roughness_limit.GetValue())
        dA = float(self.dA.GetValue())
        dz = float(self.dz.GetValue())

        # Set the current experiment with the new values based on the users input
        self.experiment.step_interfaces = step_interfaces
        self.experiment.roughness_limit = roughness_limit
        self.experiment.dA = dA
        self.experiment.dz = dz
        if self.current_file != os.getcwd():
            self.experiment.probe = self.probe
        self.ResetView()

    def OnCancel(self, event):
        """
        Exit the Layer Dialog.
        Dont do anything.
        """
        self.Destroy()

    def OnExperimentOpen(self, evt):
        """
        Open a FileDialog and load data to be processed in the experiment.
        """
        dlg = wx.FileDialog(self,
                            message="Select File",
                            # defaultDir=os.getcwd(),
                            #defaultFile="",
                            wildcard=(REFL_FILES),
                            style=wx.OPEN | wx.CHANGE_DIR)
        status = dlg.ShowModal()
        path = dlg.GetPath()

        if status == wx.ID_OK:
            # print 'FileBrowseButton: %s\n' % (path)
            D = numpy.loadtxt(path)
            Q = D[:, 0]
            R = D[:, 1]
            if D.shape[1] < 3:
                self.dRoR = float(self.dRoR.GetValue())
                dR = R * self.dRoR / 100.
            else:
                dR = D[:, 2]
            if D.shape[1] < 4:
                self.dQoQ = float(self.dQoQ.GetValue())
                dQ = Q * self.dQoQ / 100.
            else:
                dQ = D[:, 3]
            from refl1d.probe import QProbe
            self.probe = QProbe(Q, dQ, data=(R, dR))
            print type(self.file_text)
            self.file_text.SetValue(path)
            dlg.Destroy()

    def ResetView(self):
        """
        Reset the profile view so that changes take effect
        immediately after an event (i.e. delete, insert, or edit layer)
        """
        self.profile._find_layer_boundaries()
        self.profile.thickness_interactor.reset_markers()
        self.profile.update()
        self.Destroy()