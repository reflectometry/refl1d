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
        submenu = wx.Menu()
        item = submenu.Append(wx.ID_ANY, "&Insert Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnInsert)
        item = submenu.Append(wx.ID_ANY, "&Remove Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnRemove)
        item = submenu.Append(wx.ID_ANY, "&Edit Layer")
        wx.EVT_MENU(self, item.GetId(), self.OnEdit)
        popup.AppendMenu(wx.ID_ANY, "Layer", submenu)
        item = popup.Append(wx.ID_ANY, "&Experiment")
        wx.EVT_MENU(self, item.GetId(), self.OnExperiment)
        self.PopupMenu(popup, (sx, sy))
        return False

    def OnInsert(self, event):
        #  Will not attempt to insert Layer
        #  before the current first layer
        if self.profile.layer_num != 0:
            ld = LayerDialog(self, profile=self.profile, insert=True)
            ld.ShowModal()
        else:
            print "You cannot insert a layer before the first layer!"
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

##===============================Validator class===============================##
'''
ALPHA_ONLY = 1
DIGIT_ONLY = 2

class MyValidator(wx.PyValidator):
    def __init__(self, flag=None, pyVar=None):
        wx.PyValidator.__init__(self)
        self.flag = flag
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def Clone(self):
        return MyValidator(self.flag)

    def Validate(self, win):
        tc = self.GetWindow()
        val = tc.GetValue()

        if self.flag == ALPHA_ONLY:
            for x in val:
                if x not in string.letters:
                    return False

        elif self.flag == DIGIT_ONLY:
            for x in val:
                if x not in string.digits:
                    return False

        return True


    def OnChar(self, event):
        key = event.GetKeyCode()

        if key < wx.WXK_SPACE or key == wx.WXK_DELETE or key > 255:
            event.Skip()
            return

        if self.flag == ALPHA_ONLY and chr(key) in string.letters:
            event.Skip()
            return

        if self.flag == DIGIT_ONLY and chr(key) in string.digits:
            event.Skip()
            return

        if not wx.Validator_IsSilent():
            wx.Bell()

        # Returning without calling even.Skip eats the event before it
        # gets to the text control
        return
'''
##===============================Validator class===============================##


class LayerDialog(wx.Dialog):
    '''
    This is a class in which the user can enter, edit, and remove layers from the profile.
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
                 thickness=0.0,
                 interface=0.0,
                 insert=False):  # if this is true, the dialog will 'insert' on OK. If false, OK will cause an edit
        wx.Dialog.__init__(self, parent, id, title, pos, size, style)
        self.profile = profile

        if material != None:
            materialName = material.name
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
            materialName = "*name*"
            rho = 0
            irho = 0
            thickness = 0
            interface = 0

        # Sizers
        self.wholeSizer = wx.BoxSizer(wx.VERTICAL)
        self.contentSizer = wx.GridBagSizer(3, 4)
        self.buttonSizer = wx.StdDialogButtonSizer()

        # Content
        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, "Name:"), (1, 0), (1, 1))
        self.materialName = wx.TextCtrl(self, -1, str(materialName))
        self.contentSizer.Add(self.materialName, (1, 1), (1, 1))
        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, "rho:"), (2, 0), (1, 1))
        self.rho = wx.TextCtrl(self, -1, str(rho))
        self.contentSizer.Add(self.rho, (2, 1), (1, 1))
        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, "irho:"), (3, 0), (1, 1))
        self.irho = wx.TextCtrl(self, -1, str(irho))
        self.contentSizer.Add(self.irho, (3, 1), (1, 1))
        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, "Thickness:"), (1, 3), (1, 1))
        self.thickness = wx.TextCtrl(self, -1, str(thickness))
        self.contentSizer.Add(self.thickness, (1, 4), (1, 1))
        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, "Interface:"), (2, 3), (1, 1))
        self.interface = wx.TextCtrl(self, -1, str(interface))
        self.contentSizer.Add(self.interface, (2, 4), (1, 1))

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

        self.buttonSizer.AddButton(ok_btn)
        self.buttonSizer.AddButton(cancel_btn)
        self.buttonSizer.Realize()

        #Size it all in the wholeSizer
        self.wholeSizer.Add(self.contentSizer, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL,5)
        self.wholeSizer.Add(self.buttonSizer, 0, wx.ALIGN_BOTTOM|wx.ALIGN_RIGHT|wx.ALL, 5)
        self.SetSizer(self.wholeSizer)
        self.Fit()

    def __GetSlabObject(self):
        # Strings values of the parameters
        namestr = str(self.materialName.GetValue())
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

    # Events for the buttons
    def OnDelLayer(self, event): #TODO debug: Sometimes you get an index out of bounds error
        del self.profile.experiment.sample[self.profile.layer_num]
        #if we are on the last layer, decrement layer_num
        if self.profile.layer_num == len(self.profile.experiment.sample):
            self.profile.layer_num -= 1
        self.ResetView()

    def OnInsertLayer(self, event):
        slabObject = self.__GetSlabObject()
        self.profile.experiment.sample.insert(self.profile.layer_num, slabObject)
        self.ResetView()
        self.Destroy()

    def OnCancel(self, event):
        self.Destroy()

    def OnEditLayer(self, event):
        slabObject = self.__GetSlabObject()
        self.profile.experiment.sample[self.profile.layer_num] = slabObject
        self.ResetView()

    def ResetView(self):
        self.profile._find_layer_boundaries()
        self.profile.thickness_interactor.reset_markers()
        self.profile.update()
        self.Destroy()

'''
How to access paul's code for messing with layers:
examples:
slabObject = self.profile.experiment.sample[self.profile.layer_num] #__getitem__
del self.profile.experiment.sample[self.profile.layer_num] #__delitem__
self.profile.experiment.sample[self.profile.layer_num] = slabObject#__setitem__
'''

REFL_FILES = "Refl files (*.refl)|*.refl"


class ExperimentDialog(wx.Dialog):
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
        dA = self.experiment.dA
        step_interfaces = "Yes" if self.experiment.step_interfaces else "No"
        currentFile = os.getcwd()

        #  Create a dialog
        wx.Dialog.__init__(self, parent, id, title, pos, size, style)

        #sizers
        self.wholeSizer = wx.BoxSizer(wx.VERTICAL)
        self.contentSizer = wx.GridBagSizer(4, 2)
        self.buttonSizer = wx.StdDialogButtonSizer()

        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, " File Entry: "), (0, 0), (1, 1))
        self.fileText = wx.TextCtrl(self, wx.ID_ANY, currentFile, size=(299, -1))
        self.contentSizer.Add(self.fileText, (0, 1), (1, 1))
        open_btn = wx.Button(self, wx.ID_OPEN, "Open", size=(-1, -1))
        self.contentSizer.Add(open_btn, (0, 2), (1, 1))
        self.Bind(wx.EVT_BUTTON, self.OnExperimentOpen, open_btn)

        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, " Use Step Interface: "), (1, 0), (1, 1))
        self.step_interfaces = wx.ComboBox(self, wx.ID_ANY, value=step_interfaces, pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           choices=['Yes', 'No'], style=wx.CB_DROPDOWN, name=wx.ComboBoxNameStr)
        self.contentSizer.Add(self.step_interfaces, (1, 1), (1, 1))

        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, " Roughness Limit: "), (2, 0), (1, 1))
        self.roughness_limit = wx.TextCtrl(self, -1, str(roughness_limit))
        self.contentSizer.Add(self.roughness_limit, (2, 1), (1, 1))

        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, " dA: "), (3, 0), (1, 1))
        self.dA = wx.TextCtrl(self, -1, str(dA))
        self.contentSizer.Add(self.dA, (3, 1), (1, 1))

        self.contentSizer.Add(wx.StaticText(self, wx.ID_ANY, " dz: "), (4, 0), (1, 1))
        self.dz = wx.TextCtrl(self, -1, str(dz))
        self.contentSizer.Add(self.dz, (4, 1), (1, 1))

        ok_btn = wx.Button(self, wx.ID_OK, "Ok")
        ok_btn.SetDefault()
        cancel_btn = wx.Button(self, wx.ID_CANCEL, "Cancel")

        #Binding
        self.Bind(wx.EVT_BUTTON, self.OnOk, ok_btn)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        self.buttonSizer.AddButton(ok_btn)
        self.buttonSizer.AddButton(cancel_btn)
        self.buttonSizer.Realize()

        #Size it all in the wholeSizer
        self.wholeSizer.Add(self.contentSizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.wholeSizer.Add(self.buttonSizer, 0, wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT | wx.ALL, 5)
        self.SetSizer(self.wholeSizer)
        self.Fit()

    def OnOk(self, event):
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
        self.experiment.probe = self.probe
        self.ResetView()

        '''
        if step_interfaces == True:
            print "True"
            # approximate the interface using microslabs with step size *dz*
        else:
            print "False"
            # use the Nevot-Croce analytic expression for the interface between slabs.
        '''


    def OnCancel(self, event):
        self.Destroy()

    def OnExperimentOpen(self, evt):
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
                dRoR = 1
                dR = R * dRoR / 100.
            else:
                dR = D[:, 2]
            if D.shape[1] < 4:
                dQoQ = 1
                dQ = Q * dQoQ / 100.
            else:
                dQ = D[:, 3]
            from refl1d.probe import QProbe
            self.probe = QProbe(Q, dQ, data=(R, dR))
            print type(self.fileText)
            self.fileText.SetValue(path)

            dlg.Destroy()

    def ResetView(self):
        self.profile._find_layer_boundaries()
        self.profile.thickness_interactor.reset_markers()
        self.profile.update()
        self.Destroy()