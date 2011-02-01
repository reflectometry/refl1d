import wx
import sys
import  wx.gizmos   as  gizmos
import  wx.lib.scrolledpanel as scrolled
from wx.lib.pubsub import Publisher as pub
import wx.lib.newevent
from refl1d.mystic.parameter import Parameter, BaseParameter
from refl1d.profileview.panel import ProfileView

EVT_RESULT_ID = 1

class Fit_Tab(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent

        extra_space1 = wx.BoxSizer(wx.HORIZONTAL)
        extra_space1.Add((5,5), 1)

        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        label1 = wx.StaticText(self, 1, label="Store Folder:")

        self.Store_file = wx.TextCtrl(self, 2, value="",style=wx.TE_RIGHT)
        sizer1.Add(label1, 0, border=5,
                        flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)
        sizer1.Add(self.Store_file, 1, wx.EXPAND|wx.RIGHT, border=10)

        # Create the Compute button.
        self.btn_preview = wx.Button(self, wx.ID_ANY, "Preview")
        self.btn_preview.SetToolTip(wx.ToolTip("click to preview the plots"))
        self.Bind(wx.EVT_BUTTON, self.OnPreview, self.btn_preview)

        self.btn_fit = wx.Button(self, wx.ID_ANY, "Fit")
        self.btn_fit.SetToolTip(wx.ToolTip("click to start fit"))
        self.Bind(wx.EVT_BUTTON, self.OnFit, self.btn_fit)

        # Create a horizontal box sizer for the buttons.
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.Add((5,5), 1)  # stretchable whitespace
        sizer2.Add(self.btn_preview, 0)
        sizer2.Add((2,2), 1)  # stretchable whitespace
        sizer2.Add(self.btn_fit, 0)
        sizer2.Add((5,5), 1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(extra_space1, 0, wx.ALL,5)
        sizer.Add(sizer1, 0, wx.ALL, 5)
        sizer.Add(extra_space1, 0, wx.ALL,5)
        sizer.Add(sizer2, 0, wx.ALL, 5)

        self.SetSizer(sizer)

    def OnFit(self, event):
        btnLabel = self.btn_fit.GetLabel()
        if btnLabel == "Fit":
            self.btn_preview.Enable(False)
            self.btn_fit.SetLabel("Stop")
            self.t = 1
            ################LOGIC######################
            # send fit event message to panel with
            # all required data to fit
            # the panel will listen to event and start
            # the fit.
            ###########################################
            pub.sendMessage("fit", self.t)

        else:
            print 'stop logic goes here'
            self.btn_fit.SetLabel("Fit")
            pass


    def OnPreview(self, event):
        pass

class Log_tab(scrolled.ScrolledPanel):
    def __init__(self, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)

        INTRO_TEXT = "Fitting Progress Log:"

        self.intro_text = INTRO_TEXT
        self.log_text = wx.StaticText(self, wx.ID_ANY, label=INTRO_TEXT)

        # Create a horizontal box sizer to hold the title and progress bar.
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(self.log_text, 0, wx.ALIGN_CENTER_VERTICAL)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self.progress.Clear()
        vsizer.Add(sizer1, 0, wx.EXPAND|wx.ALL, border=10)
        vsizer.Add(self.progress, 1, wx.EXPAND)
        self.SetSizer(vsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()

        # subscribe to the log (fit update) message coming from the
        # main GUI panel and update the log tab with the log tab
        # with the log messages.
        pub.subscribe(self.Onlog, "log") # recieving fit message from fit tab


    def Onlog(self,event):
        space = "    "
        self.progress.AppendText('\n')
        self.progress.AppendText(space + event.data)

class Summary_tab(scrolled.ScrolledPanel):
    """
    Notebook tab for showing fit update logs
    """
    def __init__(self, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)

        INTRO_TEXT = "Fitting Parameter Summary:"

        self.intro_text = INTRO_TEXT
        self.log_text = wx.StaticText(self, wx.ID_ANY, label=INTRO_TEXT)

        # Create a horizontal box sizer to hold the title and progress bar.
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.Add(self.log_text, 0, wx.ALIGN_CENTER_VERTICAL)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        self.progress = wx.TextCtrl(self,-1,style=wx.TE_MULTILINE|wx.HSCROLL)
        self.progress.Clear()
        vsizer.Add(sizer1, 0, wx.EXPAND|wx.ALL, border=10)
        vsizer.Add(self.progress, 1, wx.EXPAND)
        self.SetSizer(vsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()

        # subscribe to the log (fit update) message coming from the
        # main GUI panel and update the log tab with the log tab
        # with the log messages.
        pub.subscribe(self.OnSummary, "summary") # recieving fit message from fit tab


    def OnSummary(self,event):
        self.progress.Clear()
        space = "    "
        self.progress.AppendText('\n')
        self.progress.AppendText(space + event.data)


class Parameter_Tab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        #sizers
        vbox = wx.BoxSizer(wx.VERTICAL)
        text_hbox = wx.BoxSizer(wx.HORIZONTAL)

        # create textctrl box and add to sizer
        self.empty_label = wx.StaticText(self, wx.ID_ANY, '            ')
        self.fixed_value = wx.TextCtrl(self, wx.ID_ANY, '')
        self.layer_name = wx.TextCtrl(self, wx.ID_ANY, '')
        self.min_range = wx.TextCtrl(self, wx.ID_ANY, '')
        self.max_range = wx.TextCtrl(self, wx.ID_ANY, '')
        self.fittable = wx.CheckBox(self, wx.ID_ANY, '')

        # add textctrl and checkbox to sizer
        text_hbox.Add(self.empty_label, 1, wx.LEFT|wx.RIGHT|wx.TOP|wx.BOTTOM,10)
        text_hbox.Add(self.fixed_value, 1, wx.LEFT|wx.RIGHT|wx.TOP|wx.BOTTOM,10)
        text_hbox.Add(self.layer_name, 1, wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.TOP,10)
        text_hbox.Add(self.min_range, 1, wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.TOP,10)
        text_hbox.Add(self.max_range, 1, wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.TOP,10)
        text_hbox.Add(self.fittable, 1, wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.TOP,10)
        vbox.Add(text_hbox, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 2)

        self.tree = gizmos.TreeListCtrl(self, -1, style =
                                        wx.TR_DEFAULT_STYLE
                                        | wx.TR_HAS_BUTTONS
                                        | wx.TR_TWIST_BUTTONS
                                        | wx.TR_ROW_LINES
                                        | wx.TR_COLUMN_LINES
                                        | wx.TR_NO_LINES
                                        | wx.TR_FULL_ROW_HIGHLIGHT
                                   )

        isz = (16,16)
        il = wx.ImageList(isz[0], isz[1])
        fldridx     = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FOLDER,      wx.ART_OTHER, isz))
        fldropenidx = il.Add(wx.ArtProvider_GetBitmap(wx.ART_FILE_OPEN,   wx.ART_OTHER, isz))
        fileidx     = il.Add(wx.ArtProvider_GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, isz))

        self.tree.SetImageList(il)
        self.il = il

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

        # got the parameter data from the script then load the treectlrlist
        pub.subscribe(self.OnParameter, "parameter") # recieving paramater message from fit tab

        self.tree.GetMainWindow().Bind(wx.EVT_RIGHT_UP, self.OnRightUp)
        self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.OnActivate)
        vbox.Add(self.tree, 1, wx.EXPAND)
        self.SetSizer(vbox)
        self.SetAutoLayout(1)


    def OnParameter(self, event):

        # Add a root node
        self.root = self.tree.AddRoot("Model")
        self.tree.Expand(self.root)
        # Add nodes from our data set
        self.AddTreeNodes(self.root, event.data)

    def AddTreeNodes(self, branch, nodes):

        if isinstance(nodes,dict) and nodes != {}:
            for k in sorted(nodes.keys()):
                child = self.tree.AppendItem(branch, k)
                self.AddTreeNodes(child,nodes[k])
        elif ( ( isinstance(nodes, tuple) and nodes != () ) or
              ( isinstance(nodes, list) and nodes != [] ) ):
            for i,v in enumerate(nodes):
                child = self.tree.AppendItem(branch, '[%d]'%i)
                self.AddTreeNodes(child,v)

        elif isinstance(nodes, BaseParameter):

            if nodes.fixed:
                low_bounds = ""
                up_bounds = ""
            else:
                low_bounds = nodes.bounds.limits[0]
                up_bounds = nodes.bounds.limits[1]

            self.tree.SetItemPyData(branch, nodes)
            self.tree.SetItemText(branch, str(nodes.value), 1)
            self.tree.SetItemText(branch, str(nodes.name), 2)
            self.tree.SetItemText(branch, str(low_bounds), 3)
            self.tree.SetItemText(branch, str(up_bounds), 4)


    def OnActivate(self, evt):
        pass


    def OnRightUp(self, evt):
        pos = evt.GetPosition()
        item, flags, col = self.tree.HitTest(pos)


def load_problem(args):
    file, options = args[0], args[1:]
    ctx = dict(__file__=file)
    argv = sys.argv
    sys.argv = [file] + options
    execfile(file, ctx) # 2.x
    sys.argv = argv
    try:
        problem = ctx["problem"]
    except AttributeError:
        raise ValueError(file+" does not define 'problem=FitProblem(models)'")
    problem.file = file
    problem.options = options
    return problem
