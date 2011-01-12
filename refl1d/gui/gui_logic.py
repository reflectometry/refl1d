import wx
import sys
import  wx.lib.scrolledpanel as scrolled
from wx.lib.pubsub import Publisher as pub
import wx.lib.newevent
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
