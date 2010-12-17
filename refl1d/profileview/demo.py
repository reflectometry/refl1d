import wx

def demo(job):
    from refl1d.profileview.panel import ProfileView
    app = wx.PySimpleApp(0)
    frame = wx.Frame(None, -1, "profile interactor")
    panel = ProfileView(frame)
    panel.SetProfile(job.fitness)
    panel.profile.freeze_axes()
    frame.Show(True)
    app.MainLoop()
