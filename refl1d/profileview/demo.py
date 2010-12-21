import wx

def demo(job):
    from refl1d.profileview.panel import ProfileView
    app = wx.PySimpleApp(0)
    frame = wx.Frame(None, -1, "profile interactor")
    panel = ProfileView(frame)
    panel.quit_on_error()
    try: experiment = job.fits[0].fitness
    except: experiment = job.fitness
    panel.SetProfile(experiment)
    frame.Show(True)
    app.MainLoop()
