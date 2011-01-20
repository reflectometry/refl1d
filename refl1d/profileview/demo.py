import wx

def demo(problem):
    from refl1d.profileview.panel import ProfileView
    app = wx.PySimpleApp(0)
    frame = wx.Frame(None, -1, "profile interactor")
    panel = ProfileView(frame)
    panel.quit_on_error()
    try: experiment = problem.fits[0].fitness
    except: experiment = problem.fitness
    panel.SetProfile(experiment)
    frame.Show(True)
    app.MainLoop()
