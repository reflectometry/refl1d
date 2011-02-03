import wx

from .panel import ProfileView

import numpy
import png

def make_frame(experiment):
    frame = wx.Frame(None, -1, "profile interactor")
    panel = ProfileView(frame)
    panel.quit_on_error()
    panel.SetProfile(experiment)
    return frame

def demo(experiment):
    app = wx.PySimpleApp(0)
    frame = show_frame(experiment)
    frame.Show(True)
    app.MainLoop()

def example_experiment():
    import numpy
    from ..names import Experiment, NeutronProbe, Material, silicon, air

    T = numpy.linspace(0,3,100)
    probe = NeutronProbe(L=4.75, dL=4.75*0.02, T=T, dT=T*0.01)
    chrome = Material("Cr")
    gold = Material("Au")
    sample = silicon(0,5) | chrome(80,5) | gold(50,5) | air
    return Experiment(probe=probe, sample=sample)

if __name__ == "__main__":
    demo(example_experiment())
