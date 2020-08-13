import wx

from .model_view import ModelView

def make_frame(experiment):
    frame = wx.Frame(None, -1, "profile interactor")
    panel = ModelView(frame)
    panel.quit_on_error()
    panel.set_job(experiment)
    frame.panel = panel
    return frame

def demo(experiment):
    app = wx.PySimpleApp(0)
    frame = make_frame(experiment)
    frame.Show(True)
    app.MainLoop()

def example_experiment():
    import numpy as np
    from ..names import Experiment, NeutronProbe, Material, silicon, air

    T = np.linspace(0,3,100)
    probe = NeutronProbe(L=4.75, dL=4.75*0.02, T=T, dT=T*0.01)
    chrome = Material("Cr")
    gold = Material("Au")
    sample = silicon(0,5) | chrome(80,5) | gold(50,5) | air
    return Experiment(probe=probe, sample=sample)

if __name__ == "__main__":
    demo(example_experiment())
