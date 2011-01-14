from refl1d import preview
from refl1d.staj import MlayerModel
from refl1d.stajconvert import load_mlayer

import refl1d
refl1d.Probe.view = 'log'

def demo():
    staj = MlayerModel.load("xrayl523c6A15.staj")
    print staj
    M = load_mlayer("xrayl523c6A15.staj")
    #M = load_mlayer("wsh20029.staj")
    for i in range(1,len(M.sample)-1):
        M.sample[i].thickness.pmp(5)
        M.sample[i].interface.pmp(5)
        M.sample[i].material.rho.pmp(5)
        M.sample[i].material.irho.pmp(5)
    #preview(models=M)
    refl1d.fit(models=M)

if __name__ == "__main__":
    demo()
