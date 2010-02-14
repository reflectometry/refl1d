import sys; sys.path.append('../..')
from refl1d import preview
from refl1d.staj import MlayerMagnetic, MlayerModel
from refl1d.stajconvert import load_mlayer

import refl1d
refl1d.Probe.view = 'log'

def demo():
    print "----------------"
    print "Magnetic example"
    print "----------------"
    print MlayerMagnetic.load("n101G.sta")
    print "--------------------"
    print "Non-magnetic example"
    print "--------------------"
    print MlayerModel.load("xrayl523c6A15.staj")

    M = load_mlayer("xrayl523c6A15.staj")
    #M = load_mlayer("wsh20029.staj")
    preview(models=[M])

if __name__ == "__main__":
    demo()
