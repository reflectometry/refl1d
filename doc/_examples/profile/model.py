# Magic to import a module into the refl1d namespace
import os, imp; imp.load_source('refl1d.flayer',os.path.join(os.path.dirname(__file__),'flayer.py'))


from refl1d.names import *
from numpy import sin, pi

from refl1d.flayer import FunctionalProfile as FP


def profile(z, period, phase):
    return sin(2*pi*(z/period + phase))

sample = (silicon(0,5)
          | FP(100,0,name="sin",profile=profile, period=10, phase=1)
          | air)

sample['sin'].period.range(0,100)
sample['sin'].phase.range(0,1)
sample['sin'].thickness.range(0,1000)

T = numpy.linspace(0, 5, 100)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(probe=probe, sample=sample, dz=0.1)
M.simulate_data(5)
problem = FitProblem(M)
