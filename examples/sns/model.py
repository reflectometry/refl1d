import numpy
import pylab
from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

# Sample description

Pt = Material('Pt')
NiFe = Mixture.bymass('Ni','Fe', 20)
PtFe = Mixture.bymass('Pt','Fe', 45)
Cr = Material('Cr')
substrate = silicon

sample = (silicon%10 + Cr/30%10 + PtFe/150%10 + NiFe/500%10 + Pt/30%10 + air)

sample[0].interface.pm(5)
for layer in sample[1:-1]:
    layer.thickness.pmp(30)
    layer.interface.pm(5)

# Data files

xslits = 0.02*(ncnrdata.XRay.d_s1 - ncnrdata.XRay.d_s2)
xinstrument = ncnrdata.XRay(dLoL=0.005,Tlo=numpy.inf,slits_at_Tlo=xslits)
xprobe = xinstrument.load('xray.dat')


M = Experiment(probe=xprobe, sample=sample)

# Needed by dream fitter
problem = FitProblem(M)
problem.dream_opts = dict(chains=20,draws=30000,burn=120000)
problem.name = "Example"
problem.title = "xray"
problem.store = "T3"
