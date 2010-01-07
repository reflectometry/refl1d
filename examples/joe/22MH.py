import sys; sys.path.append('../..')

import pylab
from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

# Data files
# TODO: set resolution of fixed region to opening 0.1
instrument = ncnrdata.ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)
probe = instrument.load('n5hl2_Short.refl')

# Sample description
L1 = SLD('L1',rho=-0.35, mu=0.0001234)
L2 = SLD('L2',rho=2.231, mu=0.05296)
L3 = SLD('L3',rho=4.3775, mu=0.01305)
sample = (sapphire%0.85 + L1/480%31.6 + L2/790.286%0.43 + L3/555.35%0.43 + air)

# Experiment
experiment = Experiment(probe=probe, sample=sample)

print "Sample:",sample
experiment.format_parameters()
# Fitting parameters
if 1:  # composition
    for i in range(1,4):
        sample[i].material.rho.range(-1,5)
        #sample[i].material.mu.range(0,0.1)

if 1:  # size
    for i in range(1,4):
        sample[i].thickness.range(300,1000)
        #sample[i].thickness.value = 0

if 0:  # roughness
    for i in range(0,4):
        sample[i].interface.pmp(50)



#preview(experiment)
fit(experiment, npop=20)
