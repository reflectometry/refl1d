import sys; sys.path.append('../..')

import pylab
from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

# Data files
# TODO: set resolution of fixed region to opening 0.1
instrument = ncnrdata.ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)
probe = instrument.load('s5vho_Reduction1.refl')


# Sample description
Sub = SLD('Sub',rho=2.0725198, mu=0.000227402)
PtSi = SLD('PtSi',rho=4.597587, mu=0.009285)
SiO2 = SLD('SiO2',rho=3.468883, mu=0.000100808)
Pt_sub = SLD('Pt_sub',rho=6.352714, mu=0.0180725)
PtO_sub = SLD('PtO_sub',rho=6.192573, mu=0.0109804)
H2O_sub = SLD('H2O_sub',rho=-0.5598827, mu=0.0542424)
Nafion = SLD('Nafion',rho=3.34, mu=0.00180725)
H2O_top = SLD('H2O_top',rho=-0.5598827, mu=0.0542424)
PtO_top = SLD('PtO_top',rho=6.192573, mu=0.0109804)
Pt_top = SLD('Pt_top',rho=6.352714, mu=0.0180725)
PtO_air = SLD('PtO_air',rho=6.192573, mu=0.0109804)

sample = (Sub%0.85 + PtSi/11.4343%0.01 + SiO2/15.5%11 + Pt_sub/100%31.6 + Nafion/1200%0.43 + Pt_top/100%0.43 + air)

size = 6

# Experiment
experiment = Experiment(probe=probe, sample=sample)

print "Sample:",sample
experiment.format_parameters()

# Fitting parameters

if 1:  # composition
    for i in range(1,size):
        sample[i].material.rho.pmp(30)
        #sample[i].material.mu.range(0,0.1)

if 1:  # size
    for i in range(1,size):
        sample[i].thickness.pmp(20)
        #sample[i].thickness.value = 0

if 1:  # roughness
    for i in range(0,size):
        sample[i].interface.range(0,50)

preview(experiment)
#fit(experiment, npop=20)
