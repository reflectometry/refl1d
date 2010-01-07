import sys; sys.path.append('../..')

from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

# Data files
instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)
probe1 = instrument.load('n5hl1_Short.refl')
probe2 = instrument.load('n5hl2_Short.refl')

## Sample description
L1 = SLD('L1',rho=-0.35, mu=0.0001234)
L2 = SLD('L2',rho=2.231, mu=0.05296)
L3 = SLD('L3',rho=4.3775, mu=0.01305)
sample1 = (sapphire%0.85 + L1/480%31.6 + L2/790.286%0.43 + L3/555.35%0.43 + air)
sample2 = sample1[:]
sample2[3] = L3/555%.5

# Experiment
exp1 = Experiment(probe=probe1, sample=sample1)
exp2 = Experiment(probe=probe2, sample=sample2)

# Fitting parameters
layers = sample1[1:4] + sample2[3:4]
if 1:  # composition
    for L in layers:
        L.material.rho.pmp(10)
        L.material.mu.pmp(10)

if 1:  # size
    for L in layers:
        L.thickness.pmp(10)

if 1:  # roughness
    for L in layers:
        L.interface.pmp(10)

#preview((exp1,exp2))
#fit(models=(exp1,exp2), npop=20)
fit(models=exp1, npop=5)
