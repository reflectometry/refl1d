
import pylab
from numpy import linspace
from refl1d import *

instrument = ncnrdata.ANDR(Tlo=0.35, slits_at_Tlo=0.21)

# Materials
MgO = Material('MgO', bulk_density=3.58)
V = Material('V')
FeV = Mixture.bymass('Fe',3,'V',2)
Pd = Material('Pd')

# Sample description
V_FeV = V/14%3+FeV/35%5
sample = (MgO%3 + 14*V_FeV + V/14%3 + FeV/60%5 + 85*V_FeV + V/14%2
          + Pd/20%4 + air)

#data = instrument.load('12v2b006_nobkgr_corr.refl')
#experiment = Experiment(data, )

Probe.view = 'log'
print "Sample 1:",sample
plot_sample(sample)


# Sample 2 is just like sample 1, but it has a different thickness for
# the 4th layer.
sample2 = sample[:]
sample2[3] = sample[3]/200
print "Sample 2:",sample2
#pylab.figure(2); plot_sample(sample2)

pylab.show()
