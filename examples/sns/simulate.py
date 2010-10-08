import sys; sys.path.extend(('../..','../../dream','../../../reflectometry/build/lib.win32-2.5'))

#TODO: xray has smaller beam spot
# => smaller roughness
# => thickness depends on where the beam spot hits the sample
# Xray thickness variance = neutron roughness - xray roughness

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

sample = (silicon%7 + Cr/23%9 + PtFe/147%9 + NiFe/509%13 + Pt/32%11 + air)
#sample = silicon%7 + air

# Data files

ninstrument = snsdata.Liquids()
Mn = ninstrument.simulate(sample,
                          #slits=[0.02]*4,
                          #dLoL=0.02,
                          T=[0.09,0.4,1,2],
                          uncertainty = 0.1,
                          theta_offset = 0.01,
                          #background=1e-6,
                          #normalize=False,
                          #back_reflectivity=True,
                          )

xray = ncnrdata.XRay(dLoL=0.005,slits_at_Tlo=0.02)
Mx = xray.simulate(sample, T=numpy.linspace(0.01,2.5,100))

for i,p in enumerate(Mn.probe.probes):
    numpy.savetxt('neutron%d.dat'%i,numpy.vstack((p.Qo,p.Ro,p.dR)).T)
numpy.savetxt('xray.dat',numpy.vstack((Mx.probe.Qo,Mx.probe.Ro,Mx.probe.dR)).T)

if 1:
    #Mn.plot_profile()
    Mn.plot_reflectivity()
    #pylab.figure()
    Mx.plot_reflectivity()
    #Mx.plot_profile()

pylab.show()
