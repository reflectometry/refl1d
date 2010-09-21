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

sample = (silicon%7 + Cr/23%9 + PtFe/14.7%9 + NiFe/50.9%13 + Pt/32%11 + air)
#sample = silicon%7 + air

# Data files

ninstrument = snsdata.Liquids()
nprobes = ninstrument.simdata(sample, 
                              #slits=[0.02]*4,
                              #dLoL=0.02,
                              T=[0.09,0.4,1,2],
                              uncertainty = 0.1,
                              #theta_offset = 0.01, 
                              #background=1e-6,
                              #normalize=False,
                              #back_reflectivity=True,
                              )

xslits = 0.02*(ncnrdata.XRay.d_s1 - ncnrdata.XRay.d_s2)
xinstrument = ncnrdata.XRay(dLoL=0.005,Tlo=numpy.inf,slits_at_Tlo=xslits)
xprobe = xinstrument.simdata(sample, T=numpy.linspace(0.01,2.5,300))

for i,p in enumerate(nprobes):
    numpy.savetxt('neutron%d.dat'%i,numpy.vstack((p.Qo,p.Ro,p.dR)).T)
numpy.savetxt('xray.dat',numpy.vstack((xprobe.Qo,xprobe.Ro,xprobe.dR)).T)

if 0:
    for p in nprobes:
        p.plot(substrate=silicon, surface=air)
    pylab.xlabel("Q (inv A)")
    pylab.ylabel("Reflectivity")

    pylab.figure()
    xprobe.plot(substrate=silicon, surface=air)
    pylab.xlabel("Q (inv A)")
    pylab.ylabel("Reflectivity")

elif 1:
    M = Experiment(probe=ProbeSet(nprobes),sample=sample)
    #M.plot_profile()
    M.plot_reflectivity()
    #pylab.figure()
    #M = Experiment(probe=xprobe, sample=sample)
    #M.plot_reflectivity()
    #M.plot_profile()

elif 1:
    M = [Experiment(probe=p, sample=sample) for p in nprobes]
    for m in M:
        m.plot_reflectivity()
        pylab.hold(True)

pylab.show()
