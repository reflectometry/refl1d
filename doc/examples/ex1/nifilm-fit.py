"""
TOF reflectivity for a Ni film with fitted thickness.
"""

from refl1d.names import *

nickel = Material('Ni')
sample = silicon(0,10) | nickel(125,10) | air

sample[0].interface.range(3,12)
sample[1].interface.range(3,12)
sample[1].thickness.pmp(50)

instrument = SNS.Liquids()
files = ['nifilm-tof-%d.dat'%d for d in 1,2,3,4] 
probe = ProbeSet(instrument.load(f) for f in files)

M = Experiment(probe=probe, sample=sample)

problem = FitProblem(M)

