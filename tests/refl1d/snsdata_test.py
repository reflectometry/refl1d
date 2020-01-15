import os

import numpy as np
from numpy import sqrt
from numpy.linalg import norm
from numpy.random import randn

from refl1d.names import *

Probe.view = 'log' # log, linear, fresnel, or Q**4

# Measurement parameters
T = [1.0]
slits = (0.2, 0.2)
dLoL = 0.05

# Simulate a sample
SiO2 = Material('SiO2', density=2.634)
sample = silicon(0, 1) | SiO2(200, 2) | air


# Compute reflectivity with resolution and added noise
instrument = SNS.Liquids()
M = instrument.simulate(sample, T=T, slits=slits, dLoL=dLoL)
probe = M.probe.probes[0]
Q, R = M.reflectivity()
I = SNS.boltzmann_feather(probe.L, counts=1e6)
dR = sqrt(R/I)
R += randn(len(Q))*dR
probe.R, probe.dR = R, dR

#preview(models=M)

# Save to file
data = np.array((probe.Q, probe.dQ, probe.R, probe.dR, probe.L))
filename = 'liquids-SiO2.txt'
try:
    with open(filename, 'w') as outfile:
        outfile.write("""\
#F /SNSlocal/REF_L/2007_1_4B_SCI/2893/NeXus/REF_L_1001.nxs
#E 1174593179.87
#D 2007-03-22 15:52:59
#C Run Number: 1001
#C Title: 100 A SiO2 on Si
#C Notes: Fake data for 100A SiO2 on Si
#C Detector Angle: (1.0, 'degree')
#C Proton Charge: 35.6334991455

#S 1 Spectrum ID ('bank1', (87, 152))
#N 3
#L Q(inv Angstrom) dQ(inv Angstrom) R() dR() L(Angstrom)
""")
        np.savetxt(outfile, data.T)

    probe2 = SNS.load(filename, slits=slits)
finally:
    os.remove(filename)

assert norm(probe2.Q-probe.Q) < 3e-14, str(norm(probe2.Q-probe.Q))
assert norm(probe2.R-probe.R) < 3e-14, str(norm(probe2.R-probe.R))
assert norm(probe2.L-probe.L) < 3e-14, str(norm(probe2.L-probe.L))
assert norm(probe2.T-probe.T) < 3e-14, str(norm(probe2.T-probe.T))
assert norm(probe2.dQ-probe.dQ) < 3e-14, str(norm(probe2.dQ-probe.dQ))
assert norm(probe2.dR-probe.dR) < 3e-14, str(norm(probe2.dR-probe.dR))
assert norm(probe2.dL-probe.dL) < 3e-14, str(norm(probe2.dL-probe.dL))
assert norm(probe2.dT-probe.dT) < 3e-14, str(norm(probe2.dT-probe.dT))
