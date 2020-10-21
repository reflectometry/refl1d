import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#%matplotlib notebook

import refl1d
from refl1d.names import *

q = np.logspace(np.log10(0.005), np.log10(0.2), num=150)
dq = 0.02*q/2.35
probe = QProbe(q, dq)

sample = Slab(material=SLD(name='Si', rho=2.07), interface=0.3)
sample = sample | Slab(material=SLD(name='SiOx', rho=3.2), thickness=15, interface=0.8)
sample = sample | Slab(material=SLD(name='hPS', rho=1.4), thickness=1100, interface=25)
sample = sample | Slab(material=SLD(name='dPS', rho=6), thickness=500, interface=4)
sample = sample | Slab(material=SLD(name='air', rho=0))

expt = Experiment(probe=probe, sample=sample)

q1, r1 = expt.reflectivity()
probe.oversample(5)
expt.update()
q2, r2 = expt.reflectivity()
probe.oversample(10)
expt.update()
q3, r3 = expt.reflectivity()
probe.critical_edge()
expt.update()
q4, r4 = expt.reflectivity()

ax = plt.figure()
plt.plot(q1, r1, label='150 Q points')
plt.plot(q2, r2*2, label='oversample=5')
plt.plot(q3, r3*4, label='oversample=10')
plt.plot(q4, r4*8, label='critical edge')
plt.gca().legend()
plt.xlabel('Q [$1/\AA$]')
plt.ylabel('R')
plt.xscale('log')
plt.yscale('log')

plt.show()