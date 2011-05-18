from refl1d.names import *

instrument = SNS.Liquids(Tlo=0.3,slits_at_Tlo=0.2,TOF_range=(12000,40000))
probes = [instrument.load('REF_L_%d.txt'%run, angle=T, intensity=I)
          for run,T,I in ((2893,0.3,2.93),
                          (2894,0.7,25.23),
                          (2895,1.5,127.88),
                          (2896,3.0,328.16))]
probe = ProbeSet(probes)

SiOx = SLD(name="SiOx",rho=3.8)
P2VP = SLD(name="P2VP",rho=1.7863)
dPS = SLD(name="dPS",rho=6.3086)
sample = silicon(0,10) | SiOx(17.2,10) | P2VP(20,7) | dPS(90,5) | air

# ==== Parameters ====
SiOx.rho.range(2.07,4.18) # Si -> SiO2
P2VP.rho.pmp(20)
dPS.rho.pmp(20)
for L in sample[1:-1]:
    L.thickness.pmp(50)
    L.interface.range(0,20)
for p in probes:
    p.intensity.pmp(5)
probes[0].theta_offset.pm(0.05)
for p in probes[1:]:
    p.theta_offset = probes[0].theta_offset


M = Experiment(sample=sample, probe=probe, name="DU 53")
problem = FitProblem(M)

