from refl1d.names import *
nickel = Material('Ni')
copper = Material('Cu')

#nickel = SLD(rho=9.4)

s_mid = Parameter(name='sigma mid', value=5)
s_low = s_mid/5
s_high = s_mid*3
#interface_below, interface_above = s_mid, s_mid
interface_below, interface_above = s_high, s_low
#interface_below, interface_above = s_low, s_high
#thetaM, thetaM2 = 90., 270.   # spin flip

thetaM, thetaM2 = 90., 90.   # non spin flip
sample = (
    silicon(0, s_mid)
    | nickel(100, s_mid, magnetism=Magnetism(rhoM=5, thetaM=thetaM,
                                             interface_below=interface_below,
                                             interface_above=interface_above))
    | copper(10, 5)
    | nickel(100, 5, magnetism=Magnetism(rhoM=5, thetaM=thetaM2))
    | copper(10, 5)
    | air
)
rev_sample = (
    air(0, sample[4].interface)
    | copper(sample[4].thickness, sample[3].interface)
    | nickel(sample[3].thickness, sample[2].interface,
             magnetism=Magnetism(rhoM=5, thetaM=thetaM2))
    | copper(sample[2].thickness, sample[1].interface)
    | nickel(sample[1].thickness, sample[0].interface,
             magnetism=Magnetism(rhoM=5, thetaM=thetaM,
                                 interface_below=interface_above,
                                 interface_above=interface_below))
    | silicon
)
#del sample[2:5]

T = numpy.linspace(0, 5, 400)
dT, L, dL = 0.02, 4.75, 0.0475

xs = [NeutronProbe(T=T*(-1 if name == "neg" else 1), dT=0.01, L=4.75, dL=0.0475, name=name)
      for name in ("Nevot-Croce", "fine", "coarse", "neg")]
# spin flip/non spin flip
if thetaM != thetaM2:
    probes = [PolarizedNeutronProbe([v, v, v, v], H=0, Aguide=270) for v in xs]
else:
    probes = [PolarizedNeutronProbe([v, None, None, v], H=0, Aguide=270) for v in xs]

nc, fine, coarse, neg = probes

M1 = Experiment(probe=nc, sample=sample, name="Nevot-Croce")
M2 = Experiment(probe=fine, sample=sample, name="fine step",
                step_interfaces=True, dz=0.1)
M3 = Experiment(probe=coarse, sample=sample, name="coarse step",
                step_interfaces=True, dz=2)
M4 = Experiment(probe=neg, sample=rev_sample, name="neg",
                step_interfaces=True, dz=2)

#M1.simulate_data(0.05)
#M2.simulate_data(0.05)

problem = FitProblem([M1, M2])
