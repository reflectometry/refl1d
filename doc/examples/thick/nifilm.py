from refl1d.names import *
nickel = Material('Ni')

#nickel = SLD(rho=9.4)

sample = silicon(0,5) | nickel(10000,5) | air

T = numpy.linspace(0, 5, 400)
dT,L,dL = 0.02,4.75, 0.0475

probe = NeutronProbe(T=T, dT=dT, L=L, dL=dL)
probe2 = NeutronProbe(T=T, dT=dT, L=L, dL=dL)
probe2.oversample(n=100)

M1 = Experiment(probe=probe, sample=sample, name="Ni layer w/o oversampling")
M2 = Experiment(probe=probe2, sample=sample, name="Ni layer w/ oversampling")
M3 = Experiment(probe=probe, sample=sample[1:], name="Ni substrate")

#M.simulate_data(5)

problem = MultiFitProblem([M1,M2,M3])
