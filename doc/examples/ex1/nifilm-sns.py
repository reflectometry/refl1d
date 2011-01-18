from refl1d.names import *
    
nickel = Material('Ni')
sample = silicon(0,5) | nickel(100,5) | air

instrument = SNS.Liquids()
M = instrument.simulate(sample,
                        T=[0.09,0.3,1,3],
                        uncertainty = 1,
                        theta_offset = 0.01,
                        )
    
problem = FitProblem(M)

