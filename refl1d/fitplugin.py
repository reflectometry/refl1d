import numpy

def new_model():
    import refl1d.names as refl
    stack = refl.silicon(0,10) | refl.air
    instrument = refl.NCNR.NG1()
    probe = instrument.probe(T=numpy.linspace(0,5,200),
                             Tlo=0.2, slits_at_Tlo=2)
    M = refl.Experiment(sample=stack, probe=probe)
    problem = refl.FitProblem(M)
    return problem
