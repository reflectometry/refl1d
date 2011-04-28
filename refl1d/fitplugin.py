
def new_model():
    import refl1d.names as refl
    stack = refl.silicon | refl.air
    instrument = refl.ncnr.NG1()
    probe = instrument.simulate(T=numpy.linspace(0,0.5,100))
    M = refl.Experiment(sample=stack, probe=probe)
    problem = refl.FitProblem(M)
    return problem
