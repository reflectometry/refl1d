import numpy
import pylab
from refl1d.fitter import load_problem, BFGSFit, DEFit, RLFit, PTFit

SEED = 1

def plot_model(filename):
    numpy.random.seed(SEED)
    p = load_problem('examples/'+filename)
    p.plot()
    pylab.show()

def fit_model(filename):
    numpy.random.seed(SEED)
    p =load_problem('examples/'+filename)
    #x = RLFit(p).solve(steps=1000, burn=99)
    #x = DEFit(p).solve(steps=1000, pop=20)
    x = PTFit(p).solve(steps=100,burn=400)
    #x = BFGSFit(p).solve(steps=200)
    chisq = p(x)
    print "chisq=",chisq
    if chisq>2:
        raise RuntimeError("Fit did not converge")
    p.plot()
    pylab.show()
    