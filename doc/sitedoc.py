import numpy
import pylab
from refl1d.fitter import load_problem, BFGSFit, DEFit, RLFit, PTFit

SEED = 24

def plot_model(filename):
    numpy.random.seed(SEED)
    load_problem('examples/'+filename).plot()
    pylab.show()

def fit_model(filename):
    numpy.random.seed(SEED)
    p =load_problem('examples/'+filename)
    x = PTFit(p).solve(steps=20000)
    p.setp(x)
    p.plot()
    pylab.show()
    