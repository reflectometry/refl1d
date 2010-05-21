"""
The Rosenbrock banana function

Demonstration that sampling works even when the density is unstable.
"""
from dream import *
from pylab import *
from numpy.random import lognormal

def rosen(x):  
    x = asarray(x)
    s = sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
    return -lognormal(s,sqrt(s)) # Poisson style: variance = # counts
    

n=3
sampler = Dream(model=LogDensity(rosen),
                population=randn(20*n,n),
                thinning=1,
                generations=2000,
                )

state = sampler.sample()
#plot_corr(state); show()
plot_state(state, portion=0.5)
