#!/usr/bin/env python

"""
The Rosenbrock banana function
"""
from dream import *
from pylab import *

def rosen(x):  
    x = asarray(x)
    return -sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

n=6
sampler = Dream(model=LogDensity(rosen),
                population=randn(20*n+4,n),
                thinning=1,
                generations=1000,
                )

state = sampler.sample()
#plot_corr(state); show()
plot_all(state)
