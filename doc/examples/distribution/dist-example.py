from refl1d.names import *
from dist import Weights, DistributionExperiment

# Materials
nickel = Material('Ni', fitby='relative_density')

ridge = silicon(0,5) | nickel(1000,5) | air
valley = silicon(0,5) | air

T = numpy.linspace(0, 1.5, 200)
probe = NeutronProbe(T=T, dT=0.01, L=4.75, dL=0.0475)
M = Experiment(sample=ridge, probe=probe)

def mixture_cdf(x, loc=0, scale=1):
    r"""
    Returns a weighting function representing a certain portion of the
    sample having the nickel layer, a certain portion having no nickel
    layer and the parts in between with part nickel and part no-nickel.

    In the equation for a line $y=mx+b$, *loc* is $b$ and *scale* 
    is $m \gt 0$.
    
    We are interested in the region $[0,1]$, representing the relative 
    density of nickel.  If some portion $p_s$ of the sample is pure 
    silicon, then set $b=p_s$.  If some portion $p_n$ of the sample is
    pure nickel layer, then set $m = 1-p_n - b$.  If all portions of the
    sample are some mixture of the two, then $b \lt 0$ and $m \gt 1-b$.
    """
    return numpy.clip((x-loc)/scale, 0, 1)
edges = numpy.linspace(0,1,21)

dist = Weights(edges=numpy.linspace(0,1,41), truncated=False,
               cdf=mixture_cdf, loc=0.4, scale=0.2)
DM = DistributionExperiment(experiment=M, P=nickel.relative_density,
                           distribution=dist)
DM.simulate_data(0.05)
#DM.plot_weights()
#import pylab; pylab.figure()
#nickel.relative_density.value=0.5
#M.update()
problem = FitProblem(DM)