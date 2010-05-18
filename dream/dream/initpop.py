"""
Population initialization routines.

To start the analysis an initial population is required.  This will be
an array of size M x N, where M is the number of dimensions in the fitting
problem and N is the number of Markov chains.

Two functions are provided:

1. lhs_init(N, bounds) returns a latin hypercube sampling, which tests every
parameter at each of N levels.

2. cov_init(N, x, cov) returns a Gaussian sample along the ellipse 
defined by the covariance matrix, cov.  Covariance defaults to
diag(dx) if dx is provided as a parameter, or to I if it is not.

Additional options are a random box, RNG.rand(M,N) or RNG.randn(M,N)
where the random number generator (RNG) to use is numpy.random.
"""

from __future__ import division

__all__ = ['lhs_init', 'cov_init']

import numpy.random
from numpy import eye, diag, asarray

def lhs_init(N, bounds, RNG=numpy.random):
    """
    Latin Hypercube Sampling

    Returns an array whose columns each have *N* samples from equally spaced
    bins between *bounds*=(xmin, xmax) for the column.  DREAM bounds
    objects, with bounds.low and bounds.high can be used as well.  
    
    Note: Indefinite ranges are not supported.

    The additional parameter *RNG* is for conditions where you want to 
    control the random number generator to use.  It can, for example,
    be used with RNG=numpy.random.RandomState(seed) to always generate
    the same population.
    """
    try:
        xmin, xmax = bounds
    except:
        xmin, xmax = bounds.low, bounds.high
    
    # Define the size of xmin
    nvar = len(xmin)
    # Initialize array ran with random numbers
    ran = RNG.rand(N,nvar)

    # Initialize array s with zeros
    s = numpy.empty((N,nvar))

    # Now fill s
    for j in range(nvar):
        # Random permutation
        idx = RNG.permutation(N)+1
        P = (idx-ran[:,j])/N
        s[:,j] = xmin[j] + P*(xmax[j]-xmin[j])

    return s

def cov_init(N, x, cov=None, dx=None, RNG=numpy.random):
    """
    Initialize *N* sets of random variables from a center at *x* and a 
    covariance matrix *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, x=x, N=20)

    The additional parameter *RNG* is for conditions where you want to 
    control the random number generator to use.  It can, for example,
    be used with RNG=numpy.random.RandomState(seed) to always generate
    the same population.
    """
    #return mean + dot(RNG.randn(N,len(mean)), chol(cov))
    if cov == None and dx == None: 
        cov = eye(len(x))
    elif cov == None:
        cov = diag(asarray(dx)**2)
    return RNG.multivariate_normal(mean=x, cov=cov, size=N)

def demo():
    from numpy import arange
    print "Three ways of calling cov_init:"
    print "with cov",cov_init(N=4, x=[5,6], cov=diag([0.1,0.001]))
    print "with dx",cov_init(N=4, x=[5,6], dx=[0.1,0.001])
    print "with nothing",cov_init(N=4, x=[5,6])
    print """
The following array should have four columns.  Column 1 should have the
numbers from 10 to 19, column 2 from 20 to 29, etc.  The columns are in
random order with a random fractional part.
"""
    pop = lhs_init(N=10, bounds=(arange(1,5),arange(2,6)))*10

if __name__ == "__main__":
    demo()