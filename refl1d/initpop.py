"""
Population initialization routines.

To start the analysis an initial population is required.  This will be
an array of size M x N, where M is the number of dimensions in the fitting
problem and N is the number of individuals in the population.

Three functions are provided:

1. lhs_init(N, pars) returns a latin hypercube sampling, which tests every
parameter at each of N levels.

2. cov_init(N, pars, cov) returns a Gaussian sample along the ellipse
defined by the covariance matrix, cov.  Covariance defaults to
diag(dx) if dx is provided as a parameter, or to I if it is not.

3. rand_init(N, pars) returns a random population following the
prior distribution of the parameter values.

Additional options are random box: rand(M,N) or random scatter: randn(M,N).
"""

# Note: borrowed from DREAM and extended.

from __future__ import division

__all__ = ['lhs_init', 'cov_init', 'random_init']

from numpy import eye, diag, asarray, array, empty, random

def lhs_init(N, pars):
    """
    Latin Hypercube Sampling

    Returns an array whose columns each have *N* samples from equally spaced
    bins between *bounds*=(xmin, xmax) for the column.

    Note: Indefinite ranges are not supported.
    """
    
    xmin,xmax = zip(*[p.bounds.limits for p in pars])

    # Define the size of xmin
    nvar = len(xmin)

    # Initialize array ran with random numbers
    ran = random.rand(N,nvar)

    # Initialize array s with zeros
    s = empty((N,nvar))

    # Now fill s
    for j in range(nvar):
        # Random permutation
        idx = random.permutation(N)+1
        P = (idx-ran[:,j])/N
        s[:,j] = xmin[j] + P*(xmax[j]-xmin[j])

    return s

def cov_init(N, pars, cov=None, dx=None):
    """
    Initialize *N* sets of random variables from a gaussian model.

    The center is at *x* with an uncertainty ellipse specified by the
    1-sigma independent uncertainty values *dx* or the full covariance
    matrix uncertainty *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, pars=p, N=20)
    """
    x = array([p.value for p in pars])
    #return mean + dot(RNG.randn(N,len(mean)), chol(cov))
    if cov == None and dx == None:
        cov = eye(len(x))
    elif cov == None:
        cov = diag(asarray(dx)**2)
    population = random.multivariate_normal(mean=x, cov=cov, size=N)
    return population

def random_init(N, pars):
    """
    Generate a random population from the problem parameters.
    """
    population = [p.bounds.random(N-1) for p in pars]
    population.append([p.value for p in pars])
    population = array(population).T

    return population

