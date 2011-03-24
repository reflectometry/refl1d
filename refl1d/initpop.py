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

def lhs(N, pars, include_current=False):
    """
    Latin Hypercube Sampling

    Returns an array whose columns each have *N* samples from equally spaced
    bins between *bounds*=(xmin, xmax) for the column.

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population, preserving the the
    LHS property.

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
        if include_current:
            # Put current value at position 0 in population
            s[0,j] = p.value
            # Find which bin the current value belongs in
            xidx = int(N*p.value/(xmax[j]-xmin[j]))
            # Generate random permutation of remaining bins
            idx = random.permutation(N-1)
            idx[idx>=xidx] += 1  # exclude current value bin
            # Assign random value within each bin
            P = (idx+ran[1:,j])/N
            s[1:,j] = xmin[j] + P*(xmax[j]-xmin[j])
        else:
            # Random permutation of bins
            idx = random.permutation(N)
            # Assign random value within each bin
            P = (idx+ran[:,j])/N
            s[:,j] = xmin[j] + P*(xmax[j]-xmin[j])

    return s

def cov(N, pars, include_current=False, cov=None, dx=None):
    """
    Initialize *N* sets of random variables from a gaussian model.

    The center is at *x* with an uncertainty ellipse specified by the
    1-sigma independent uncertainty values *dx* or the full covariance
    matrix uncertainty *cov*.

    For example, create an initial population for 20 sequences for a
    model with local minimum x with covariance matrix C::

        pop = cov_init(cov=C, pars=p, N=20)

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    x = array([p.value for p in pars])
    #return mean + dot(RNG.randn(N,len(mean)), chol(cov))
    if cov == None and dx == None:
        cov = eye(len(x))
    elif cov == None:
        cov = diag(asarray(dx)**2)
    population = random.multivariate_normal(mean=x, cov=cov, size=N)
    if include_current:
        population[0] = [p.value for p in pars]
    return population

def random(N, pars, include_current=False):
    """
    Generate a random population from the problem parameters.

    If include_current is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    population = [p.bounds.random(N) for p in pars]
    population = array(population).T
    if include_current:
        population[0] = [p.value for p in pars]
    return population

