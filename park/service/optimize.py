"""
Optimization service

Takes an optimizer and a kernel, and returns the minimum.  If kernel is
fitness, then the optimizer returns the best fit.

***WARNING*** being able to import arbitrary symbols as service and kernel
is a security risk.  Possible solutions: list valid symbols in park.config; 
mark valid symbols with @park.export; have a directory of component definition
files; others?
"""
import sys
import numpy

import park
import park.client
from park.util import import_symbol

def chisq(fy, y, dy=1):
    return numpy.sum(((fy-y)/dy)**2)

@park.export
def fitness_kernel(env, input):
    x = numpy.array(input['x'])
    y = numpy.array(input['y'])
    dy = numpy.array(input['dy'])
    f = import_symbol(input['f']['name'])(env,input['f']['input'])
    return lambda p: chisq(f(p,x), y, dy)

def fitness(f,x,y,dy=1):
    f = park.client.make_kernel(f)
    x,y = list(x),list(y)
    dy = list(dy) if not numpy.isscalar(dy) else dy
    return dict(name='park.service.optimize.fitness_kernel',
                input=dict(x=x,y=y,dy=dy,f=f))


def diffev(f, parameters, 
           maxiter=None, ftol=5e-3,
           CR=0.5, F=2.0, npop=3, 
           crossover="c_exp", mutate="best1u"):
    if maxiter is None: maxiter = len(parameters)*100

    input = dict(CR=CR, F=F, npop=npop, crossover=crossover, mutate=mutate,
                 parameters=parameters)
    service = dict(name="park.service.optimize.diffev_service",
                   input=input)
    job = park.client.JobDescription(requires=[],service=service,kernel=f)
    return job.submit(park.client.default_server())
    #result = _diffev(service['input'],lambda _,v: map(f,v))
    #return park.client.CompletedJob(job,result,None)

def diffev_service(env,input):
    input = dict((str(k),v) for k,v in input.items()) # clean up unicode
    return _diffev(input, lambda f,v: env.mapper(v))
    
def _diffev(input, mapper):    
    from refl1d.mystic import stop
    from refl1d.mystic.optimizer import de
    from refl1d.mystic.solver import Minimizer
    from refl1d.mystic.problem import Function

    # Parameters
    parameters = input.pop('parameters')
    
    # Lookup crossover function
    if 'crossover' not in input: input['crossover'] = 'c_exp'
    if input['crossover'] not in de.CROSSOVER:
        raise ValueError('Invalid crossover '+input['crossover'])
    input['crossover'] = import_symbol("refl1d.mystic.optimizer.de."+input['crossover'])

    # Lookup mutate function
    if 'mutate' not in input: input['mutate'] = 'best1u'
    if input['mutate'] not in de.MUTATE:
        raise ValueError('Invalid mutater '+input['mutate'])
    input['mutate'] = import_symbol("refl1d.mystic.optimizer.de."+input['mutate'])

    # Stopping conditions
    ftol = input.pop('ftol',5e-3)
    maxiter = input.pop('maxiter',100*len(parameters))
    maxfun = maxiter*input['npop']
    success = stop.Cf(tol=ftol,scaled=False)
    failure = stop.Calls(maxfun)|stop.Steps(maxiter)

    _, po, lo, hi = zip(*parameters)
    bounds = zip(lo,hi)
    problem = Function(None, ndim=len(parameters), po=po, bounds=bounds)
    strategy = de.DifferentialEvolution(**input)
    minimize = Minimizer(strategy=strategy, problem=problem,
                         success=success, failure=failure)
    x = minimize(mapper=mapper)
    return list(x)
