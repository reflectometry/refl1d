"""
Simple service/worker example

Service is sum(f(v)) for kernel f(v):v**2 and input v=1 2 3 4 5 6.
"""
from __future__ import with_statement
import sys; sys.path.append('../..')
from park.client import Job

def sum_service(env,input):
    import numpy
    if isinstance(input, int):
        v = xrange(input)
    else:
        v = input
    fv = env.mapper(v)
    return numpy.sum(fv)

def square_kernel(env,input):
    return lambda x: x**2

def sum(kernel, v, server=None):
    service = dict(name="park.examples.sumsq.sum_service",
                   input=v)
    job = Job(requires=[],service=service,kernel=kernel)
    return job.submit(server)

square = dict(name="park.examples.sumsq.square_kernel", input=None)

if __name__ == "__main__":
    import park.client
    #server = park.client.connect("http://sparkle.ncnr.nist.gov:8000")
    #job = sum(square, [1,2,3,4,5,6,7,8,9], server=server)
    #job = sum(lambda x: -x, [1,2,3,4,5,6])
    #job = sum(lambda x: 1, 1000000)
    with park.client.connect("http://sparkle.ncnr.nist.gov:8000"):
        job = sum(square, 100)
