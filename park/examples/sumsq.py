"""
Simple service/worker example

Service is sum(f(v)) for kernel f(v):v**2 and input v=1 2 3 4 5 6.
"""
import sys; sys.path.append('../..')


def sum(env,input):
    import numpy
    v = input
    fv = env.mapper(v)
    return numpy.sum(fv)

def square(env,input):
    return lambda x: x**2

if __name__ == "__main__":
    import park.client
    server = park.client.default_server()
    service=dict(name="park.examples.sumsq.sum",
                 input=[1, 2, 3, 4, 5, 6])
    kernel=dict(name="park.examples.sumsq.square",input=None)
    jobid = server.submit(dict(requires=[],
                               service=service,
                               kernel=kernel))
