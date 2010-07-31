import numpy

import park
from park.client import connect
from park.service.optimize import diffev, fitness


@park.export
def cost_kernel(env, input):
    from park.service.optimize import chisq
    x,y = numpy.loadtxt(env.get_workfile('data')).T
    dy = 1
    return lambda p: chisq(numpy.polyval(p,x), y, dy)
cost = dict(name="park.examples.fit.cost_kernel", input="",
            files=dict(data="mydata.txt"))
    



def main():
    # Fake data
    p = [-2,5,3]
    x = numpy.linspace(0,3,20)
    y = numpy.polyval(p,x)
    y += numpy.random.randn(*x.shape)*0.1

    numpy.savetxt('mydata.txt',numpy.vstack((x,y)).T)

    # cost function is chisq, with kernel f(p,x) -> y
    ##cost = lambda p: numpy.sum(abs(p))+1  #Fails
    #cost = lambda p: sum(abs(v) for v in p)+1
    #cost = fitness('numpy.polyval',x,y,dy=0.1)
    #cost = fitness(numpy.polyval,x,y,dy=0.1)

    parameters = ('p2',0,-10,10), ('p1',0,-10,10), ('p0',0,-10,10)
    with connect("http://sparkle.ncnr.nist.gov:8000"):
        job = diffev(cost,parameters,ftol=1e-5,maxiter=100,npop=10)
    print job.wait()

if __name__ == "__main__": 
    main()
