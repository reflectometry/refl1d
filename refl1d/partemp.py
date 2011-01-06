from __future__ import division
import numpy
from numpy import asarray, zeros, ones, exp, diff, argmin, std
from numpy.random import rand,randn

def every_ten(step,x,fx):
    if step%10: print step, fx, x

class History(object):
    def __init__(self, filename):
        self.file= open(filename,'w')
        print >>self.file, "# Step Temp Energy Point"
    def __call__(self, step, P, E, T, A):
        for p,e,t,a in zip(P,E,T,A):
            if a: 
                pt = " ".join("%.6g"%v for v in p)
                print >>self.file,step,t,e,pt
        self.file.flush()

def parallel_tempering(nllf, p, bounds, T=None, steps=1000,
                       monitor=every_ten):
    log = lambda *args: 0
    #log = History("partemp.log")
    dT = diff(1./asarray(T))
    N = len(T)
    P = [p]*N
    bounder = ReflectBounds(*bounds)
    stepper = Stepper(bounds, tol=0.2/T[-1])
    E = ones(N)*nllf(p)
    total_accept = zeros(N)
    total_swap = zeros(N-1)
    Pbest = p
    Ebest = E[0]
    for step in range(steps):
        # Metropolis at each temperature
        Pnext = [stepper(p,t) for p,t in zip(P,T)]
        Pnext = [bounder.apply(p) for p in Pnext]
        Enext = asarray([nllf(p) for p in Pnext])
        #print step,"T",T
        #print step,"E",E
        #print step,"En",Enext
        #print "p",exp(-(Enext-E)/T)
        accept = exp(-(Enext-E)/T) > rand(N)
        E[accept] = Enext[accept]
        P = [(pn if a else p) for p,pn,a in zip(P,Pnext,accept)]
        total_accept += accept
        log(step, P,E,T,accept)

        idx = argmin(E)
        if E[idx] < Ebest:
            #print "update"
            Ebest = E[idx]
            Pbest = P[idx]

        # Swap chains across temperatures
        swap = zeros(N-1)
        for i in range(N-1):
            #print "swap",E[i+1]-E[i],dT[i],exp((E[i+1]-E[i])*dT[i])
            if exp((E[i+1]-E[i])*dT[i]) > rand(1)[0]:
                swap[i] = 1
                E[i],E[i+1] = E[i+1],E[i]
                P[i],P[i+1] = P[i+1],P[i]
        total_swap += swap


        # Monitoring
        monitor(step,Pbest,Ebest)
        interval = 100
        if 0 and step%interval == 0:
            print "max r",max(["%.1f"%numpy.linalg.norm(p-P[0]) for p in  P[1:]])
            #print "min AR",argmin(total_accept),min(total_accept)
            #print "min SR",argmin(total_swap),min(total_swap)
            print "AR",total_accept
            print "SR",total_swap
            print "s(d)",[int(std([p[i] for p in P])) for i in 3,7,11,-1]
            total_accept *= 0
            total_swap *= 0

    return Ebest, Pbest

class Stepper(object):
    def __init__(self, bounds, tol):
        low, high = bounds
        self.step = (high-low)*tol
    def __call__(self, p, t):
        return self.subspace(p,t,3)
    def subspace(self, p, t, ndim):
        n = len(self.step)
        if n < ndim:
            idx = slice(None)
            ndim = n
        else:
            idx = numpy.random.permutation(n)[:ndim]
        p = p+0
        p[idx] += randn(ndim)*self.step[idx]*t
        return p
    def full(self, p, t):
        return p + randn(len(p))*self.step*t
    def part(self, p, t, cr):
        while True:
            idx = rand(len(p))>cr
            if sum(idx)>0: break
        return p + idx*randn(len(p))*self.step*t

class ReflectBounds(object):
    """
    Reflect parameter values into bounded region
    """
    def __init__(self, low, high):
        self.low, self.high = [asarray(v,'d') for v in low, high]

    def apply(self, y):
        """
        Update x so all values lie within bounds

        Returns x for convenience.  E.g., y = bounds.apply(x+0)
        """
        minn, maxn = self.low, self.high
        # Reflect points which are out of bounds
        idx = y < minn; y[idx] = 2*minn[idx] - y[idx]
        idx = y > maxn; y[idx] = 2*maxn[idx] - y[idx]

        # Randomize points which are still out of bounds
        idx = (y < minn) | (y > maxn)
        y[idx] = minn[idx] + rand(sum(idx))*(maxn[idx]-minn[idx])
        return y
