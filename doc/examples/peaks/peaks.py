import sys
import numpy as np

from refl1d.mystic.parameter import Parameter


def plot(X,Y,theory,data,err):
    import pylab

    fig=pylab.gcf()
    ax=fig.add_subplot(3,1,2)
    pylab.pcolormesh(X,Y, theory)
    ax=fig.add_subplot(3,1,1)
    pylab.pcolormesh(X,Y, data)
    ax=fig.add_subplot(3,1,3)
    pylab.pcolormesh(X,Y, (data-theory)/(err+1))

class Gaussian(object):
    def __init__(self, A=1, xc=0, yc=0, s1=1, s2=1, theta=0, name=""):
        self.A = Parameter(A,name=name+"A")
        self.xc = Parameter(xc,name=name+"xc")
        self.yc = Parameter(yc,name=name+"yc")
        self.s1 = Parameter(s1,name=name+"s1")
        self.s2 = Parameter(s2,name=name+"s2")
        self.theta = Parameter(theta,name=name+"theta")

    def parameters(self):
        return dict(A=self.A,
                    xc=self.xc, yc=self.yc,
                    s1=self.s1, s2=self.s2,
                    theta=self.theta)

    def __call__(self, x, y):
        height=self.A.value
        s1=self.s1.value
        s2=self.s2.value
        t=self.theta.value
        x_center=self.xc.value
        y_center=self.yc.value
        a=(np.cos(t)**2)/(2*s1**2) + (np.sin(t)**2)/(2*s2**2)
        b=(-1)*np.sin(2*t)/(4*s1**2) + (1)*np.sin(2*t)/(4*s2**2)
        c=(np.sin(t)**2)/(2*s1**2) + (np.cos(t)**2)/(2*s2**2)
        normalization=1.0/(2*np.pi*s1*s2)
        Zf = np.exp( - (a*(x-x_center)**2 + 2*b*(x-x_center)*(y-y_center) + c*(y-y_center)**2))
        return Zf*np.abs(height)

class Background(object):
    def __init__(self, C=0, name=""):
        self.C = Parameter(C,name=name+"background")
    def parameters(self):
        return dict(C=self.C)
    def __call__(self, x, y):
        return self.C.value

class Fitness(object):
    def __init__(self, parts, X, Y, data, err):
        self.X,self.Y,self.data,self.err = X, Y, data, err
        self.parts = parts

    def numpoints(self):
        return np.prod(self.data.shape)

    def parameters(self):
        return [p.parameters() for p in self.parts]

    def theory(self):
        return sum(M(self.X,self.Y) for M in self.parts)

    def residuals(self):
        return (self.theory()-self.data)/self.err

    def nllf(self):
        return 0.5*np.sum(self.residuals()**2)

    def __call__(self):
        return 2*self.nllf()/self.dof

    def plot(self):
        plot(self.X, self.Y, self.theory(), self.data, self.err)

    def save(self, basename):
        pass

    def update(self):
        pass
