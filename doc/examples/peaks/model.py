import numpy as np
from peaks import Fitness, Gaussian, Background
from refl1d.names import Parameter, pmath, FitProblem

def read_data():
#    data= Z1.T
    X = np.linspace(0.4840, 0.5080,13)
    Y = np.linspace(-0.5180,-0.4720,23)
    X,Y = np.meshgrid(X, Y)
    A = np.genfromtxt('XY_mesh2.txt',unpack=True)
    Z1 = A[26:39]
    data= Z1.T
    err=np.sqrt(data)
    #yerr= A[39:54]
    return X, Y, data, err

def build_problem():

    M = Fitness([Gaussian(name="G1-"),
                 Gaussian(name="G2-"),
                 Gaussian(name="G3-"),
                 Gaussian(name="G4-"),
                 Background()],
                *read_data())
    peak1 = M.parts[0]

    if 1:
        # Let peak centers and heights drift
        for peak in M.parts[:4]:
            peak.A.range(20,200)
            peak.xc.range(0.45,0.55)
            peak.yc.range(-0.55,-0.4)
    else:
        # Alternatively, peak centers follow a line 
        theta=Parameter(np.pi/4, name="theta")
        theta.range(np.pi/6,np.pi/2)
        peak1.xc.range(0.45,0.55)
        peak1.yc.range(-0.55,-0.4)
        for i,peak in enumerate(M.parts[1:4]):
            delta=Parameter(.0045, name="delta-%d"%(i+2))
            delta.range(0,0.006)
            peak.xc = peak1.xc + delta*pmath.cos(theta)
            peak.yc = peak1.yc + delta*pmath.sin(theta)

        # Let peak heights vary
        for peak in M.parts[:4]:
            peak.A.range(20,200)

    # Peak shape is the same across all peaks
    peak1.s1.range(0,0.012)
    peak1.s2.range(0,0.012)
    peak1.theta.range(-np.pi, 0)
    for peak in M.parts[1:4]:
        peak.s1 = peak1.s1
        peak.s2 = peak1.s2
        peak.theta = peak1.theta

    return FitProblem(M)

problem = build_problem()

