from os.path import join as joinpath, dirname, exists
import tempfile
import os

import numpy as np
from numpy import radians

from bumps.util import pushdir
from refl1d.reflectivity import magnetic_amplitude as refl

H2K = 2.91451e-5
B2SLD = 2.31929e-06

def Rplot(Qz, R, format):
    import pylab
    pylab.hold(True)
    for name,xs in zip(('--','+-','-+','++'),R):
        Rxs = abs(xs)**2
        if (Rxs>1e-8).any():
            pylab.plot(Qz, Rxs, format, label=name)
    pylab.xlabel('$Q_z = 2k_{z0}$', size='large')
    pylab.ylabel('R')
    pylab.legend()
    
def rplot(Qz, R, format):
    import pylab
    pylab.hold(True)
    pylab.figure()
    for name,xs in zip(('++','+-','-+','--'),R):
        rr = xs.real
        if (rr>1e-8).any():
            pylab.plot(Qz, rr, format, label=name + 'r')
    pylab.legend()
    pylab.figure()
    for name,xs in zip(('++','+-','-+','--'),R):
        ri = xs.imag
        if (ri>1e-8).any():
            pylab.plot(Qz, ri, format, label=name + 'i')
    pylab.legend()
    
    pylab.figure()
    for name,xs in zip(('++','+-','-+','--'),R):
        phi = np.arctan2(xs.imag, xs.real)
        if (ri>1e-8).any():
            pylab.plot(Qz, phi, format, label=name + 'i')
    pylab.legend()

def compare(name, layers, Aguide):
    depth, rho, rhoM, thetaM = list(zip(*layers))

    NL = len(rho)-2
    NC = 1
    QS = 0.001
    DQ = 0.0001
    NQ = 300
    EPS = Aguide
    ROSUP = rho[-1] + rhoM[-1]
    ROSUM = rho[-1] - rhoM[-1]
    ROINP = rho[0]  +  rhoM[0]
    ROINM = rho[0]  -  rhoM[0]

    path = tempfile.gettempdir()
    gepore = joinpath(path, 'gepore')
    header = joinpath(path, 'inpt.d')
    layers = joinpath(path, 'tro.d')
    rm_real = joinpath(path, 'rrem.d')
    rm_imag = joinpath(path, 'rimm.d')
    rp_real = joinpath(path, 'rrep.d')
    rp_imag = joinpath(path, 'rimp.d')

    if not exists(gepore):
        gepore_source = joinpath(dirname(__file__), '..','..','refl1d','lib','gepore.f')
        status = os.system('gfortran -O2 -o %s %s'%(gepore,gepore_source))
        if status != 0:
            raise RuntimeError("Could not compile %r"%gepore_source)
        if not exists(gepore):
            raise RuntimeError("No gepore created in %r"%gepore)

    with open(layers, 'w') as fid:
        for T,BN,PN,THE in list(zip(depth,rho,rhoM,thetaM))[1:-1]:
            fid.write('%f %e %e %f %f\n'%(T,1e-6*BN,1e-6*PN,radians(THE),0.0))

    for IP in (0.0, 1.0):
        with open(header, 'w') as fid:
            fid.write('%d %d %f %f %d %f (%f,0.0) (%f,0.0) %e %e %e %e\n'
                      %(NL,NC,QS,DQ,NQ,radians(EPS),IP,1-IP,
                        1e-6*ROINP,1e-6*ROINM,1e-6*ROSUP,1e-6*ROSUM))
        with pushdir(path):
            status = os.system('./gepore >/dev/null')
            if status != 0:
                raise RuntimeError("Could not run gepore")
        rp = np.loadtxt(rp_real).T[1] + 1j*np.loadtxt(rp_imag).T[1]
        rm = np.loadtxt(rm_real).T[1] + 1j*np.loadtxt(rm_imag).T[1]
        if IP == 1.0:
            Rpp, Rpm = rp, rm
        else:
            Rmp, Rmm = rp, rm

    Qz = np.arange(NQ)*DQ+QS
    #Rplot(Qz, [Rpp, Rpm, Rmp, Rmm], '-'); import pylab; pylab.show(); return

    kz = Qz[::4]/2
    R = refl(kz, depth, rho, 0, rhoM, thetaM, 0, Aguide)

    Rplot(Qz, [Rmm, Rpm, Rmp, Rpp], '-'); Rplot(2*kz, R, '.'); import pylab; pylab.show(); return
    
    assert np.linalg.norm((R[0]-Rpp)/Rpp) < 1e-13, "fail ++ %s"%name
    assert np.linalg.norm((R[1]-Rpm)/Rpm) < 1e-13, "fail +- %s"%name
    assert np.linalg.norm((R[2]-Rmp)/Rmp) < 1e-13, "fail -+ %s"%name
    assert np.linalg.norm((R[3]-Rmm)/Rmm) < 1e-13, "fail -- %s"%name

def compare_phase(name, layers, Aguide):
    depth, rho, rhoM, thetaM = list(zip(*layers))
    NL = len(rho)-2
    NC = 1
    QS = 0.001
    DQ = 0.0001
    NQ = 300
    EPS = Aguide
    Qz = np.arange(NQ)*DQ+QS
    #Rplot(Qz, [Rpp, Rpm, Rmp, Rmm], '-'); import pylab; pylab.show(); return

    kz = Qz[::4]/2
    R = refl(kz, depth, rho, 0, rhoM, thetaM, 0, Aguide)

    rplot(2*kz, R, '-'); import pylab; pylab.show(); return

def simple():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM
        [ 0, 0.0, 0.0, 270],
        [200, 4.0, 1.0, 359.9],
        [200, 2.0, 1.0, 270],
        [ 0, 4.0, 0.0, 270],
    ]
    return "Si-Fe-Au-Air", layers, Aguide

def twist():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM
        [ 0, 2.1, 0.0, 270],
        [20, 8.0, 5.0, 270],
        [20, 8.0, 5.0, 220],
        [20, 8.0, 5.0, 180],
        [10, 4.5, 0.0, 270],
        [ 0, 0.0, 0.0, 270],
        ]
    return "twist", layers, Aguide

def magsub():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM
        [50, 8.0, 5.0, 270],
        [ 0, 2.1, 0.0, 270],
        [10, 4.5, 0.0, 270],
        [ 0, 0.0, 0.0, 270],
        ]
    return "magnetic substrate", layers, Aguide

def magsurf():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM
        [ 0, 0.0, 0.0, 270],
        [200, 4.0, 1.0, 0.01],
        [200, 2.0, 1.0, 270],
        [200, 4.0, 0.0, 270],
        ]
    return "magnetic surface", layers, Aguide
    
def Yaohua_example():
    Aguide = 270
    rhoB = B2SLD * 0.4 * 1e6
    layers = [
        # depth rho rhoM thetaM
        [ 0, 0.0, rhoB, 270],
        [ 200, 4.0, rhoB + 1.0, np.arctan2(-rhoB, 1.0)],
        [ 200, 2.0, rhoB + 1.0, 270],
        [ 0, 4.0, rhoB, 270 ],
        ]
    return "Yaohua example", layers, Aguide
    
def zf_Yaohua_example():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM
        [ 0, 0.0, 0.0, 270],
        [ 200, 4.0, 1.0, 0.0001],
        [ 200, 2.0, 1.0, 270],
        [ 0, 4.0, 0.0, 270 ],
        ]
    return "Yaohua example", layers, Aguide 

def demo():
    """run demo"""
    #compare(*simple())
    #compare(*twist())
    #compare(*magsub())
    #compare(*magsurf())
    compare(*Yaohua_example())

if __name__ == "__main__":
    demo()
