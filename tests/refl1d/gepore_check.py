from os.path import join as joinpath, dirname, exists
import tempfile
import os

import numpy as np
from numpy import radians

from bumps.util import pushdir
from refl1d.reflectivity import magnetic_amplitude as refl

H2K = 2.91451e-5
B2SLD = 2.31929e-06
GEPORE_SRC = 'gepore_zeeman.f'

def add_H(spec, H=0.0, theta_H=0.0, phi_H=0.0):
    """ Take H (vector) as input and add H to 4piM:
    In the parametrization of the Chatterji chapter, 
    phi_H is (90 - AGUIDE), and theta_H = 0
    """
    comment, layers, Aguide = spec
    new_layers = []
    for layer in layers:
        thickness, sld_n, sld_m, theta_m, phi_m = layer
        # we read phi_m, but it must be zero so we don't use it.
        sld_m_x = sld_m * np.cos(theta_m*np.pi/180.0) # phi_m = 0
        sld_m_y = sld_m * np.sin(theta_m*np.pi/180.0) # phi_m = 0
        sld_m_z = 0.0 # by Maxwell's equations, H_demag = mz so we'll just cancel it here
        sld_h = B2SLD * 1.0e6 * H        
        sld_h_x = sld_h * np.cos(theta_H * np.pi/180.0)
        sld_h_y = sld_h * np.sin(theta_H*np.pi/180.0)*np.cos(phi_H*np.pi/180.0)
        sld_h_z = sld_h * np.sin(phi_H * np.pi/180.0)*np.sin(phi_H*np.pi/180.0)
        sld_b_x = sld_h_x + sld_m_x
        sld_b_y = sld_h_y + sld_m_y
        sld_b_z = sld_h_z + sld_m_z
        sld_b = np.sqrt((sld_b_z)**2 + (sld_b_y)**2 + (sld_b_x)**2)
        theta_b = np.arctan2(sld_b_y, sld_b_x)
        theta_b = np.mod(theta_b, 2.0*np.pi)
        phi_b = np.arcsin(sld_b_z/sld_b)
        phi_b = np.mod(phi_b, 2.0*np.pi)
        new_layer = [thickness, sld_n, sld_b, theta_b*180.0/np.pi, phi_b*180.0/np.pi]
        new_layers.append(new_layer)
    return comment, new_layers, Aguide

def Rplot(Qz, R, format):
    import pylab
    pylab.hold(True)
    for name,xs in zip(('--','+-','-+','++'),R):
        Rxs = abs(xs)**2
        if (Rxs>1e-8).any():
            pylab.plot(Qz, Rxs, format, label=name)
    pylab.xlabel('$2k_{z0}$', size='large')
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
    depth, rho, rhoM, thetaM, phiM = list(zip(*layers))

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
        gepore_source = joinpath(dirname(__file__), '..','..','refl1d','lib',GEPORE_SRC)
        status = os.system('fort77 -O2 -o %s %s'%(gepore,gepore_source))
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
            status = os.system('./gepore') # >/dev/null')
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
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 270, 0],
        [200, 4.0, 1.0, 359.9, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [ 0, 4.0, 0.0, 270, 0.0],
    ]
    return "Si-Fe-Au-Air", layers, Aguide

def twist():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 2.1, 0.0, 270, 0.0],
        [20, 8.0, 5.0, 270, 0.0],
        [20, 8.0, 5.0, 220, 0.0],
        [20, 8.0, 5.0, 180, 0.0],
        [10, 4.5, 0.0, 270, 0.0],
        [ 0, 0.0, 0.0, 270, 0.0],
        ]
    return "twist", layers, Aguide

def magsub():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [50, 8.0, 5.0, 270, 0.0],
        [ 0, 2.1, 0.0, 270, 0.0],
        [10, 4.5, 0.0, 270, 0.0],
        [ 0, 0.0, 0.0, 270, 0.0],
        ]
    return "magnetic substrate", layers, Aguide

def magsurf():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 270, 0.0],
        [200, 4.0, 1.0, 0.01, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [200, 4.0, 0.0, 270, 0.0],
        ]
    return "magnetic surface", layers, Aguide

def NSF_example():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 270, 0.0],
        [200, 4.0, 1.0, 270, 0.0],
        [200, 2.0, 1.0, 270, 0.0],
        [200, 4.0, 0.0, 270, 0.0],
        ]
    return "non spin flip", layers, Aguide
    
def Yaohua_example():
    Aguide = 270
    rhoB = B2SLD * 0.4 * 1e6
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, rhoB, 90, 0.0],
        [ 200, 4.0, rhoB + 1.0, np.arctan2(rhoB, 1.0), 0.0],
        [ 200, 2.0, rhoB + 1.0, 90, 0.0],
        [ 0, 4.0, rhoB, 90 , 0.0],
        ]
    return "Yaohua example", layers, Aguide
    
def zf_Yaohua_example():
    Aguide = 270
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, 0.0, 90, 0.0],
        [ 200, 4.0, 1.0, 0.0001, 0.0],
        [ 200, 2.0, 1.0, 90, 0.0],
        [ 0, 4.0, 0.0, 90, 0.0],
        ]
    return "Yaohua example", layers, Aguide 

def demo():
    """run demo"""
    import pylab
    #compare(*simple())
    #compare(*twist())
    #compare(*magsub())
    #compare(*magsurf())
    pylab.figure()
    compare(*zf_Yaohua_example())
    pylab.figure()
    compare(*add_H(zf_Yaohua_example(), 0.4, 90, 0)) # 4000 Gauss
    pylab.figure()
    compare(*NSF_example())
    pylab.figure()
    compare(*add_H(NSF_example(), 1.0, 90, 0))
    pylab.show()

if __name__ == "__main__":
    demo()
