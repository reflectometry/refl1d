import numpy
from numpy import tan, cos, sqrt, radians, pi
from .experiment import Experiment
from .staj import MlayerModel, MlayerMagnetic
from .model import Slab, Stack
from .material import SLD
from .util import QL2T
from .probe import NeutronProbe, XrayProbe

def load_mlayer(filename):
    s = MlayerModel.load(filename)
    sample = mlayer_to_stack(s)
    probe = mlayer_probe(s)
    return Experiment(sample=sample,probe=probe)

def save_mlayer(filename, model):
    to_mlayer(model).save(filename)

def load_gj2(filename):
    return from_gj2(MlayerMagnetic.load(filename))

def save_gj2(filename, model):
    to_gj2(model).save(filename)

def mlayer_to_stack(s):
    """
    Return a sample stack based on the model used in the staj file.
    """
    i1 = s.num_top
    i2 = s.num_top+s.num_middle

    # Construct slabs
    slabs = []
    for i in reversed(range(len(s.rho))):
        if i == 0:
            name = 'V'
        elif i < i1:
            name = 'T%d'%(i+1)
        elif i < i2:
            name = 'M%d'%(i-i1+1)
        else:
            name = 'B%d'%(i-i2+1)
        slabs.append(Slab(material=SLD(rho=s.rho[i],irho=s.irho[i],name=name),
                          thickness=s.thickness[i], 
                          interface=s.sigma_roughness[i]))
        
    # Build stack
    if s.num_repeats == 0:
        stack = Stack(slabs[:i1]+slabs[i2:])
    elif s.num_repeats == 1:
        stack = Stack(slabs)
    else:
        stack = Stack(slabs[:i1]+[Repeat(Stack(slabs[i1:i2]))]+slabs[i2:])

    return stack

def mlayer_probe(s):
    """
    Return a model probe based on the data used for the staj file.
    """
    if s.data_file == "":
        Q = numpy.linspace(s.Qmin, s.Qmax, s.num_Q)
        R,dR = None,None
    else:
        Q,R,dR = numpy.loadtxt(s.data_file).T
    # (dQ/Q)^2 = (dL/L)^2 + (dT/tan(T))^2
    # Transform so that Q=0 is not an error
    # =>  dT = tan(T)*sqrt((dQ/Q)^2 - (dL/L)^2))
    #        = sqrt( (dQ*L/4*pi*sin(T))^2*(sin(T)/cos(T))^2 - (tan(T)*dL/L)^2 )
    #        = sqrt( (dQ*L/4*pi*cos(T))^2 - (tan(T)*dL/L)^2 )
    dQ = s.resolution(Q)
    L,dL = s.wavelength,s.wavelength_dispersion
    T = QL2T(Q=Q,L=s.wavelength)
    dT = sqrt( (dQ*L/(4*pi*cos(radians(T))))**2 - (tan(radians(T))*dL/L)**2 )
    
    # Hack: X-ray is 1.54; anything above 2 is neutron.  Doesn't matter since
    # materials are set as SLD rather than composition.
    if s.wavelength < 2:
        probe = XrayProbe
    else:
        probe = NeutronProbe
    return probe(T=T,dT=dT,L=L,dL=dL,data=(R,dR),
                 theta_offset=s.theta_offset,
                 background=s.background,
                 intensity=s.intensity)
