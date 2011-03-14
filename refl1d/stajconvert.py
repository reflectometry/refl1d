# This program is in the public domain
# Author: Paul Kienzle
"""
Convert staj files to Refl1D models
"""

import numpy
from numpy import tan, cos, sqrt, radians, degrees, pi
from .staj import MlayerModel
from .model import Slab, Stack, Repeat
from .material import SLD
from .resolution import QL2T,sigma2FWHM
from .probe import NeutronProbe, XrayProbe

def load_mlayer(filename, fit_pmp=0):
    """
    Load a staj file as a model.
    """
    staj = MlayerModel.load(filename)
    model = mlayer_to_model(staj)
    if fit_pmp != 0:
        fit_all(model, pmp=fit_pmp)
    return model

def save_mlayer(experiment, filename):
    """
    Save a model to a staj file.
    """
    staj = model_to_mlayer(experiment)
    #print staj
    staj.save(filename)

def fit_all(M, pmp=20):
    """
    Set all non-zero parameters to fitted parameters inside the model.
    """

    # Exclude unlikely fitting parameters
    exclude = set((M.sample[0].thickness,
               M.sample[-1].thickness,
               M.sample[-1].interface,
               M.probe.back_absorption,
               ))
    if M.probe.intensity.value == 1:
        exclude.add(M.probe.intensity)
    if M.probe.background.value < 2e-10:
        exclude.add(M.probe.background.value)

    # Fit everything else using a range of +/- pmp %
    for p in parameter.unique(M.parameters()):
        if p in exclude: continue
        if p.value != 0: p.pmp(pmp)
        #p.fixed = False

def mlayer_to_model(staj):
    """
    Convert a loaded staj file to a refl1d experiment.

    Returns a new experiment
    """
    from .experiment import Experiment
    sample = _mlayer_to_stack(staj)
    probe = _mlayer_to_probe(staj)
    return Experiment(sample=sample,probe=probe)

def _mlayer_to_stack(s):
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

def _mlayer_to_probe(s):
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
    dQ = s.FWHMresolution(Q)
    L,dL = s.wavelength,s.wavelength_dispersion
    T = QL2T(Q=Q,L=s.wavelength)
    dT = sqrt( (dQ*L/(4*pi*cos(radians(T))))**2 - (tan(radians(T))*dL/L)**2 )
    dT = degrees(dT)

    # Hack: X-ray is 1.54; anything above 2 is neutron.  Doesn't matter since
    # materials are set as SLD rather than composition.
    if s.wavelength < 2:
        probe = XrayProbe
    else:
        probe = NeutronProbe
    probe = probe(T=T,dT=dT,L=L,dL=dL,data=(R,dR),
                  theta_offset=s.theta_offset,
                  background=s.background,
                  intensity=s.intensity)
    return probe


def model_to_mlayer(model):
    """
    Return an mlayer model based on the a slab stack.

    Raises TypeError if model cannot be stored as a staj file.
    """
    #TODO: when back reflectivity is handled properly, need to support it here
    stack = model.sample
    probe = model.probe
    staj = MlayerModel(roughness_steps=51)

    # Set up beam info
    if (probe.L != probe.L[0]).any():
        # Reason is that mlayer uses mu/(2 lambda) rather than irho
        raise TypeError("Mlayer only supports monochromatic sources")

    staj.set(wavelength=probe.L[0],
             intensity=probe.intensity.value,
             background=probe.background.value,
             theta_offset=probe.theta_offset.value)
    if hasattr(probe, 'filename'):
        staj.data_file = probe.filename
    else:
        staj.Qmin, staj.Qmax = min(probe.Q), max(probe.Q)
        staj.num_Q = len(probe.Q)
    staj.fit_FWHMresolution(probe.Q, sigma2FWHM(probe.dQ))

    # Interpret slabs and repeats
    sections = []
    section = []
    repeats = 0
    for l in stack:
        if isinstance(l, Slab):
            section.append(l)
        elif isinstance(l, Repeat):
            sections.append(section)
            sections.append(l[:])
            repeats = l.repeats
            section = []
        else:
            raise TypeError("Only slabs supported")
    sections.append(section)
    if len(sections) > 3:
        raise TypeError("Only one repeated section supported")
    if len(sections) == 3:
        for l in sections[1]:
            if not isinstance(l, Slab):
                raise TypeError("Only slabs supported in repeat section")
        num_top = len(sections[2]) - 1
        num_middle = len(sections[1])
        num_bottom = len(sections[0])
        if num_top > 9 or num_middle > 9 or num_bottom > 9:
            raise TypeError("Maximum section length of 9")
        if num_top < 1 or num_middle < 1 or num_bottom < 1:
            raise TypeError("Need at least one slab per section, plus vacuum")
        model.num_top = num_top
        model.num_middle = num_middle
        model.num_bottom = num_bottom
        model.num_repeats = repeats
        slabs = []
        slabs.extend(reversed(sections[2]))
        slabs.extend(reversed(sections[1]))
        slabs.extend(reversed(sections[0]))
    else:
        # must be only one section
        slabs = list(reversed(sections[0]))
        if len(slabs) > 28:
            raise TypeError("Too many slabs (only 28 slabs allowed)")

    # Convert slabs to sld parameters
    #print "\n".join(str(s) for s in slabs)
    values = []
    for layer in slabs:
        rho,irho = layer.material.sld(probe)
        thickness = layer.thickness.value
        roughness = layer.interface.value
        values.append((rho,irho,thickness,roughness))
    vectors = [numpy.array(v) for v in zip(*values)]
    staj.sigma_roughness = numpy.array([1,2,3])
    staj.rho,staj.irho,staj.thickness,staj.sigma_roughness = vectors

    # If no repeats, split the model into sections
    if len(sections) == 1:
        staj.split_sections()

    return staj
