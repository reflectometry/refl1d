# Author: Paul Kienzle
"""
Convert staj files to refl1d
"""

import os

import numpy as np
from bumps import parameter
from numpy import cos, degrees, pi, radians, sqrt, tan

from ...probe.resolution import QL2T, sigma2FWHM
from ...sample.layers import Repeat, Slab, Stack
from ...sample.magnetism import Magnetism
from ...sample.material import SLD
from ..probe import NeutronProbe, PolarizedNeutronProbe, XrayProbe
from .staj import MlayerMagnetic, MlayerModel


def load_mlayer(filename, fit_pmp=0, name=None, layers=None):
    """
    Load a staj file as a model.
    """
    if filename.endswith(".staj"):
        staj = MlayerModel.load(filename)
        model = mlayer_to_model(staj, name=name, layers=layers)
    else:
        sta = MlayerMagnetic.load(filename)
        model = mlayer_magnetic_to_model(sta, name=name, layers=layers)
    if fit_pmp != 0:
        fit_all(model, pmp=fit_pmp)
    return model


def save_mlayer(experiment, filename, datafile=None):
    """
    Save a model to a staj file.
    """
    staj = model_to_mlayer(experiment, datafile)
    # print staj
    staj.save(filename)


def fit_all(M, pmp=20):
    """
    Set all non-zero parameters to fitted parameters inside the model.
    """

    # Exclude unlikely fitting parameters
    exclude = set(
        (
            M.sample[0].thickness,
            M.sample[-1].thickness,
            M.sample[-1].interface,
            M.probe.back_absorption,
        )
    )
    if M.probe.intensity.value == 1:
        exclude.add(M.probe.intensity)
    if M.probe.background.value < 2e-10:
        exclude.add(M.probe.background.value)

    # Fit everything else using a range of +/- pmp %
    for p in parameter.unique(M.parameters()):
        if p in exclude:
            continue
        if p.value != 0:
            p.pmp(pmp)
        # p.fixed = False


def mlayer_to_model(staj, name=None, layers=None):
    """
    Convert a loaded staj file to a refl1d experiment.

    Returns a new experiment
    """
    from ...experiment import Experiment

    sample = _mlayer_to_stack(staj, name, layers)
    probe = _load_probe(staj, name, xs="")
    return Experiment(sample=sample, probe=probe)


def _mlayer_to_stack(s, name, layers):
    """
    Return a sample stack based on the model used in the staj file.
    """
    # check pars
    if layers and len(layers) != len(s.rho):
        raise ValueError("Have %d staj layers but got %d layer names" % (len(s.rho), len(layers)))

    # prepend datafile name to layers
    if name is None:
        name = os.path.splitext(s.data_file)[0]
    if name and not name.endswith(" "):
        name += " "

    i1 = s.num_top + 1
    i2 = s.num_top + s.num_middle + 1

    # Construct slabs
    slabs = []
    for i in reversed(range(len(s.rho))):
        if layers:
            Lname = layers[len(layers) - i - 1]
        elif i == 0:
            Lname = "V"
        elif i < i1:
            Lname = "T%d" % (i)
        elif i < i2:
            Lname = "M%d" % (i - i1 + 1)
        else:
            Lname = "B%d" % (i - i2 + 1)
        slabs.append(
            Slab(
                material=SLD(rho=s.rho[i], irho=s.irho[i], name=name + Lname),
                thickness=s.thickness[i],
                interface=s.sigma_roughness[i],
            )
        )

    # Build stack
    if s.num_repeats == 0:
        stack = Stack(slabs[:i1] + slabs[i2:])
    elif s.num_repeats == 1:
        stack = Stack(slabs)
    else:
        stack = Stack(slabs[:i1] + [Repeat(Stack(slabs[i1:i2]))] + slabs[i2:])

    return stack


XS = {"A": "--", "B": "-+", "C": "+-", "D": "++", "": ""}


def _load_probe(s, name, xs):
    if name is None:
        name = os.path.splitext(s.data_file)[0]

    if s.data_file == "":
        filename = "simulated"
        Q = np.linspace(s.Qmin, s.Qmax, s.num_Q)
        R, dR = None, None
    else:
        filename = s.data_file
        Q, R, dR = np.loadtxt(s.data_file + xs).T

    # Use Q and wavelength L from the staj file to determine angle T
    L = s.wavelength
    T = QL2T(Q=Q, L=s.wavelength)

    # Refl1D uses the following for dQ:
    #
    #     (dQ/Q)^2 = (dL/L)^2 + (dT/tan(T))^2
    #
    # Given the mlayer resolution dQ, and wavelength divergence dL, we can solve the
    # above for dT such that dQ in refl1d matches dQ in mlayer.
    #
    #      dT = sqrt( tan(T)^2 * ( (dQ/Q)^2 - (dL/L)^2) ) )
    #         = sqrt( (dQ*L/4*pi*sin(T))^2*(sin(T)/cos(T))^2 - (tan(T)*dL/L)^2 )
    #         = sqrt( (dQ*L/4*pi*cos(T))^2 - (tan(T)*dL/L)^2 )
    dQ = s.FWHMresolution(Q)
    dL = s.wavelength_dispersion
    dT = degrees(sqrt((dQ * L / (4 * pi * cos(radians(T)))) ** 2 - (tan(radians(T)) * dL / L) ** 2))

    # Hack: X-ray is 1.54; anything above 2 is neutron.  Doesn't matter since
    # materials are set as SLD rather than composition.
    if s.wavelength < 2:
        probe = XrayProbe
    else:
        probe = NeutronProbe
    probe = probe(
        T=T,
        dT=dT,
        L=L,
        dL=dL,
        data=(R, dR),
        theta_offset=getattr(s, "theta_offset", 0),  # gj2 has no theta offset
        background=s.background,
        intensity=s.intensity,
        name=name,
    )
    probe.filename = filename
    # probe.oversample(n=10)
    return probe


def model_to_mlayer(model, datafile):
    """
    Return an mlayer model based on the a slab stack.

    Raises TypeError if model cannot be stored as a staj file.
    """
    stack = model.sample
    probe = model.probe

    staj = MlayerModel(roughness_steps=51)

    # Set up beam info
    if (probe.L != probe.L[0]).any():
        # Reason is that mlayer uses mu/(2 lambda) rather than irho
        raise TypeError("Mlayer only supports monochromatic sources")

    staj.set(
        wavelength=probe.L[0],
        intensity=probe.intensity.value,
        background=probe.background.value,
        theta_offset=probe.theta_offset.value,
    )
    if datafile:
        staj.data_file = os.path.basename(datafile)
    else:
        staj.Qmin, staj.Qmax = min(probe.Qo), max(probe.Qo)
        staj.num_Q = len(probe.Qo)
    staj.fit_FWHMresolution(probe.Qo, sigma2FWHM(probe.dQ))

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
            repeats = l.repeat
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
        staj.num_top = num_top
        staj.num_middle = num_middle
        staj.num_bottom = num_bottom
        staj.num_repeats = repeats
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
    # print "\n".join(str(s) for s in slabs)
    values = []
    for layer in slabs:
        rho, irho = layer.material.sld(probe)
        thickness = layer.thickness.value
        roughness = layer.interface.value
        values.append((rho, irho, thickness, roughness))
    vectors = [np.array(v) for v in zip(*values)]

    # If back reflectivity, reverse the layers and move the interfaces
    # from the top of the layer to the bottom.
    if probe.back_reflectivity:
        vectors = [v[::-1] for v in vectors]
        vectors[3] = np.roll(vectors[3], 1)
        staj.num_top, staj.num_bottom = staj.num_bottom, staj.num_top

    staj.rho, staj.irho, staj.thickness, staj.sigma_roughness = vectors

    # If no repeats, split the model into sections
    if len(sections) == 1:
        # If the stack is too short, add layers at the top until it is
        # tall enough.  These layers should have the same SLD as the
        # old top layer, and a thickness large enough to accommodate
        # the interface between the top layer and the second layer.
        while len(staj.rho) < 4:
            staj.rho = np.hstack((staj.rho[0], staj.rho))
            staj.irho = np.hstack((staj.irho[0], staj.irho))
            staj.thickness = np.hstack((0, 3.5 * staj.sigma_roughness[1], staj.thickness[1:]))
            staj.sigma_roughness = np.hstack((0, staj.sigma_roughness))

        staj.split_sections()

    return staj


def mlayer_magnetic_to_model(sta, name=None, layers=None):
    """
    Convert a loaded sta file to a refl1d experiment.

    Returns a new experiment
    """
    from ...experiment import Experiment

    sample = _mlayer_magnetic_to_stack(sta, name, layers)
    probe = _mlayer_magnetic_to_probe(sta, name)
    return Experiment(sample=sample, probe=probe, dz=0.1)


def _mlayer_magnetic_to_stack(s, name, layers):
    """
    Return a sample stack based on the model used in the sta file.
    """
    # check pars
    if layers and len(layers) != len(s.rho):
        raise ValueError("Have %d sta layers but got %d layer names" % (len(s.rho), len(layers)))

    # prepend datafile name to layers
    if name is None:
        name = os.path.splitext(s.data_file)[0]
    if name and not name.endswith(" "):
        name += " "

    # Construct slabs
    magnetic_offset = np.cumsum(s.thickness - s.mthickness)
    slabs = []
    nlayers = len(s.rho)
    for i in range(nlayers - 1, -1, -1):
        if layers:
            Lname = layers[len(layers) - i - 1]
        elif i == 0:
            Lname = "V"
        else:
            Lname = "M%d" % i
        slab_i = Slab(
            material=SLD(rho=s.rho[i], irho=s.irho[i], name=name + Lname),
            thickness=s.thickness[i],
            interface=s.sigma_roughness[i],
        )
        if s.mrho[i] != 0.0:
            slab_i.magnetism = Magnetism(
                s.mrho[i],
                s.mtheta[i],
                interface_below=s.sigma_mroughness[i + 1] if i < nlayers - 1 else 0,
                interface_above=s.sigma_mroughness[i],
                dead_below=magnetic_offset[i + 1] if i < nlayers - 1 else 0,
                dead_above=magnetic_offset[i],
            )
        slabs.append(slab_i)

    return Stack(slabs)


def _mlayer_magnetic_to_probe(s, name):
    """
    Return a model probe based on the data used for the staj file.
    """
    if name is None:
        name = os.path.splitext(s.data_file)[0]

    active_xsec = s.active_xsec.upper()
    xs = [_load_probe(s, name, xs) if (xs in active_xsec) else None for xs in "ABCD"]
    probe = PolarizedNeutronProbe(xs, Aguide=s.guide_angle)
    # probe.oversample(n=6)
    return probe
