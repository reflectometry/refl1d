from refl1d.names import PolarizedNeutronProbe, NeutronProbe
import numpy as np
import os
from refl1d.probe.resolution import QT2L
from pathlib import Path


# TODO: See if we can wrap np.geomspace to give the same behaviour as below?
# Currently, np.geomspace does not allow for a step size to be defined,
# only the number of points.
# This is not ideal for TOF data where we want to define a step size in dQ/Q.
def logstep(start, stop, step, base=10.0):
    """
    Creates a log spaced 1d array by defining a step size and a base
    In the form of dQ/Q - i.e. dQ\\Q*Qpoint
    """
    logrange = [start]
    point = start
    while point < stop:
        point = point + base ** (np.log10(step * point) / np.log10(base))

        logrange.append(point)

    return np.array(logrange)


def TOF_loader(T=0.25, dQoQ=0.02, Q_sim_range=(0.005, 0.2), filename=None, name=None, skiprows=1, **kw):
    """
    Loads and creates NeutronProbe objects for TOF stitched datasets
    I.e. from multiple angles. In the case of ISIS NR instruments we
    typically have a constant dq/q resolution which the data is binned to at
    the end of the reduction.

    *T* incident theta for the lowest angle
    *dQoQ* dq/q resolution data has been binned to
    *filename* filename of the data set to be loaded
    *kw* keyword arguments (kwargs) to be passed to NeutronProbe()
    """

    # np.loadtext is currently set for simple 3 column POLREF data
    # if you aim to use this loader for other data, talk to your local contact
    # to understand the data format, and how best to load it.

    if filename is not None:
        data = np.loadtxt(filename, skiprows=skiprows).T
        if dQoQ is None:
            Q, R, dR, dQo = data
        else:
            Q, R, dR = data
            dQo = Q * dQoQ
        data_in = (R, dR)
    else:
        Q = logstep(Q_sim_range[0], Q_sim_range[1], dQoQ, base=dQoQ)
        data_in = None

    L = QT2L(Q, T)
    # Converting the dq/q resolution into a dq value for each Q point
    # dQ = FWHM2sigma(dQo)
    # Since we take dL/L = 0, dQ/Q = dT/T, so dT = T * dQoQ
    dT = T * dQoQ
    # print(f"dT = {dT}")

    probe_out = NeutronProbe(
        name=name,
        T=T,
        dT=dT,
        L=L,
        dL=0,
        data=data_in,
        # For standard TOF measurements resolution is assumed to be normal (gaussian)
        #  For measurements with many wavelengths and many angles (say cw measurements)
        #  then a uniform resolution can be used instead.
        resolution="normal",
        **kw,
    )

    return probe_out


def load_probe_polref(filename, angle, dQoQ, name=None, path=None, pol_mode=None, field=None, **kw):
    """
    creates one probe (Neutron, Polarized - PA or PNR) from one measurement - could be one angle or stitched dataset.
    If polarized, sets some default values and links instrumental parameters for each cross-section together
    """

    if name is None:
        name = filename
    if path is None:
        path = os.getcwd()

    filepath = Path(path) / filename

    # Load parts
    if pol_mode == "pa":  # Fully polarized: --, -+, +-, ++
        parts = ["_dd", "_du", "_ud", "_uu"]
    elif pol_mode == "pnr":  # Half_polarized: --, ++
        parts = ["_d", "_u"]
    else:  # Unpolarized
        parts = [""]
    data = [TOF_loader(T=angle, dQoQ=dQoQ, filename=f"{filepath}{part}.dat", name=name, **kw) for part in parts]

    # Unpolarized: return first and only part
    if len(parts) == 1:
        return data[0]

    # Polarized: sort parts into cross sections and return polarized probe
    if len(parts) == 2:
        cross_sections = [data[0], None, None, data[1]]
    else:
        cross_sections = data
    if field is None:
        field = 0.0
    probe = PolarizedNeutronProbe(cross_sections, Aguide=270, H=field, name=name)
    probe.shared_beam()

    return probe


# TODO: Add simulation wrapper based on the loader above for simulating POLREF data.
