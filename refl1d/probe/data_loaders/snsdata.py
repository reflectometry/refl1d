# This program is in the public domain
# Author: Paul Kienzle
"""
SNS data loaders

The following instruments are defined::

    Liquids, Magnetic

These are :class:`resolution.Pulsed` classes tuned with
default instrument parameters and loaders for reduced SNS data.
See :mod:`resolution` for details.
"""

import math
import re

import numpy as np
from bumps.data import parse_file

from ...probe.probe import make_probe
from .. import resolution
from ..instrument import Pulsed
from .rebin import rebin

## Estimated intensity vs. wavelength for liquids reflectometer
LIQUIDS_FEATHER = np.array(
    [
        (2.02555, 20.6369),
        (2.29927, 23.6943),
        (2.57299, 23.6943),
        (2.87409, 21.1146),
        (3.22993, 15.5732),
        (3.58577, 12.8981),
        (4.07847, 9.4586),
        (4.5438, 6.59236),
        (5.11861, 4.68153),
        (5.7208, 3.05732),
        (6.37774, 1.91083),
        (7.19891, 1.24204),
        (8.04745, 0.955414),
        (9.06022, 0.573248),
        (10.1825, 0.477707),
        (11.4142, 0.382166),
        (12.8102, 0.191083),
        (14.3431, 0.286624),
    ]
).T


def load(filename, instrument=None, **kw):
    """
    Return a probe for SNS data.
    """
    if instrument is None:
        instrument = Pulsed()
    header, data = parse_sns_file(filename)
    header.update(**kw)  # calling parameters override what's in the file.
    # print "\n".join(k+":"+str(v) for k, v in header.items())
    # Guess what kind of data we have
    if has_columns(header, ("Q", "dQ", "R", "dR", "L")):
        probe = QRL_to_data(instrument, header, data)
    elif has_columns(header, ("time_of_flight", "data", "Sigma")):
        probe = TOF_to_data(instrument, header, data)
    else:
        raise IOError("Unknown columns: " + ", ".join(header["columns"]))
    probe.title = header["title"]
    probe.date = header["date"]
    probe.instrument = header["instrument"]
    return probe


def has_columns(header, v):
    return len(header["columns"]) == len(v) and all(ci == si for ci, si in zip(header["columns"], v))


def QRL_to_data(instrument, header, data):
    """
    Convert data to T, L, R
    """
    Q, dQ, R, dR, L = data
    dL = resolution.binwidths(L)
    if "angle" in header and "slits_at_Tlo" in header:
        T = header.pop("angle", header.pop("T", None))
        probe = instrument.probe(L=L, dL=dL, T=T, data=(R, dR), **header)
    else:
        T = resolution.QL2T(Q[0], L[0])
        dT = resolution.dQdL2dT(Q[0], dQ[0], L[0], dL[0])
        probe = make_probe(T=T, dT=dT, L=L, dL=dL, data=(R, dR), **header)
    return probe


def TOF_to_data(instrument, header, data):
    """
    Convert TOF data to neutron probe.

    Wavelength is set from the average of the times at the edges of the
    bins, not the average of the wavelengths.  Wavelength resolution is
    set assuming the wavelength at the edges of the bins defines the
    full width at half maximum.

    The correct answer is to look at the wavelength distribution within
    the bin including effects of pulse width and intensity as a function
    wavelength and use that distribution, or a gaussian approximation
    thereof, when computing the resolution effects.
    """
    TOF, R, dR = data
    Ledge = resolution.TOF2L(instrument.d_moderator, TOF)
    L = resolution.TOF2L(instrument.d_moderator, (TOF[:-1] + TOF[1:]) / 2)
    dL = (Ledge[1:] - Ledge[:-1]) / 2.35  # FWHM is 2.35 sigma
    R = R[:-1]
    dR = dR[:-1]
    min_time, max_time = header.get("TOF_range", instrument.TOF_range)
    keep = np.isfinite(R) & np.isfinite(dR) & (TOF[:-1] >= min_time) & (TOF[1:] <= max_time)
    L, dL, R, dR = [v[keep] for v in (L, dL, R, dR)]
    T = np.array([header.get("angle", header.get("T", None))], "d")
    T, dT, L, dL = instrument.resolution(L=L, dL=dL, T=T, **header)
    probe = make_probe(T=T, dT=dT, L=L, dL=dL, data=(R, dR), **header)
    return probe


def parse_sns_file(filename):
    """
    Parse SNS reduced data, returning *header* and *data*.

    *header* dictionary of fields such as 'data', 'title', 'instrument'
    *data* 2D array of data
    """
    raw_header, data = parse_file(filename)
    header = {}

    # guess instrument from file name
    original_file = raw_header.get("F", "unknown")
    if "REF_L" in original_file:
        instrument = "Liquids"
    elif "REF_M" in original_file:
        instrument = "Magnetic"
    else:
        instrument = "unknown"
    header["instrument"] = instrument
    header["filename"] = original_file
    header["radiation"] = "neutron"

    # Plug in default instrument values for slits
    if "instrument" in header and header["instrument"] in INSTRUMENTS:
        instrument = INSTRUMENTS[header["instrument"]]
        header["d_s1"] = instrument.d_s1
        header["d_s2"] = instrument.d_s2

    # Date-time field for the file
    header["date"] = raw_header.get("D", "")

    # Column names and units
    columnpat = re.compile(r"(?P<name>\w+)\((?P<units>[^)]*)\)")
    columns, units = zip(*columnpat.findall(raw_header.get("L", "")))
    header["columns"] = columns
    header["units"] = units

    # extra information like title, angle, etc.
    commentpat = re.compile(r"(?P<name>.*)\s*:\s*(?P<value>.*)\s*\n")
    comments = dict(commentpat.findall(raw_header.get("C", "")))
    header["title"] = comments.get("Title", "")
    header["description"] = comments.get("Notes", "")

    # parse values of the form "Long Name: (value, 'units')" in comments
    valuepat = re.compile(r"[(]\s*(?P<value>.*)\s*, \s*'(?P<units>.*)'\s*[)]")

    def parse_value(valstr):
        d = valuepat.match(valstr).groupdict()
        return float(d["value"]), d["units"]

    if "Detector Angle" in comments:
        header["angle"], _ = parse_value(comments["Detector Angle"])

    return header, data


def write_file(filename, probe, original=None, date=None, title=None, notes=None, run=None, charge=None):
    """
    Save probe as SNS reduced file.
    """
    ## Example header
    # F /SNSlocal/REF_L/2007_1_4B_SCI/2895/NeXus/REF_L_2895.nxs
    # E 1174593434.7
    # D 2007-03-22 15:57:14
    # C Run Number: 2895
    # C Title: MK NR4 dry 032007_No2Rep0
    # C Notes: MK NR 4 DU 53 dry from air
    # C Detector Angle: (0.0, 'degree')
    # C Proton Charge: 45.3205833435

    # S 1 Spectrum ID ('bank1', (85, 151))
    # N 3
    # L time_of_flight(microsecond) data() Sigma()
    from datetime import datetime as dt

    parts = []
    if original is None:
        original = filename
    if date is None:
        date = dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")
    parts.append("#F " + original)
    parts.append("#D " + date)
    if run is not None:
        parts.append("#C Run Number: %s" % run)
    if title is not None:
        parts.append("#C Title: %s" % title)
    if notes is not None:
        parts.append("#C Notes: %s" % notes)
    parts.append("#C Detector Angle: (%g, 'degree')" % probe.T[0])
    if charge is not None:
        parts.append("#C Proton Charge: %s" % charge)
    parts.append("")
    parts.append("#N 5")
    parts.append("#L Q(1/A) dQ(1/A) R() dR() L(A)")
    parts.append("")
    header = "\n".join(parts)
    probe.write_data(filename, columns=["Q", "dQ", "R", "dR", "L"], header=header)


class SNSData(object):
    def load(self, filename, **kw):
        return load(filename, instrument=self, **kw)


# TODO: print "Insert correct slit distances for Liquids and Magnetic"
class Liquids(SNSData, Pulsed):
    """
    Loader for reduced data from the SNS Liquids instrument.
    """

    instrument = "Liquids"
    radiation = "neutron"
    feather = LIQUIDS_FEATHER
    wavelength = 2.0, 15.0
    # wavelength = 0.5, 5
    # wavelength = 5.5, 10
    # wavelength = 10.5, 15
    dLoL = 0.02
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0
    d_moderator = 14.850  # moderator to detector distance
    TOF_range = (6000, 60000)


class Magnetic(SNSData, Pulsed):
    """
    Loader for reduced data from the SNS Magnetic instrument.
    """

    instrument = "Magnetic"
    radiation = "neutron"
    wavelength = 1.8, 14
    dLoL = 0.02
    d_s1 = 75 * 2.54
    d_s2 = 14 * 2.54


# Instrument names assigned by reflpak
INSTRUMENTS = {
    "Liquids": Liquids,
    "Magnetic": Magnetic,
}

# ===== utils ==============


def intensity_from_spline(Lrange, dLoL, feather):
    L0, L1 = Lrange
    n = math.ceil(math.log(L1 / L0) / math.log(1 + dLoL))
    L = L0 * (1 + dLoL) ** np.arange(0, n)
    return (L[:-1] + L[1:]) / 2, rebin(feather[0], feather[1], L)


def boltzmann_feather(L, counts=100000, range=None):
    """
    Return expected intensity as a function of wavelength given the TOF
    feather range and the total number of counts.

    TOF feather is approximately a boltzmann distribution with gaussian
    convolution.  The following looks pretty enough; don't know how well it
    corresponds to the actual SNS feather.
    """
    import scipy.stats

    y = np.linspace(-4, 4, 10)
    G = np.exp(-(y**2) / 10)
    x = np.arange(12, 85)
    B = scipy.stats.boltzmann.pmf(x, 0.05, counts, loc=16)
    BGz = np.convolve(B, G, mode="same")
    # if range is None: range = L[0], L[-1]
    # if range[0] > range[1]: range = range[::-1]
    # range = range[0]*(1-1e-15), range[1]*(1+1e-15)
    # z = np.linspace(range[0], range[1], len(BGz))
    z = np.linspace(2, 16.5, len(BGz))  # Wavelength range for liquids
    pL = np.interp(L, z, BGz, left=0, right=0)
    nL = pL / sum(pL) * counts
    return nL
