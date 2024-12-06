# This program is in the public domain
# Author: Paul Kienzle
"""
NCNR data loaders

The following instruments are defined:

    MAGIK, PBR, ANDR, NG1, NG7 and XRay

These are :class:`refl1d.probe.instrument.Monochromatic` classes tuned with default
instrument parameters and loaders for reduced NCNR data.

The instruments can be used to load data or to compute resolution functions
for the purposes.

Example loading data:

    >>> import numpy as np
    >>> import pylab
    >>> from refl1d.names import Experiment, NCNR, air, gold, permalloy, sample_data, silicon
    >>> datafile = sample_data('chale207.refl')
    >>> instrument = NCNR.ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)
    >>> probe = instrument.load(datafile)
    >>> probe.plot(view='log')

Magnetic data has multiple cross sections and often has fixed slits:

    >>> datafile = sample_data('lha03_255G.refl')
    >>> instrument = NCNR.NG1(slits_at_Tlo=1)
    >>> probe = instrument.load_magnetic(datafile)
    >>> probe.plot(view='SA', substrate=silicon) # Spin asymmetry view

For simulation, you need a probe and a sample:

    >>> instrument = NCNR.ANDR(Tlo=0.5, slits_at_Tlo=0.2, slits_below=0.1)
    >>> probe = instrument.probe(T=np.linspace(0, 5, 51))
    >>> probe.plot_resolution()
    >>> sample = silicon(0, 10) | gold(100, 10) | air
    >>> M = Experiment(probe=probe, sample=sample)
    >>> M.simulate_data() # Optional
    >>> M.plot()

And for magnetic:

    >>> instrument = NCNR.NG1(slits_at_Tlo=1)
    >>> #sample = silicon(0, 10) | Magnetic(permalloy(100, 10), rho_M=3) | air
    >>> #M = Experiment(probe=probe, sample=sample)
    >>> #M.simulate_data()
    >>> #M.plot()
    >>> #probe = instrument.simulate_magnetic(sample, T=np.linspace(0, 5, 51))
    >>> #h = pylab.plot(probe.Q, probe.dQ)
    >>> #h = pylab.ylabel('resolution (1-sigma)')
    >>> #h = pylab.xlabel('Q (inv A)')

See :mod:`instrument <refl1d.probe.instrument>` for details.
"""

import os

from bumps.data import parse_file

from ...sample.reflectivity import BASE_GUIDE_ANGLE
from ..instrument import Monochromatic
from ..probe import PolarizedNeutronProbe


def load(filename, instrument=None, **kw):
    """
    Return a probe for NCNR data.

    Keyword arguments are as specified Monochromatic instruments.
    """
    if filename is None:
        return None
    if instrument is None:
        instrument = Monochromatic()
    header, data = parse_ncnr_file(filename)
    # calling parameters override what's in the file.
    header.update(filename=filename, **kw)
    Q, R, dR = data
    header.pop("Q", None)  # if columns are preceded by # Q R dR
    probe = instrument.probe(Q=Q, data=(R, dR), **header)
    probe.title = header["title"] if "title" in header else filename
    probe.date = header["date"] if "date" in header else "unknown"
    probe.instrument = header["instrument"] if "instrument" in header else instrument.instrument
    return probe


def load_magnetic(filename, Aguide=BASE_GUIDE_ANGLE, H=0, shared_beam=True, **kw):
    """
    Return a probe for magnetic NCNR data.

    *filename* (string, or 4x string)
        If it is a string, then filenameA, filenameB, filenameC, filenameD,
        are the --, -+, +-, ++ cross sections, otherwise the individual
        cross sections should the be the file name for the cross section or
        None if the cross section does not exist.
    *Aguide* (degrees)
        Angle of the guide field relative to the beam.  270 is the default.
    *shared_beam* (True)
        Use false if beam parameters should be fit separately for the
        individual cross sections.

    Other keyword arguments are for the individual cross section loaders
    as specified in
    :class:`instrument.Monochromatic <refl1d.probe.instrument.Monochromatic>`.

    The data sets should are the base filename with an additional character
    corresponding to the spin state::

        'a' corresponds to spin --
        'b' corresponds to spin -+
        'c' corresponds to spin +-
        'd' corresponds to spin ++

    Unfortunately the interpretation is a little more complicated than
    this as the data acquisition system assigns letter on the basis of
    flipper state rather than neutron spin state.  Whether flipper on
    or off corresponds to spin up or down depends on whether the
    polarizer/analyzer is a supermirror in transmission or reflection
    mode, or in the case of ^3He polarizers, whether the polarization
    is up or down.

    For full control, specify filename as a list of files, with None
    for the missing cross sections.
    """
    probes = [load(v, **kw) for v in find_xsec(filename)]
    if all(p is None for p in probes):
        raise IOError("Data set has no magnetic cross sections: %r" % filename)
    probe = PolarizedNeutronProbe(probes, Aguide=Aguide, H=H)
    if shared_beam:
        probe.shared_beam()  # Share the beam parameters by default
    return probe


def find_xsec(filename):
    """
    Find files containing the polarization cross-sections.

    Returns tuple with file names for ++ +- -+ -- cross sections, or
    None if the spin cross section does not exist.
    """
    # Check if it is a string.  If not, assume it is a length 4 tuple
    try:
        filename + "s"
    except Exception:
        return filename

    if filename[-1] in "abcdABCD":
        filename = filename[:-1]

    def check(a):
        if os.path.exists(filename + a):
            return filename + a
        elif os.path.exists(filename + a.lower()):
            return filename + a.lower()
        else:
            return None

    return check("A"), check("B"), check("C"), check("D")


def parse_ncnr_file(filename):
    """
    Parse NCNR reduced data file returning *header* and *data*.

    *header* dictionary of fields such as 'data', 'title', 'instrument'
    *data* 2D array of data

    If 'columns' is present in header, it will be a list of the names of
    the columns.  If 'instrument' is present in the header, the default
    instrument geometry will be specified.

    Slit geometry is set to the default from the instrument if it is not
    available in the reduced file.
    """
    header, data = parse_file(filename)

    # Fill in instrument parameters, if not available from the file
    if "instrument" in header and header["instrument"] in INSTRUMENTS:
        instrument = INSTRUMENTS[header["instrument"]]
        header.setdefault("radiation", instrument.radiation)
        header.setdefault("wavelength", str(instrument.wavelength))
        header.setdefault("dLoL", str(instrument.dLoL))
        header.setdefault("d_s1", str(instrument.d_s1))
        header.setdefault("d_s2", str(instrument.d_s2))

    if "columns" in header:
        header["columns"] = header["columns"].split()
    for key in ("wavelength", "dLoL", "d_s1", "d_s2"):
        if key in header:
            header[key] = float(header[key])

    return header, data


class NCNRData(object):
    def readfile(self, filename):
        return parse_ncnr_file(filename)

    def load(self, filename, **kw):
        return load(filename, instrument=self, **kw)

    def load_magnetic(self, filename, **kw):
        return load_magnetic(filename, instrument=self, **kw)


class MAGIK(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR MAGIK diffractometer/reflectometer.
    """

    instrument = "MAGIK"
    radiation = "neutron"
    wavelength = 5.0042
    dLoL = 0.009
    d_s1 = 1321.0 + 438.0
    d_s2 = 1321.0 - 991.0


class PBR(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR PBR reflectometer.
    """

    instrument = "PBR"
    radiation = "neutron"
    wavelength = 4.75
    dLoL = 0.015
    d_s1 = 1835
    d_s2 = 343
    d_s3 = 380
    d_s4 = 1015


class NG7(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR NG-7 reflectometer.
    """

    instrument = "NG-7"
    radiation = "neutron"
    wavelength = 4.768
    dLoL = 0.025  # 2.5% FWHM wavelength spread
    d_s2 = 222.25
    d_s1 = 1722.25
    d_detector = 2000.0


class XRay(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR X-ray reflectometer.

    Normal dT is in the range 2e-5 to 3e-4.

    Slits are fixed throughout the experiment in one of a
    few preconfigured openings.  Please update this file with
    the standard configurations when you find them.

    You can choose to ignore the geometric calculation entirely
    by setting the slit opening to 0 and using sample_broadening
    to define the entire divergence.  Note that Probe.sample_broadening
    is a fittable parameter, so you need to access its value::

        >>> from refl1d.names import sample_data, NCNR
        >>> file = sample_data("spin_valve01.refl")
        >>> xray = NCNR.XRay(slits_at_Tlo=0)
        >>> data = xray.load(file, sample_broadening=1e-4)
        >>> print(data.sample_broadening.value)
        0.0001
    """

    instrument = "X-ray"
    radiation = "xray"
    wavelength = 1.5416
    dLoL = 1e-3 / wavelength
    d_s1 = 275.5
    d_s2 = 192.5
    d_s3 = 175.0
    d_detector = None


class ANDR(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR AND/R diffractometer/reflectometer.
    """

    instrument = "AND/R"
    radiation = "neutron"
    wavelength = 5.0042
    dLoL = 0.009
    d_s1 = 230.0 + 1856.0
    d_s2 = 230.0


class NG1(NCNRData, Monochromatic):
    """
    Instrument definition for NCNR NG-1 reflectometer.
    """

    instrument = "NG-1"
    radiation = "neutron"
    wavelength = 4.75
    dLoL = 0.015
    d_s1 = 75 * 25.4
    d_s2 = 14 * 25.4
    d_s3 = 9 * 25.4
    d_s4 = 42 * 25.4


# Instrument names assigned by reflpak
INSTRUMENTS = {
    "CG-D": MAGIK,
    "NG-D": PBR,
    "CG-1": ANDR,
    "NG-1": NG1,
    "NG-7": NG7,
    "Xray": XRay,
}

_ = r'''
def _counting_time(instrument, sample, uncertainty,
    Qrange, Qstep, beam_rate, num_parts):
    r"""
    Simulate counting time for a particular sample.

    :Parameters:
        *sample* : Stack
            Model of the sample.
        *uncertainty* = 0.01 : float
            Relative uncertainty in the measurement.

    Additional :meth:`probe` keyword parameters are required to define
    the set of angles to be measured

    Returns
    -------

    *experiment* : Experiment
        Sample + probe with simulated data.

    Algorithm
    ---------

    Assuming our counts follow approximately the Fresnel reflectivity
    of the sample, $F$, and we are targeting an fractional uncertainty
    $\Delta R/R = \sigma$, we can calculate the desired incident beam
    $I = 1/(F\sigma^2)$ that will yield this uncertainty.  With $I$,
    we can compute the  expected number of counts on the detector due
    to reflection off the sample (this is just $R_{\rm th} I$) and use
    that to simulate detector counts $D$ by drawing from a Poisson
    distribution $D ~ P(R_{\rm th} I)$.  Given $D$ and $I$ we can use
    the normal reflectometry reduction process to get $(R, \Delta R)$
    as:

    .. math:

        I &=& 1/(F \sigma^2) //
        D &~& P(R_{\rm th} I) //
        R &=& D/I //
        \Delta R &=& \sqrt(D)/I
    """
    import numpy as np
    from refl1d.names.experiment import Experiment
    probe = self.probe(**kw)
    M = Experiment(probe=probe, sample=sample)
    if 1: # Fresnel counting
        I = 1/(M.fresnel()* uncertainty**2)
    else: # Q^4 counting
        I = 1/(100*probe.Q**4 * uncertainty**2)
    D = np.random.poisson( Rth * I )
    R, dR = D/I, np.sqrt(D)/I
    probe.data = R, dR

    return M
'''
