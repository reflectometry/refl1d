# standard imports
import json

# third party imports
from bumps.data import parse_multi, strip_quotes
import numpy as np
from orsopy.fileio.orso import load_nexus, load_orso

# refl1d imports
from refl1d.probe import (
    NeutronProbe,
    PolarizedNeutronProbe,
    PolarizedQProbe,
    Probe,
    QProbe,
    XrayProbe,
)
from refl1d.probe.resolution import QL2T, QT2L, FWHM2sigma, dQdL2dT, dQdT2dLoL, sigma2FWHM
from refl1d.sample.reflectivity import BASE_GUIDE_ANGLE


def parse_orso(filename):
    """
    Load an ORSO text (.ort) or binary (.orb) file containing one or more datasets

    Parameters
    ----------
    filename : str
        The path to the ORSO file to be loaded.

    Returns
    -------
    list of tuple
        A list of tuples, each containing a header dictionary and a data array derived from each loaded dataset.
        The header dictionary contains metadata about the measurement,
        and the data array contains the measurement data.

    Notes
    -----
    The function supports both ORSO text (.ort) and binary (.orb) files.
    The polarization information is converted using a predefined mapping.
    The header dictionary includes keys for polarization, angle, angular resolution,
    wavelength, and wavelength resolution.
    """
    if filename.endswith(".ort"):
        entries = load_orso(filename)
    elif filename.endswith(".orb"):
        entries = load_nexus(filename)

    POL_CONVERSION = {
        "po": "++",
        "mo": "--",
        "mm": "--",
        "mp": "-+",
        "pm": "+-",
        "pp": "++",
    }

    entries_out = []
    for entry in entries:
        header = entry.info
        data = entry.data
        settings = header.data_source.measurement.instrument_settings
        columns = header.columns
        polarization = POL_CONVERSION.get(settings.polarization, "unpolarized")
        header_out = {"polarization": polarization}

        def get_key(orso_name, refl1d_name, refl1d_resolution_name):
            """
            Extract value and error from one of the ORSO columns. If no column corresponding
            to entry `orso_name` is found, search in the instrument settings.

            Parameters
            ----------
            orso_name : str
                The name of the ORSO column or instrument setting to extract.
            refl1d_name : str
                The corresponding refl1d name for the value of entry `orso_name`
            refl1d_resolution_name : str
                The corresponding refl1d error name the error of entry `orso_name`

            Notes
            -----
            This function requires the instrument setting `orso_name` to have a "magnitue" and "error" attribute.
            """
            column_index = next(
                (i for i, c in enumerate(columns) if getattr(c, "physical_quantity", None) == orso_name),
                None,
            )
            if column_index is not None:
                # NOTE: this is based on column being second index (under debate in ORSO)
                header_out[refl1d_name] = data[:, column_index]
                cname = columns[column_index].name
                resolution_index = next(
                    (i for i, c in enumerate(columns) if getattr(c, "error_of", None) == cname),
                    None,
                )
                if resolution_index is not None:
                    header_out[refl1d_resolution_name] = data[:, resolution_index]
            else:
                v = getattr(settings, orso_name, None)
                if hasattr(v, "magnitude"):
                    header_out[refl1d_name] = v.magnitude
                if hasattr(v, "error"):
                    header_out[refl1d_resolution_name] = v.error.error_value

        get_key("incident_angle", "angle", "angular_resolution")
        get_key("wavelength", "wavelength", "wavelength_resolution")

        entries_out.append((header_out, np.array(data).T))
    return entries_out


def load4(
    filename,
    keysep=":",
    sep=None,
    comment="#",
    name=None,
    intensity=1,
    background=0,
    back_absorption=1,
    back_reflectivity=False,
    Aguide=BASE_GUIDE_ANGLE,
    H=0,
    theta_offset=None,
    sample_broadening=None,
    L=None,
    dL=None,
    T=None,
    dT=None,
    dR=None,
    FWHM=False,
    radiation=None,
    columns=None,
    data_range=(None, None),
    resolution="normal",
    oversampling=None,
):
    r"""
    Load in four column data Q, R, dR, dQ.

    The file is loaded with *bumps.data.parse_multi*.  *keysep*
    defaults to ':' so that header data looks like JSON key: value
    pairs.  *sep* is None so that the data uses white-space separated
    columns.  *comment* is the standard '#' comment character, used
    for "# key: value" lines, for commenting out data lines using
    "#number number number number", and for adding comments after
    individual data lines.  The parser isn't very sophisticated, so
    be nice.

    *intensity* is the overall beam intensity, *background* is the
    overall background level, and *back_absorption* is the relative
    intensity of data measured at negative Q compared to positive Q
    data.  These can be values or a bumps *Parameter* objects.

    *back_reflectivity* is True if reflectivity was measured through
    the substrate.  This allows you to arrange the model from substrate
    to surface regardless of whether you are measuring through the
    substrate or reflecting off the surface.

    *theta_offset* indicates sample alignment.  In order to use theta
    offset you need to be able to convert from Q to wavelength and angle
    by providing values for the wavelength or the angle, and the associated
    resolution.

    *L*, *dL* in Angstroms can be used to recover angle and angular resolution
    for monochromatic sources where wavelength is fixed and angle is varying.
    These values can also be stored in the file header as::

        # wavelength: 4.75  # Ang
        # wavelength_resolution: 0.02  # Ang (1-sigma)

    *T*, *dT* in degrees can be used to recover wavelength and wavelength
    dispersion for time of flight sources where angle is fixed and wavelength
    is varying, or you can store them in the header of the file::

        # angle: 2  # degrees
        # angular_resolution: 0.2  # degrees (1-sigma)

    If both angle and wavelength are varying in the data, you can specify
    a separate value for each point, such the following::

        # wavelength: [1, 1.2, 1.5, 2.0, ...]
        # wavelength_resolution: [0.02, 0.02, 0.02, ...]

    *dR* can be used to replace the uncertainty estimate for R in the
    file with $\Delta R = R * \text{dR}$.  This allows files with only
    two columns, *Q* and *R* to be loaded.  Note that points with *dR=0*
    are automatically set to the minimum *dR>0* in the dataset.

    Instead of constants, you can provide function, *dT = lambda T: f(T)*,
    *dL = lambda L: f(L)* or *dR = lambda Q, R, dR: f(Q, R, dR)* for more
    complex relationships (with *dR()* returning 1-$\sigma$ $\Delta R$).

    *sample_broadening* in degrees FWHM adds to the angular_resolution.
    Scale 1-$\sigma$ rms by $2 \surd(2 \ln 2) \approx 2.34$ to convert to FWHM.

    *Aguide* and *H* are parameters for polarized beam measurements
    indicating the magnitude and direction of the applied field.

    Polarized data is represented using a multi-section data file,
    with blank lines separating each section.  Each section must
    have a polarization keyword, with value "++", "+-", "-+" or "--".

    *FWHM* is True if dQ, dT, dL are given as FWHM rather than 1-$\sigma$.
    *dR* is always 1-$\sigma$.  *sample_broadening* is always FWHM.

    *radiation* is 'xray' or 'neutron', depending on whether X-ray or
    neutron scattering length density calculator should be used for
    determining the scattering length density of a material.
    Default is 'neutron'

    *columns* is a string giving the column order in the file.  Default
    order is "Q R dR dQ".  Note: include dR and dQ even if the file only
    has two or three columns, but put the missing columns at the end.

    *data_range* indicates which data rows to use.  Arguments are the
    same as the list slice arguments, *(start, stop, step)*.  This follows
    the usual semantics of list slicing, *L[start:stop:step]*, with
    0-origin indices, stop is last plus one and step optional.  Use negative
    numbers to count from the end.  Default is *(None, None)* for the entire
    data set.

    *resolution* is 'normal' (default) or 'uniform'. Use uniform if you
    are merging Q points from a finely stepped energy sensitive measurement.

    *oversampling* is None or a positive integer indicating how many points to add
    between data point to support sparse data with denser theory (for PolarizedNeutronProbe)
    """
    json_header_encoding = False

    if filename.endswith(".ort") or filename.endswith(".orb"):
        entries = parse_orso(filename)
    else:
        json_header_encoding = True  # for .refl files, header values are json-encoded
        entries = parse_multi(filename, keysep=keysep, sep=sep, comment=comment)

    if columns:
        actual = columns.split()
        natural = "Q R dR dQ".split()
        column_order = [actual.index(k) for k in natural]
    else:
        column_order = [0, 1, 2, 3]
    index = slice(*data_range)
    probe_args = dict(
        name=name,
        filename=filename,
        intensity=intensity,
        background=background,
        back_absorption=back_absorption,
        back_reflectivity=back_reflectivity,
        theta_offset=theta_offset,
        sample_broadening=sample_broadening,
        resolution=resolution,
    )
    data_args = dict(
        radiation=radiation,
        FWHM=FWHM,
        T=T,
        L=L,
        dT=dT,
        dL=dL,
        dR=dR,
        column_order=column_order,
        index=index,
    )
    if len(entries) == 1:
        probe = _data_as_probe(entries[0], json_header_encoding, probe_args, **data_args)
    else:
        data_by_xs = {
            strip_quotes(entry[0]["polarization"]): _data_as_probe(entry, json_header_encoding, probe_args, **data_args)
            for entry in entries
        }
        if not set(data_by_xs.keys()) <= set("-- -+ +- ++".split()):
            raise ValueError("Unknown cross sections in: " + ", ".join(sorted(data_by_xs.keys())))
        xs = [data_by_xs.get(xs, None) for xs in ("--", "-+", "+-", "++")]

        if any(isinstance(d, QProbe) for d in xs if d is not None):
            probe = PolarizedQProbe(xs, Aguide=Aguide, H=H)
        else:
            probe = PolarizedNeutronProbe(xs, Aguide=Aguide, H=H, oversampling=oversampling)
    return probe


def _data_as_probe(
    entry,
    json_header_encoding,
    probe_args,
    T,
    L,
    dT,
    dL,
    dR,
    FWHM,
    radiation,
    column_order,
    index,
):
    decoder = json.loads if json_header_encoding else lambda x: x
    name = probe_args["filename"]
    header, data = entry
    if len(data) == 2:
        data_Q, data_R = (data[k][index] for k in column_order[:2])
        data_dR, data_dQ = None, None
        if dR is None:
            raise ValueError("Need dR for two column data in %r" % name)
    elif len(data) == 3:
        data_Q, data_R, data_dR = (data[k][index] for k in column_order[:3])
        data_dQ = None
    else:
        data_Q, data_R, data_dR, data_dQ = (data[k][index] for k in column_order)

    if FWHM and data_dQ is not None:  # dQ is already 1-sigma when not FWHM
        data_dQ = FWHM2sigma(data_dQ)

    # Override dR in the file if desired.
    # Make sure the computed dR is positive (otherwise chisq is infinite) by
    # choosing the smallest positive uncertainty to replace the invalid values.
    if dR is not None:
        data_dR = dR(data_Q, data_R, data_dR) if callable(dR) else data_R * dR
        data_dR[data_dR <= 0] = np.min(data_dR[data_dR > 0])

    # support calculation of sld from material based on radiation type
    if radiation is not None:
        data_radiation = radiation
    elif "radiation" in header:
        data_radiation = decoder(header["radiation"])
    else:
        # Default to neutron data if radiation not given in head.
        data_radiation = "neutron"
        # data_radiation = None

    if data_radiation == "xray":
        make_probe = XrayProbe
    elif data_radiation == "neutron":
        make_probe = NeutronProbe
    else:
        make_probe = Probe

    # Get T,dT,L,dL from header if it is not provided as an argument
    def fetch_key(key, override):
        # Note: pulls header and index pulled from context
        if override is not None:
            return override
        elif key in header:
            v = decoder(header[key])
            return np.array(v)[index] if isinstance(v, list) else v
        else:
            return None

    # Get T and L, either from user input or from datafile.
    data_T = fetch_key("angle", T)
    data_L = fetch_key("wavelength", L)

    # If one of T and L is missing, reconstruct it from Q
    if data_T is None and data_L is not None:
        data_T = QL2T(data_Q, data_L)
    if data_L is None and data_T is not None:
        data_L = QT2L(data_Q, data_T)

    # Get dT and dL, either from user input or from datafile.
    data_dL = fetch_key("wavelength_resolution", dL)
    data_dT = fetch_key("angular_resolution", dT)
    # print(header['angular_resolution'], data_dT)

    # Support dT = f(T), dL = f(L)
    if callable(data_dT):
        if data_T is None:
            raise ValueError("Need T to determine dT for %r" % name)
        data_dT = data_dT(data_T)
    if callable(data_dL):
        if data_L is None:
            raise ValueError("Need L to determine dL for %r" % name)
        data_dL = data_dL(data_L)

    # Convert input dT,dL to FWHM if necessary.
    if data_dL is not None and not FWHM:
        data_dL = sigma2FWHM(data_dL)
    if data_dT is not None and not FWHM:
        data_dT = sigma2FWHM(data_dT)

    # If dT or dL is missing, reconstruct it from Q.
    if data_dT is None and not any(v is None for v in (data_L, data_dL, data_dQ)):
        data_dT = dQdL2dT(data_Q, data_dQ, data_L, data_dL)
    if data_dL is None and not any(v is None for v in (data_T, data_dT, data_dQ)):
        data_dLoL = dQdT2dLoL(data_Q, data_dQ, data_T, data_dT)
        data_dL = data_dLoL * data_L

    # Check reconstruction if user provided any of T, L, dT, or dL.
    # Also, sample_offset or sample_broadening.
    offset = probe_args["theta_offset"]
    broadening = probe_args["sample_broadening"]
    if any(v is not None for v in (T, dT, L, dL, offset, broadening)):
        if data_T is None:
            raise ValueError("Need L to determine T from Q for %r" % name)
        if data_L is None:
            raise ValueError("Need T to determine L from Q for %r" % name)
        if data_dT is None:
            raise ValueError("Need dL to determine dT from dQ for %r" % name)
        if data_dL is None:
            raise ValueError("Need dT to determine dL from dQ for %r" % name)

    # Build the probe, or the Q probe if we don't have angle and wavelength.
    if all(v is not None for v in (data_T, data_L, data_dT, data_dL)):
        probe = make_probe(
            T=data_T,
            dT=data_dT,
            L=data_L,
            dL=data_dL,
            data=(data_R, data_dR),
            dQo=data_dQ,
            **probe_args,
        )
    else:
        # QProbe doesn't accept theta_offset or sample_broadening
        qprobe_args = probe_args.copy()
        qprobe_args.pop("theta_offset")
        qprobe_args.pop("sample_broadening")
        probe = QProbe(data_Q, data_dQ, data=(data_R, data_dR), **qprobe_args)

    return probe
