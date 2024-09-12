"""
Plotting functions for probes.

Design note:
These functions were originally part of the probe classes, and were moved out
for future refactor to have a common plotting interface with webview.
"""

import sys
from refl1d.probe import Probe, PolarizedNeutronProbe, ProbeSet, spin_asymmetry
from bumps.plotutil import coordinated_colors, auto_shift

import matplotlib.pyplot as plt


def plot(probe, view=None, theory=None, **kwargs):
    """
    General plotting function for probes.
    """
    if type(probe) is PolarizedNeutronProbe:
        polarized_neutron_plot(probe, view, **kwargs)
    elif type(probe) is ProbeSet:
        for p, th in probe.parts(theory):
            plot(p, theory=th, **kwargs)
    else:
        base_plot(probe, view, **kwargs)


def _xs_plot(probe, plotter, theory=None, **kwargs):
    """
    Plot available cross sections
    """

    thismodule = sys.modules[__name__]
    fn = getattr(thismodule, plotter)

    if theory is None:
        theory = (None, None, None, None)
    for x_data, x_th, suffix in zip(probe.xs, theory, ("$^{--}$", "$^{-+}$", "$^{+-}$", "$^{++}$")):
        if x_data is not None:
            fn(theory=x_th, suffix=suffix, **kwargs)


def _probeset_plot(probeset, plotter, theory=None, **kw):
    """
    Plot a ProbeSet
    """
    thismodule = sys.modules[__name__]
    fn = getattr(thismodule, plotter)

    for p, th in probeset.parts(theory):
        fn(theory=th, **kw)


def polarized_neutron_plot(probe, view=None, **kwargs):
    """
    Plot theory against data.

    Need substrate/surface for Fresnel-normalized reflectivity
    """
    view = view if view is not None else probe.view

    if view is None:
        view = Probe.view  # Default to Probe.view

    if view == "linear":
        _xs_plot(probe, "plot_linear", **kwargs)
    elif view == "log":
        _xs_plot(probe, "plot_log", **kwargs)
    elif view == "fresnel":
        _xs_plot(probe, "plot_fresnel", **kwargs)
    elif view == "logfresnel":
        _xs_plot(probe, "plot_logfresnel", **kwargs)
    elif view == "q4":
        _xs_plot(probe, "plot_Q4", **kwargs)
    elif view.startswith("resid"):
        _xs_plot(probe, "plot_residuals", **kwargs)
    elif view == "SA":
        plot_SA(probe, **kwargs)
    elif view == "resolution":
        _xs_plot(probe, "plot_resolution", **kwargs)
    else:
        raise TypeError("incorrect reflectivity view '%s'" % view)


def probeset_plot(probeset, view=None, **kwargs):
    """
    Plot a ProbeSet.
    """
    view = view if view is not None else probe.view

    if view is None:
        view = ProbeSet.view  # Default to Probe.view

    if view == "linear":
        _probeset_plot(probeset, "plot_linear", **kwargs)
    elif view == "log":
        _probeset_plot(probeset, "plot_log", **kwargs)
    elif view == "fresnel":
        _probeset_plot(probeset, "plot_fresnel", **kwargs)
    elif view == "logfresnel":
        _probeset_plot(probeset, "plot_logfresnel", **kwargs)
    elif view == "q4":
        _probeset_plot(probeset, "plot_Q4", **kwargs)
    elif view.startswith("resid"):
        _probeset_plot(probeset, "plot_residuals", **kwargs)
    elif view == "SA":
        _probeset_plot(probeset, **kwargs)
    elif view == "resolution":
        for p in probeset.probes:
            p.plot_resolution(**kwargs)
    else:
        raise TypeError("incorrect reflectivity view '%s'" % view)


def base_plot(probe, view=None, **kwargs):
    """
    Plot theory against data.

    Need substrate/surface for Fresnel-normalized reflectivity
    """
    view = view if view is not None else probe.view

    if view == "linear":
        plot_linear(probe, **kwargs)
    elif view == "log":
        plot_log(probe, **kwargs)
    elif view == "fresnel":
        plot_fresnel(probe, **kwargs)
    elif view == "logfresnel":
        plot_logfresnel(probe, **kwargs)
    elif view == "q4":
        plot_Q4(probe, **kwargs)
    elif view == "resolution":
        plot_resolution(probe, **kwargs)
    elif view.startswith("resid"):
        plot_residuals(probe, **kwargs)
    elif view == "fft":
        plot_fft(probe, **kwargs)
    elif view == "SA":  # SA does not plot since it does not exist
        pass
    else:
        raise TypeError("incorrect reflectivity view '%s'" % view)


def plot_resolution(probe, suffix="", label=None, **kwargs):
    plt.plot(probe.Q, probe.dQ, label=probe.label(prefix=label, suffix=suffix))
    plt.xlabel(r"Q ($\AA^{-1}$)")
    plt.ylabel(r"Q resolution ($1-\sigma \AA^{-1}$)")
    plt.title("Measurement resolution")


def plot_linear(probe, **kwargs):
    """
    Plot the data associated with probe.
    """

    probe._plot_pair(ylabel="Reflectivity", **kwargs)
    plt.yscale("linear")


def plot_log(probe, **kwargs):
    """
    Plot the data associated with probe.
    """

    _plot_pair(probe, ylabel="Reflectivity", **kwargs)
    plt.yscale("log")


def plot_logfresnel(probe, *args, **kw):
    """
    Plot the log Fresnel-normalized reflectivity associated with the probe.
    """

    probe.plot_fresnel(*args, **kw)
    plt.yscale("log")


def plot_fresnel(probe, substrate=None, surface=None, **kwargs):
    r"""
    Plot the Fresnel-normalized reflectivity associated with the probe.

    Note that the Fresnel reflectivity has the intensity and background
    applied before normalizing so that hydrogenated samples display more
    cleanly.  The formula to reproduce the graph is:

    .. math::

            R' = R / (F(Q) I + B)
            \Delta R' = \Delta R / (F(Q) I + B)

    where $I$ is the intensity and $B$ is the background.
    """

    if substrate is None and surface is None:
        raise TypeError("Fresnel-normalized reflectivity needs substrate or surface")
    F = probe.fresnel(substrate=substrate, surface=surface)

    # print("substrate", substrate, "surface", surface)
    def scale(Q, dQ, R, dR, interpolation=0):
        Q, fresnel = probe.apply_beam(probe.calc_Q, F(probe.calc_Q), interpolation=interpolation)
        return Q, dQ, R / fresnel, dR / fresnel

    if substrate is None:
        name = "air:%s" % surface.name
    elif surface is None or isinstance(surface, Vacuum):
        name = substrate.name
    else:
        name = "%s:%s" % (substrate.name, surface.name)
    _plot_pair(probe, scale=scale, ylabel="R/(R(%s)" % name, **kwargs)


def plot_Q4(probe, **kwargs):
    r"""
    Plot the Q**4 reflectivity associated with the probe.

    Note that Q**4 reflectivity has the intensity and background applied
    so that hydrogenated samples display more cleanly.  The formula
    to reproduce the graph is:

    .. math::

            R' = R / ( (100*Q)^{-4} I + B)
            \Delta R' = \Delta R / ( (100*Q)^{-4} I + B )

    where $B$ is the background.
    """

    def scale(Q, dQ, R, dR, interpolation=0):
        # Q4 = np.maximum(1e-8*Q**-4, probe.background.value)
        Q4 = 1e-8 * Q**-4 * probe.intensity.value + probe.background.value
        return Q, dQ, R / Q4, dR / Q4

    # Q4[Q4==0] = 1
    _plot_pair(probe, scale=scale, ylabel="R (100 Q)^4", **kwargs)


def _plot_pair(
    probe,
    theory=None,
    scale=lambda Q, dQ, R, dR, interpolation=0: (Q, dQ, R, dR),
    ylabel="",
    suffix="",
    label=None,
    plot_shift=None,
    **kwargs,
):
    c = coordinated_colors()
    plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
    trans = auto_shift(plot_shift)
    if hasattr(probe, "R") and probe.R is not None:
        Q, dQ, R, dR = scale(probe.Q, probe.dQ, probe.R, probe.dR)
        if not probe.show_resolution:
            dQ = None
        plt.errorbar(
            Q,
            R,
            yerr=dR,
            xerr=dQ,
            capsize=0,
            fmt=".",
            color=c["light"],
            transform=trans,
            label=probe.label(prefix=label, gloss="data", suffix=suffix),
        )
    if theory is not None:
        # TODO: completely restructure interpolation handling
        # Interpolation is used to show the theory curve between the
        # data points.  The _calc_Q points used to predict theory at
        # the measured data are used for the interpolated Q points, with
        # the resolution function centered on each interpolated value.
        # The result is that when interpolation != 0, there are more
        # theory points than data points, and we will need to accomodate
        # this when computing normalization curves for Fresnel and Q^4
        # reflectivity.
        # Issues with current implementation:
        # * If the resolution is too tight, there won't be sufficient
        #   support from _calc_Q to compute theory at Q interpolated.
        # * dQ for the interpolated points uses linear interpolation
        #   of dQ between neighbours.  If measurements with tight and
        #   loose resolution are interleaved the result will look very
        #   strange.
        # * There are too many assumptions about interpolation shared
        #   between Experiment and Probe objects.  In particular, the
        #   Fresnel object needs to be computed at the same interpolated
        #   points as the theory function.
        # * If there are large gaps in the data the interpolation will
        #   not fill them in correctly.  Perhaps we should set _Q_plot
        #   and _calc_Q_plot independently from the data?
        # * We sometimes show the theory curve without resolution
        #   applied; this has not been tested with interpolation
        interpolation = kwargs.get("interpolation", 0)
        Q, R = theory
        Q, dQ, R, dR = scale(Q, 0, R, 0, interpolation=interpolation)
        plt.plot(
            Q,
            R,
            "-",
            color=c["dark"],
            transform=trans,
            label=probe.label(prefix=label, gloss="theory", suffix=suffix),
        )
        # from numpy.fft import fft
        # x, y = Q[1::2], abs(fft(R)[:(len(R)-1)//2])
        # y = y * (R.max()/y.max())
        # plt.plot(x, y, '-')
    plt.xlabel("Q (inv Angstroms)")
    plt.ylabel(ylabel)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    h = plt.legend(fancybox=True, numpoints=1)
    h.get_frame().set_alpha(0.5)


def plot_residuals(probe, theory=None, suffix="", label=None, plot_shift=None, **kwargs):
    plot_shift = plot_shift if plot_shift is not None else Probe.residuals_shift
    trans = auto_shift(plot_shift)
    if theory is not None and probe.R is not None:
        c = coordinated_colors()
        Q, R = theory
        # In case theory curve is evaluated at more/different points...
        R = np.interp(probe.Q, Q, R)
        residual = (R - probe.R) / probe.dR
        plt.plot(
            probe.Q, residual, ".", color=c["light"], transform=trans, label=probe.label(prefix=label, suffix=suffix)
        )
    plt.axhline(1, color="black", ls="--", lw=1)
    plt.axhline(0, color="black", lw=1)
    plt.axhline(-1, color="black", ls="--", lw=1)
    plt.xlabel("Q (inv A)")
    plt.ylabel("(theory-data)/error")
    plt.legend(numpoints=1)


def plot_SA(probe, theory=None, label=None, plot_shift=None, **kwargs):
    if probe.pp is None or probe.mm is None:
        raise TypeError("cannot form spin asymmetry plot without ++ and --")

    plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
    trans = auto_shift(plot_shift)
    pp, mm = probe.pp, probe.mm
    c = coordinated_colors()
    if hasattr(pp, "R") and hasattr(mm, "R") and pp.R is not None and mm.R is not None:
        Q, SA, dSA = spin_asymmetry(pp.Q, pp.R, pp.dR, mm.Q, mm.R, mm.dR)
        if dSA is not None:
            res = probe.show_resolution if probe.show_resolution is not None else Probe.show_resolution
            dQ = pp.dQ if res else None
            plt.errorbar(
                Q,
                SA,
                yerr=dSA,
                xerr=dQ,
                fmt=".",
                capsize=0,
                label=pp.label(prefix=label, gloss="data"),
                transform=trans,
                color=c["light"],
            )
        else:
            plt.plot(Q, SA, ".", label=pp.label(prefix=label, gloss="data"), transform=trans, color=c["light"])
        # Set limits based on max theoretical SA, which is in (-1.0, 1.0)
        # If the error bars are bigger than that, you usually don't care.
        ylim_low, ylim_high = plt.ylim()
        plt.ylim(max(ylim_low, -2.5), min(ylim_high, 2.5))
    if theory is not None:
        mm, mp, pm, pp = theory
        Q, SA, _ = spin_asymmetry(pp[0], pp[1], None, mm[0], mm[1], None)
        plt.plot(Q, SA, label=probe.pp.label(prefix=label, gloss="theory"), transform=trans, color=c["dark"])
    plt.xlabel(r"Q (\AA^{-1})")
    plt.ylabel(r"spin asymmetry $(R^{++} -\, R^{--}) / (R^{++} +\, R^{--})$")
    plt.legend(numpoints=1)
