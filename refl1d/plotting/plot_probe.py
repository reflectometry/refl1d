"""
Plotting functions for probes.

Design note:
These functions were originally part of the probe classes, and were moved out
for future refactor to have a common plotting interface with webview.
"""

import sys
import numpy as np

from refl1d.probe import Probe, PolarizedNeutronProbe, ProbeSet, spin_asymmetry
from bumps.plotutil import coordinated_colors, auto_shift

import matplotlib.pyplot as plt


def plot(probe, view=None, theory=None, **kwargs):
    """
    General plotting function for probes.
    """
    view = view if view is not None else probe.view

    if type(probe) is PolarizedNeutronProbe:
        plotter = MagneticProbePlotter(probe, theory=theory, **kwargs)
    elif type(probe) is ProbeSet:
        # If we have a ProbeSet, plot each probe in the set and exit
        for p, th in probe.parts(theory):
            plot(p, theory=th, **kwargs)
        return
    else:
        plotter = ProbePlotter(probe, theory=theory, **kwargs)

    # Determine how to plot
    if view == "fresnel":
        plotter.plot_fresnel()
        plotter.y_linear()
    elif view == "q4":
        plotter.plot_Q4()
    elif view == "resolution":
        plotter.plot_resolution()
    elif view is not None and view.startswith("resid"):
        plotter.plot_residuals()
    elif view == "SA":
        plotter.plot_SA()
    else:
        plotter.plot()

    if view is not None and "log" in view:
        plotter.y_log()


class ProbePlotter:
    """
    Plotter for non-magnetic probes
    """

    def __init__(self, probe, **kwargs):
        self.probe = probe
        self.kwargs = kwargs

    def plot(self):
        _plot_pair(self.probe, ylabel="Reflectivity", **self.kwargs)
        plt.yscale("log")
        plt.xscale("log")

    def y_linear(self):
        plt.yscale("linear")

    def y_log(self):
        plt.yscale("log")

    def x_linear(self):
        plt.xscale("linear")

    def x_log(self):
        plt.xscale("log")

    def plot_fresnel(self):
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
        substrate = self.kwargs.get("substrate", None)
        surface = self.kwargs.get("surface", None)

        if substrate is None and surface is None:
            raise TypeError("Fresnel-normalized reflectivity needs substrate or surface")
        F = self.probe.fresnel(substrate=substrate, surface=surface)

        # print("substrate", substrate, "surface", surface)
        def scale(Q, dQ, R, dR, interpolation=0):
            Q, fresnel = self.probe.apply_beam(self.probe.calc_Q, F(self.probe.calc_Q), interpolation=interpolation)
            return Q, dQ, R / fresnel, dR / fresnel

        if substrate is None:
            name = "air:%s" % surface.name
        elif surface is None or isinstance(surface, Vacuum):
            name = substrate.name
        else:
            name = "%s:%s" % (substrate.name, surface.name)
        _plot_pair(self.probe, scale=scale, ylabel="R/(R(%s)" % name, **self.kwargs)

    def plot_Q4(self):
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
            Q4 = 1e-8 * Q**-4 * self.probe.intensity.value + self.probe.background.value
            return Q, dQ, R / Q4, dR / Q4

        # Q4[Q4==0] = 1
        _plot_pair(self.probe, scale=scale, ylabel="R (100 Q)^4", **self.kwargs)

    def plot_resolution(self):
        suffix = self.kwargs.get("suffix", "")
        label = self.kwargs.get("label", None)

        plt.plot(self.probe.Q, self.probe.dQ, label=self.probe.label(prefix=label, suffix=suffix))
        plt.xlabel(r"Q ($\AA^{-1}$)")
        plt.ylabel(r"Q resolution ($1-\sigma \AA^{-1}$)")
        plt.title("Measurement resolution")

    def plot_SA(self):
        """
        This plot is unavailable for this probe.
        """
        pass

    def plot_residuals(self):
        suffix = self.kwargs.get("suffix", "")
        label = self.kwargs.get("label", None)
        plot_shift = self.kwargs.get("plot_shift", None)
        theory = self.kwargs.get("theory", None)

        plot_shift = plot_shift if plot_shift is not None else Probe.residuals_shift
        trans = auto_shift(plot_shift)

        if theory is not None and self.probe.R is not None:
            c = coordinated_colors()
            Q, R = theory
            # In case theory curve is evaluated at more/different points...
            R = np.interp(self.probe.Q, Q, R)
            residual = (R - self.probe.R) / self.probe.dR
            plt.plot(
                self.probe.Q,
                residual,
                ".",
                color=c["light"],
                transform=trans,
                label=self.probe.label(prefix=label, suffix=suffix),
            )
        plt.axhline(1, color="black", ls="--", lw=1)
        plt.axhline(0, color="black", lw=1)
        plt.axhline(-1, color="black", ls="--", lw=1)
        plt.xlabel("Q (inv A)")
        plt.ylabel("(theory-data)/error")
        plt.legend(numpoints=1)


class MagneticProbePlotter(ProbePlotter):
    """
    Plotter for magnetic probes
    """

    def __init__(self, probe, **kwargs):
        super().__init__(probe, **kwargs)

    def magnetic(self, plot_function):
        """
        Loop over the cross sections and plot them
        """
        theory = self.kwargs.get("theory", (None, None, None, None))
        self.kwargs["theory"] = theory

        for x_data, x_th, suffix in zip(self.probe.xs, theory, ("$^{--}$", "$^{-+}$", "$^{+-}$", "$^{++}$")):
            if x_data is not None:
                self.kwargs["suffix"] = suffix
                self.kwargs["theory"] = x_th
                plotter = ProbePlotter(x_data, **self.kwargs)
                fn = getattr(plotter, plot_function)
                fn()

    def plot(self):
        self.magnetic("plot")

    def plot_fresnel(self):
        self.magnetic("plot_fresnel")

    def plot_Q4(self):
        self.magnetic("plot_Q4")

    def plot_resolution(self):
        self.magnetic("plot_resolution")

    def plot_residuals(self):
        self.magnetic("plot_residuals")

    def plot_SA(self):
        theory = self.kwargs.get("theory", (None, None, None, None))
        label = self.kwargs.get("label", None)
        plot_shift = self.kwargs.get("plot_shift", None)

        if self.probe.pp is None or self.probe.mm is None:
            raise TypeError("cannot form spin asymmetry plot without ++ and --")

        plot_shift = plot_shift if plot_shift is not None else Probe.plot_shift
        trans = auto_shift(plot_shift)
        pp, mm = self.probe.pp, self.probe.mm
        c = coordinated_colors()
        if hasattr(pp, "R") and hasattr(mm, "R") and pp.R is not None and mm.R is not None:
            Q, SA, dSA = spin_asymmetry(pp.Q, pp.R, pp.dR, mm.Q, mm.R, mm.dR)
            if dSA is not None:
                res = self.probe.show_resolution if self.probe.show_resolution is not None else Probe.show_resolution
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
            plt.plot(Q, SA, label=self.probe.pp.label(prefix=label, gloss="theory"), transform=trans, color=c["dark"])
        plt.xlabel(r"Q (\AA^{-1})")
        plt.ylabel(r"spin asymmetry $(R^{++} -\, R^{--}) / (R^{++} +\, R^{--})$")
        plt.legend(numpoints=1)


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
    plt.xlabel("Q (1/Angstroms)")
    plt.ylabel(ylabel)
    h = plt.legend(fancybox=True, numpoints=1)
    h.get_frame().set_alpha(0.5)
