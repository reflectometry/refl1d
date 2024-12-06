# -*- coding: utf-8 -*-
"""
Visual representation of model uncertainty.

For reflectivity models, this aligns and plots a set of profiles chosen
from the parameter uncertainty distribution, and plots the distribution
of the residual values.

Use *run_errors* in a model file to reload the results of a batch DREAM fit.
"""

__all__ = [
    "align_profiles",
    "calc_errors",
    "reload_errors",
    "run_errors",
    "show_errors",
    "show_profiles",
    "show_residuals",
]

import os
import sys

import numpy as np
from bumps.errplot import reload_errors
from bumps.plotutil import dhsv, form_quantiles, next_color, plot_quantiles

from .sample.reflectivity import BASE_GUIDE_ANGLE
from .utils import asbytes

# CONTOURS = (68, 95, 100)
# CONTOURS = (57, 68, 84, 95, 100)
CONTOURS = (68, 95)

# TODO: we should just keep a certain number of evaluations as a matter of
#       course during sampling rather than recomputing them after the fact.
# TODO: want similar code for covariance matrix based forward analysis
# TODO: need to delegate accumulation of models and plotting to Fitness
# TODO: move run_errors to align.py and use argparse to process sys.argv
# TODO: add controls for the additional parameters to the command line


def run_errors(**kw):
    """
    Command line tool for generating error plots from models.

    Type the following to regenerate the profile contour plots plots:

        $ refl1d align <model>.py <store> [<align>] [0|1|2|n]

    Align is either auto for the current behaviour, or it is an interface
    number. You can align on the center of a layer by adding 0.5 to the
    interface number. You can count interfaces from the surface by prefixing
    with R.  For example, 0 is the substrate interface, R1 is the surface
    interface, 2.5 is the the middle of layer 2 above the substrate.

    You can plot the profiles and residuals on one plot by setting plots to 1,
    on two separate plots by setting plots to 2, or each curve on its own
    plot by setting plots to n. Plots are saved in <store>/<model>-err#.png.
    If plots is 0, then no plots are created.

    Additional parameters include:

        *nshown*, *random* :

            see :func:`bumps.errplot.calc_errors_from_state`

        *contours*, *npoints*, *plots*, *save* :

            see :func:`show_errors`
    """

    load = {"model": None, "store": None, "nshown": 50, "random": True}
    show = {"align": "auto", "plots": 2, "contours": CONTOURS, "npoints": 400, "save": None}

    for k, v in kw.items():
        if k in load:
            load[k] = v
        elif k in show:
            show[k] = v
        else:
            raise TypeError("Unknown argument " + k)

    if len(sys.argv) > 2:
        load["model"], load["store"] = sys.argv[1], sys.argv[2]

        align_str = sys.argv[3] if len(sys.argv) >= 4 else "0"
        if align_str[0] == "R":
            align_str = "-" + align_str[1:]
        show["align"] = float(align_str) if align_str != "auto" else align_str
        plots_str = sys.argv[4] if len(sys.argv) >= 5 else "2"
        show["plots"] = int(plots_str) if plots_str != "n" else plots_str
        # print align, align_str, len(sys.argv), sys.argv

    if not load["store"] or not load["model"]:
        _usage()
        sys.exit(0)

    if show["save"] is None:
        name, _ = os.path.splitext(os.path.basename(load["model"]))
        show["save"] = os.path.join(load["store"], name)

    print("loading... this may take awhile")
    errors = reload_errors(**load)
    print("showing...")
    show_errors(errors, **show)

    if show["plots"] != 0:
        import matplotlib.pyplot as plt

        plt.show()
    sys.exit(0)  # Force refl1d to terminate.


def _usage():
    print(run_errors.__doc__)


def calc_errors(problem, points):
    """
    Align the sample profiles and compute the residual difference from the
    measured reflectivity for a set of points.

    The points should be sampled from the posterior probability
    distribution computed from MCMC, bootstrapping or sampled from
    the error ellipse calculated at the minimum.

    Each of the returned arguments is a dictionary mapping model number to
    error sample data as follows:

    Returns (profiles, slabs, Q, residuals).

    *profiles*

        Arrays of (z, rho, irho) for non-magnetic models or arrays
        of (z, rho, irho, rhoM, thetaM) for magnetic models.  There
        will be one set of arrays returned per error sample.

    *slabs*

        Array of slab thickness for the layers in the models.  There
        will be one array returned per error sample.  Using slab thickness,
        profiles can be aligned on interface boundaries and layer centers.

    *Q*

        Array of Q values for the data points in the model.  The data
        points are the same for all error samples, so only one Q array
        is needed per model.

    *residuals*

        Array of (theory-data)/uncertainty for each data point in
        the measurement.  There will be one array returned per error sample.
    """
    # Find Q
    Q = [_residQ(m) for m in _experiments(problem)]

    # Put best at slot 0, no alignment
    data = [_eval_point(problem, problem.getp())]
    for p in points:
        data.append(_eval_point(problem, p))

    profiles, slabs, residuals = zip(*data)

    # TODO: return sane datastructure
    # Make a hashable version of model which just contains the name
    # attribute, which is all that the rest of this code accesses.
    models = [_HashableModel(m, i) for i, m in enumerate(_experiments(problem))]

    profiles = {h: [v[k] for v in profiles] for k, h in enumerate(models)}
    slabs = {h: [v[k] for v in slabs] for k, h in enumerate(models)}
    residuals = {h: np.asarray([v[k] for v in residuals]).T for k, h in enumerate(models)}
    Q = {h: Q[k] for k, h in enumerate(models)}

    # from .pstruct import pstruct, sstruct
    # print("profiles", sstruct(profiles))
    # print("slabs", sstruct(slabs))
    # print("residuals", sstruct(residuals))
    # print("Q", sstruct(Q))
    # import sys; sys.exit()

    return profiles, slabs, Q, residuals


class _HashableModel:
    name: str
    index: int

    def __init__(self, model, index):
        self.name = model.name if model.name is not None else f"M{index}"
        self.index = index

    def __str__(self):
        return f"model {self.name}: {self.index}"


def _eval_point(problem, p):
    problem.chisq_str()  # Force reflectivity recalculation
    problem.setp(p)
    profiles, residuals, slabs = [], [], []
    for m in _experiments(problem):
        D = m.residuals()
        residuals.append(D + 0)
        slabs_i = [L.thickness.value for L in m.sample[1:-1]]
        slabs.append(np.array(slabs_i))
        if m.ismagnetic:
            z, rho, irho, rhoM, thetaM = m.magnetic_smooth_profile()
            profiles.append((z + 0, rho + 0, irho + 0, rhoM + 0, thetaM + 0))
        else:
            z, rho, irho = m.smooth_profile()
            profiles.append((z + 0, rho + 0, irho + 0))
    return profiles, slabs, residuals


def _experiments(problem):
    """
    Cycle through experiments yielding (k, m) pairs for each experiment.

    The iterator is necessary because bumps substitutes the values from the
    free parameters into the fitness via the model iterator in problem. In
    order to keep the parametersets in sync we need to repeat that iteration
    each time.
    """
    for m in problem.models:
        parts = getattr(m, "parts", [m])
        for p in parts:
            yield p


def _residQ(m):
    if m.probe.polarized:
        return np.hstack([xs.Q for xs in m.probe.xs if xs is not None])
    else:
        return m.probe.Q


def align_profiles(profiles, slabs, align):
    """
    Align profiles for each sample
    """
    return dict((m, _align_profile_set(profiles[m], slabs[m], align)) for m in profiles.keys())


def show_errors(errors, contours=CONTOURS, npoints=200, align="auto", plots=1, save=None, fig=None):
    """
    Plot the aligned profiles and the distribution of the residuals for
    profiles and residuals returned from calc_errors.

    *contours* can be a list of percentiles or [].  If percentiles
    are given, then show uncertainty using a contour plot with the
    given levels, otherwise just overplot sample lines.
    *contours* defaults to [68, 95, 100].

    *npoints* is the number of points to use when generating the
    profile contour.  Since the z values for the various lines do not
    correspond, the contour generator interpolates the entire profile
    range with linear spacing using this number of points.

    *align* is the interface number plus fractional distance within
    the layer following the interface.  For example, use 0 for the
    substrate interface, use -1 for the surface interface, or use 2.5
    for the center of the second slab above the substrate. If *align='auto'*
    then choose an offset that minimizes the cross-correlation between the
    first profile and the current profile.

    *plots* is the number of plots to use (1, 2, or 'n').

    *save* is the basename of the plot to save.  This should usually
    be "<store>/<model>".  The program will add '-err#.png' where '#'
    is the number of the plot.
    """
    import matplotlib.pyplot as plt

    if fig is not None and plots != 1:
        raise ValueError("can only pass in a figure object if exactly 1 plot is requested")

    if plots == 0:  # Don't create plots, just save the data
        _save_profile_data(errors, contours=contours, npoints=npoints, align=align, save=save)
        _save_residual_data(errors, contours=contours, save=save)
    elif plots == 1:  # Subplots for profiles/residuals
        if fig is None:
            fig = plt.gcf()
        ax_profiles = fig.add_subplot(211)
        show_profiles(errors, contours=contours, npoints=npoints, align=align, axes=ax_profiles)
        ax_residuals = fig.add_subplot(212)
        show_residuals(errors, contours=contours, axes=ax_residuals)
        if save:
            plt.savefig(save + "-err.png")
    elif plots == 2:  # Separate plots for profiles/residuals
        show_profiles(errors, contours=contours, npoints=npoints, align=align)
        if save:
            plt.savefig(save + "-err1.png")
        plt.figure()
        show_residuals(errors, contours=contours)
        if save:
            plt.savefig(save + "-err2.png")
    else:  # Multiple plots
        profiles, slabs, Q, residuals = errors
        fignum = 1
        for m in profiles.keys():
            plt.figure()
            show_profiles(
                errors=({m: profiles[m]}, {m: slabs[m]}, None, None), contours=contours, npoints=npoints, align=align
            )
            if save:
                plt.savefig(save + "-err%d.png" % fignum)
            fignum += 1
        for m in residuals.keys():
            plt.figure()
            show_residuals(errors=(None, None, {m: Q[m]}, {m: residuals[m]}), contours=contours)
            if save:
                plt.savefig(save + "-err%d.png" % fignum)
            fignum += 1


def show_profiles(errors, align, contours, npoints, axes=None):
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)
        _profiles_draw_align_lines(profiles, slabs, align, axes)

    if contours:
        _profiles_contour(profiles, contours, npoints, axes=axes)
    else:
        _profiles_overplot(profiles, axes=axes)


def show_residuals(errors, contours, axes=None):
    _, _, Q, residuals = errors

    if False and contours:
        _residuals_contour(Q, residuals, contours=contours)
    else:
        _residuals_overplot(Q, residuals, axes=axes)


def _save_profile_data(errors, align, contours, npoints, save):
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)
    k = 1
    for title, _, group in sorted(
        ((m.name, m.index, group) for m, group in profiles.items()), key=lambda x: (x[0], x[1])
    ):
        # Find limits of all profiles
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)

        absorbing = any((L[2] != 1e-4).any() for L in group)
        magnetic = len(group[0]) > 3
        twist = magnetic and any((L[4] != BASE_GUIDE_ANGLE).any() for L in group)

        data, columns = _build_profile_matrix(group, 1, zp, contours)
        _write_file(save + "_rho_contour%d.dat" % k, data, title, columns)
        if absorbing:
            data, columns = _build_profile_matrix(group, 2, zp, contours)
            _write_file(save + "_irho_contour%d.dat" % k, data, title, columns)
        if magnetic:
            data, columns = _build_profile_matrix(group, 3, zp, contours)
            _write_file(save + "_rhoM_contour%d.dat" % k, data, title, columns)
        if twist:
            data, columns = _build_profile_matrix(group, 4, zp, contours)
            _write_file(save + "_thetaM_contour%d.dat" % k, data, title, columns)
        k += 1


def _build_profile_matrix(group, index, zp, contours):
    # Interpolate to common z
    fp = np.vstack([np.interp(zp, L[0], L[index]) for L in group])
    # Find quantiles
    q, qval = form_quantiles(fp, contours)
    # Build and return data columns
    columns = ["z", "best"] + list("%g%%" % v for v in 100 * q.flatten())
    data = np.vstack((zp, fp[0], np.reshape(qval, (-1, qval.shape[2]))))
    return data, columns


def _save_residual_data(errors, contours, save):
    _, _, Q, residuals = errors
    k = 1
    for title, _, x, r in sorted(
        [(m.name, m.index, Q[m], v) for m, v in residuals.items()], key=lambda x: (x[0], x[1])
    ):
        q, qval = form_quantiles(r.T, contours)
        # TODO: should have columns for R, dR as well.
        data = np.vstack((x, r[:, 0], np.reshape(qval, (-1, qval.shape[2]))))
        columns = ["q", "best"] + list("%g%%" % v for v in 100 * q.flatten())
        _write_file(save + "_resid_contour%d.dat" % k, data, title, columns)
        k += 1


def _write_file(path, data, title, columns):
    with open(path, "wb") as fid:
        fid.write(asbytes("# " + title + "\n"))
        fid.write(asbytes("# " + " ".join(columns) + "\n"))
        np.savetxt(fid, data.T)


# ===== Plotting functions =====


def dark(color):
    return dhsv(color, dv=-0.2)


def _profiles_overplot(profiles, axes=None):
    for model, group in profiles.items():
        name = model.name
        absorbing = any((L[2] != 1e-4).any() for L in group)
        magnetic = len(group[0]) > 3
        # Note: Use 3 colours per dataset for consistency
        _draw_overplot(group, 1, name + " rho", axes=axes)
        if absorbing:
            _draw_overplot(group, 2, name + " irho", axes=axes)
        else:
            next_color(axes=axes)
        if magnetic:
            _draw_overplot(group, 3, name + " rhoM", axes=axes)
        else:
            next_color(axes=axes)
    _profile_labels(axes=axes)


def _draw_overplot(group, index, label, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    alpha = 0.1
    color = next_color(axes=axes)
    for L in group[1:]:
        axes.plot(L[0], L[index], "-", color=color, alpha=alpha)
    # Plot best
    L = group[0]
    axes.plot(L[0], L[index], "-", label=label, color=dark(color))


def _profiles_contour(profiles, contours=CONTOURS, npoints=200, axes=None):
    for model, group in profiles.items():
        name = model.name if model.name is not None else "model"
        absorbing = any((L[2] > 1e-4).any() for L in group)
        magnetic = len(group[0]) > 3
        # Find limits of all profiles
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        # Note: Use 3 colours per dataset for consistency
        _draw_contours(group, 1, name + " rho", zp, contours, axes=axes)
        if absorbing:
            _draw_contours(group, 2, name + " irho", zp, contours, axes=axes)
        else:
            next_color(axes=axes)
        if magnetic:
            _draw_contours(group, 3, name + " rhoM", zp, contours, axes=axes)
        else:
            next_color(axes=axes)
    _profile_labels(axes=axes)


def _draw_contours(group, index, label, zp, contours, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    color = next_color(axes=axes)
    # Interpolate on common z
    fp = np.vstack([np.interp(zp, L[0], L[index]) for L in group])
    # Plot the quantiles
    plot_quantiles(zp, fp, contours, color, axes=axes)
    # Plot the best
    axes.plot(zp, fp[0], "-", label=label, color=dark(color))


def _profile_labels(axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    axes.legend()
    axes.set_xlabel("z (Å)")
    axes.set_ylabel("SLD (10⁻⁶/Å²)")


def _residuals_overplot(Q, residuals, axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    alpha = 0.4
    shift = 0
    for m, r in residuals.items():
        color = next_color(axes=axes)
        axes.plot(Q[m], shift + r[:, 1:], ".", markersize=1, color=color, alpha=alpha)
        axes.plot(Q[m], shift + r[:, 0], ".", label=m.name, markersize=1, color=dark(color))
        # Use 3 colours from cycle so reflectivity matches rho for each dataset
        next_color(axes=axes)
        next_color(axes=axes)
        shift += 5
    _residuals_labels(axes=axes)


def _residuals_contour(Q, residuals, contours=CONTOURS):
    import matplotlib.pyplot as plt

    shift = 0
    for m, r in residuals.items():
        color = next_color()
        plot_quantiles(Q[m], shift + r.T, contours, color)
        plt.plot(Q[m], shift + r[:, 0], ".", label=m.name, markersize=1, color=dark(color))
        # Use 3 colours from cycle so reflectivity matches rho for each dataset
        next_color()
        next_color()
        shift += 5
    _residuals_labels()


def _residuals_labels(axes=None):
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()
    axes.legend()
    axes.set_xlabel("Q (1/Å)")
    axes.set_ylabel("Residuals")


def _profiles_draw_align_lines(profiles, slabs, align, axes):
    for i, m in enumerate(profiles.keys()):
        t1_offset = _find_offset(slabs[m][0], align) if align != "auto" else None
        if t1_offset is not None:
            axes.axvline(x=t1_offset, color="grey", label=f"{m}:{i}")


# ==== Helper functions =====


def _align_profile_set(profiles, slabs, align):
    """
    Align all profiles to the first profile.
    """
    p1 = profiles[0]
    t1_offset = _find_offset(slabs[0], align) if align != "auto" else None
    offsets = [0]
    for p2, t2 in zip(profiles[1:], slabs[1:]):
        offsets.append(_align_profile_pair(p1[0], p1[1], t1_offset, p2[0], p2[1], t2, align))
    profiles = [tuple([group[0] + offset] + list(group[1:])) for offset, group in zip(offsets, profiles)]
    return profiles


def _align_profile_pair(z1, r1, t1_offset, z2, r2, t2, align):
    """
    Use crosscorrelation to align r1 and r2.
    """
    if align == "auto":
        import scipy.signal

        # Assume z1, z2 have the same step size
        n2 = len(r2)
        idx = np.argmax(scipy.signal.correlate(r1, r2, "full"))
        if idx < n2:
            offset = z2[(n2 - 1) - idx] - z1[0]
        else:
            offset = z2[0] - z1[idx - (n2 - 1)]
        return -offset
    else:
        return -(_find_offset(t2, align) - t1_offset)


def _find_offset(v, align):
    """
    Find the offset of k.p, where k is the interface number and p is the
    distance into that interface.

    This may even work for interfaces defined from the left, such as
    -1.5 to specify the middle of the final layer.
    """
    idx = int(align)
    offset = np.sum(v[:idx]) + np.sum((align - idx) * v[idx : idx + 1])
    # print offset, idx, v[:idx], align
    return offset
