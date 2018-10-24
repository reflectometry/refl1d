"""
Visual representation of model uncertainty.

For reflectivity models, this aligns and plots a set of profiles chosen
from the parameter uncertainty distribution, and plots the distribution
of the residual values.

Use *run_errors* in a model file to reload the results of a batch DREAM fit.
"""
from __future__ import print_function

__all__ = ['reload_errors', 'run_errors',
           'calc_errors', 'align_profiles',
           'show_errors', 'show_profiles', 'show_residuals',
          ]

import numpy as np
from bumps.plotutil import next_color, dhsv, plot_quantiles, form_quantiles
from bumps.errplot import reload_errors

#_CONTOURS = (68, 95, 100)
#_CONTOURS = (57, 68, 84, 95, 100)
_CONTOURS = (68, 95)

def run_errors(**kw):
    """
    Argument parser for generating error plots from models.

    The model directory should contain a fake model align.py with:

        from refl1d.errors import run_errors
        run_errors(model="", store="", align='auto')

    If you are using the command line then you should be able to type the
    following at the command prompt to generate the plots:

        $ refl1d align.py <model>.py <store> [<align>] [0|1|2|n]

    If you are using the GUI, you will have to set model, store and
    align directly in align.py each time you run.

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
    import os
    import sys
    import pylab

    load = {'model': None, 'store': None, 'nshown': 50, 'random': True}
    show = {'align': 'auto', 'plots': 2,
            'contours': _CONTOURS, 'npoints': 400,
            'save': None}

    for k, v in kw.items():
        if k in load:
            load[k] = v
        elif k in show:
            show[k] = v
        else:
            raise TypeError("Unknown argument "+k)

    if len(sys.argv) > 2:
        load['model'], load['store'] = sys.argv[1], sys.argv[2]

        align_str = sys.argv[3] if len(sys.argv) >= 4 else '0'
        if align_str[0] == 'R':
            align_str = '-'+align_str[1:]
        show['align'] = float(align_str) if align_str != 'auto' else align_str
        plots_str = sys.argv[4] if len(sys.argv) >= 5 else '2'
        show['plots'] = int(plots_str) if plots_str != 'n' else plots_str
        #print align, align_str, len(sys.argv), sys.argv

    if not load['store'] or not load['model']:
        _usage()
    if show['save'] is None:
        show['save'] = os.path.join(load['store'], load['model'][:-3])

    print("loading... this may take awhile")
    errors = reload_errors(**load)
    print("showing...")
    show_errors(errors, **show)
    pylab.show()
    raise KeyboardInterrupt()  # Force refl1d to terminate

def _usage():
    print(run_errors.__doc__)
    raise RuntimeError()


# TODO: we should just keep a certain number of evaluations as a matter of
#       course during sampling rather than recomputing them after the fact.
# TODO: want similar code for covariance matrix based forward analysis
# TODO: need to delegate accumulation of models and plotting to Fitness
# TODO: move run_errors somewhere more appropriate, like cli.py

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
    # Grab the individual samples
    if hasattr(problem, 'models'):
        models = [m.fitness for m in problem.models]
    else:
        models = [problem.fitness]

    experiments = []
    for m in models:
        if hasattr(m, 'parts'):
            experiments.extend(m.parts)
        else:
            experiments.append(m)
    #probes = []
    #for m in experiments:
    #    if hasattr(m.probe, 'probes'):
    #        probes.extend(m.probe.probes)
    #    elif hasattr(m.probe, 'xs'):
    #        probes.extend([p for p in m.probe if p])
    #    else:
    #        probes.append(p)

    # Find Q
    def residQ(m):
        if m.probe.polarized:
            return np.hstack([xs.Q for xs in m.probe.xs if xs is not None])
        else:
            return m.probe.Q
    Q = dict((m, residQ(m)) for m in experiments)

    profiles = dict((m, []) for m in experiments)
    residuals = dict((m, []) for m in experiments)
    slabs = dict((m, []) for m in experiments)
    def record_point():
        problem.chisq() # Force reflectivity recalculation
        for m in experiments:
            D = m.residuals()
            residuals[m].append(D+0)
            slabs_i = [L.thickness.value for L in m.sample[1:-1]]
            slabs[m].append(np.array(slabs_i))
            if m.ismagnetic:
                z, rho, irho, rhoM, thetaM = m.magnetic_step_profile()
                profiles[m].append((z+0, rho+0, irho+0, rhoM+0, thetaM+0))
            else:
                z, rho, irho = m.smooth_profile()
                profiles[m].append((z+0, rho+0, irho+0))
    record_point() # Put best at slot 0, no alignment
    for p in points:
        problem.setp(p)
        record_point()

    # Turn residuals into arrays
    residuals = dict((k, np.asarray(v).T) for k, v in residuals.items())
    return profiles, slabs, Q, residuals

def align_profiles(profiles, slabs, align):
    """
    Align profiles for each sample
    """
    return dict((m, _align_profile_set(profiles[m], slabs[m], align))
                for m in profiles.keys())

def show_errors(errors, contours=_CONTOURS, npoints=200,
                align='auto', plots=1, save=None):
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
    for the center of the second slab above the substrate.

    *plots* is the number of plots to use (1, 2, or 'n').

    *save* is the basename of the plot to save.  This should usually
    be "<store>/<model>".  The program will add '-err#.png' where '#'
    is the number of the plot.
    """
    import pylab

    if plots == 0: # Don't create plots, just save the data
        _save_profile_data(errors, contours=contours, npoints=npoints,
                           align=align, save=save)
        _save_residual_data(errors, contours=contours, save=save)
    elif plots == 1: # Subplots for profiles/residuals
        pylab.subplot(211)
        show_profiles(errors, contours=contours, npoints=npoints, align=align)
        pylab.subplot(212)
        show_residuals(errors, contours=contours)
        if save:
            pylab.savefig(save+"-err.png")
    elif plots == 2:  # Separate plots for profiles/residuals
        show_profiles(errors, contours=contours, npoints=npoints, align=align)
        if save:
            pylab.savefig(save+"-err1.png")
        pylab.figure()
        show_residuals(errors, contours=contours)
        if save:
            pylab.savefig(save+"-err2.png")
    else: # Multiple plots
        profiles, slabs, Q, residuals = errors
        fignum = 1
        for m in profiles.keys():
            pylab.figure()
            show_profiles(errors=({m:profiles[m]}, {m:slabs[m]}, None, None),
                          contours=contours, npoints=npoints, align=align)
            if save:
                pylab.savefig(save+"-err%d.png"%fignum)
            fignum += 1
        for m in residuals.keys():
            pylab.figure()
            show_residuals(errors=(None, None, {m:Q[m]}, {m:residuals[m]}),
                           contours=contours)
            if save:
                pylab.savefig(save+"-err%d.png"%fignum)
            fignum += 1

def show_profiles(errors, align, contours, npoints):
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)

    if contours:
        _profiles_contour(profiles, contours, npoints)
    else:
        _profiles_overplot(profiles)


def show_residuals(errors, contours):
    _, _, Q, residuals = errors

    if False and contours:
        _residuals_contour(Q, residuals, contours=contours)
    else:
        _residuals_overplot(Q, residuals)


def _save_profile_data(errors, align, contours, npoints, save):
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)
    k = 1
    for title, p in sorted((m.name, p) for m, p in profiles.items()):
        # Find limits of all profiles
        z = np.hstack([line[0] for line in p])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        # Interpolate rho on common z
        rho = np.vstack([np.interp(zp, L[0], L[1]) for L in p])
        p, q = form_quantiles(rho, contours)
        data = np.vstack((zp, rho[0], np.reshape(q, (-1, q.shape[2]))))
        columns = ["z", "best"] + list("%g%%"%v for v in 100*p.flatten())
        _write_file(save+"_rho_contour%d.dat"%k, data, title, columns)
        if len(p[0]) > 3:
            rhoM = np.vstack([np.interp(zp, L[0], L[3]) for L in p])
            p, q = form_quantiles(rhoM, contours)
            data = np.vstack((zp, rhoM[0], np.reshape(q, (-1, q.shape[2]))))
            _write_file(save+"_rhoM_contour%d.dat"%k, data, title, columns)
        k += 1

def _save_residual_data(errors, contours, save):
    _, _, Q, residuals = errors
    k = 1
    for title, x, r in sorted((m.name, Q[m], v) for m, v in residuals.items()):
        p, q = form_quantiles(r.T, contours)
        # TODO: should have columns for R, dR as well.
        data = np.vstack((x, r[:, 0], np.reshape(q, (-1, q.shape[2]))))
        columns = ["q", "best"] + list("%g%%"%v for v in 100*p.flatten())
        _write_file(save+"_resid_contour%d.dat"%k, data, title, columns)
        k += 1

def _write_file(path, data, title, columns):
    with open(path, "wt") as fid:
        fid.write("# "+title+"\n")
        fid.write("# "+" ".join(columns)+"\n")
        np.savetxt(fid, data.T)


# ===== Plotting functions =====

def _profiles_labels(magnetic):
    import pylab
    pylab.xlabel('z (A)')
    if magnetic:
        pylab.ylabel('rho (1/A^2)')
    else:
        pylab.ylabel('rho, rhoM (1/A^2)')

def _profiles_overplot(profiles):
    import pylab

    alpha = 0.1
    any_magnetic = False
    for p in profiles.values():
        if len(p[0]) == 3:
            rho_color = next_color()
            for z, rho, _ in p[1:]:
                pylab.plot(z, rho, '-', color=rho_color, alpha=alpha)
            # Plot best
            z, rho, _ = p[0]
            pylab.plot(z, rho, '-', color=dhsv(rho_color, dv=-0.2))
        else:
            any_magnetic = True
            rho_color = next_color()
            rhoM_color = next_color()
            for z, rho, _, rhoM, _ in p[1:]:
                pylab.plot(z, rho, '-', color=rho_color, alpha=alpha)
                pylab.plot(z, rhoM, '-', color=rhoM_color, alpha=alpha)
            # Plot best
            z, rho, _, rhoM, _ = p[0]
            pylab.plot(z, rho, '-', color=dhsv(rho_color, dv=-0.2))
            pylab.plot(z, rhoM, '-', color=dhsv(rhoM_color, dv=-0.2))
    _profiles_labels(any_magnetic)

def _profiles_contour(profiles, contours=_CONTOURS, npoints=200):
    import pylab

    any_magnetic = False
    for p in profiles.values():
        # Find limits of all profiles
        z = np.hstack([line[0] for line in p])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        if len(p[0]) == 3:
            rho_color = next_color()
            # Interpolate rho on common z
            rho = np.vstack([np.interp(zp, L[0], L[1]) for L in p])
            # Plot the quantiles
            plot_quantiles(zp, rho, contours, rho_color)
            # Plot the best
            pylab.plot(zp, rho[0], '-', color=dhsv(rho_color, dv=-0.2))
        else:
            any_magnetic = True
            rho_color = next_color()
            rhoM_color = next_color()
            # Interpolate rho, rhoM on common z
            rho = np.vstack([np.interp(zp, L[0], L[1]) for L in p])
            rhoM = np.vstack([np.interp(zp, L[0], L[3]) for L in p])
            # Plot the quantiles
            plot_quantiles(zp, rho, contours, rho_color)
            plot_quantiles(zp, rhoM, contours, rhoM_color)
            # Plot the best
            pylab.plot(zp, rho[0], '-', color=dhsv(rho_color, dv=-0.2))
            pylab.plot(zp, rhoM[0], '-', color=dhsv(rhoM_color, dv=-0.2))
    _profiles_labels(any_magnetic)

def _residuals_labels():
    import pylab

    pylab.xlabel('Q (1/A)')
    pylab.ylabel('Residuals')

def _residuals_overplot(Q, residuals):
    import pylab
    alpha = 0.4
    shift = 0
    for m, r in residuals.items():
        color = next_color()
        pylab.plot(Q[m], shift+r[:, 1:], '.', markersize=1,
                   color=color, alpha=alpha)
        pylab.plot(Q[m], shift+r[:, 0], '.', markersize=1,
                   color=dhsv(color, dv=-0.2)) # best
        shift += 5
    _residuals_labels()

def _residuals_contour(Q, residuals, contours=_CONTOURS):
    import pylab
    shift = 0
    for m, r in residuals.items():
        color = next_color()
        plot_quantiles(Q[m], shift+r.T, contours, color)
        pylab.plot(Q[m], shift+r[:, 0], '.', markersize=1,
                   color=dhsv(color, dv=-0.2)) # best
        shift += 5
    _residuals_labels()

# ==== Helper functions =====

def _align_profile_set(profiles, slabs, align):
    """
    Align all profiles to the first profile.
    """
    p1 = profiles[0]
    t1_offset = _find_offset(slabs[0], align) if align != 'auto' else None
    offsets = [0]
    for p2, t2 in zip(profiles[1:], slabs[1:]):
        offsets.append(_align_profile_pair(p1[0], p1[1], t1_offset,
                                           p2[0], p2[1], t2,
                                           align))
    profiles = [tuple([p[0]+offset]+list(p[1:]))
                for offset, p in zip(offsets, profiles)]
    return profiles

def _align_profile_pair(z1, r1, t1_offset, z2, r2, t2, align):
    """
    Use autocorrelation to align r1 and r2.
    """
    if align == 'auto':
        # Assume z1, z2 have the same step size
        n2 = len(r2)
        idx = np.argmax(np.correlate(r1, r2, 'full'))
        if idx < n2:
            offset = z2[(n2-1)-idx] - z1[0]
        else:
            offset = z2[0] - z1[idx-(n2-1)]
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
    offset = np.sum(v[:idx]) + (align-idx)*v[idx]
    #print offset, idx, v[:idx], align
    return offset
