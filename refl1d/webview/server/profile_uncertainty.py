from typing import Optional
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from refl1d.errors import align_profiles, form_quantiles
from .colors import COLORS

CONTOURS = (68, 95)

def get_color(index: int):
    return COLORS[index % len(COLORS)]

def show_errors(errors, npoints, align):
    fig = make_subplots(rows=2, cols=1)
    contours = None
    show_profiles(errors, align, CONTOURS, npoints, fig=fig, row=1, col=1)
    show_residuals(errors, contours, fig=fig, row=2, col=1)
    return fig

def show_profiles(errors, align, contours, npoints, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None) -> go.Figure:
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)
        # _profiles_draw_align_lines(profiles, slabs, align, axes)

    if contours is not None:
        _profiles_contour(profiles, contours, npoints, fig=fig, row=row, col=col)
    else:
        _profiles_overplot(profiles, fig=fig, row=row, col=col)
    
    return

def show_residuals(errors, contours, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    _, _, Q, residuals = errors
    _residuals_overplot(Q, residuals, fig=fig, row=row, col=col)


def _profiles_overplot(profiles, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    for model_index, (model, group) in enumerate(profiles.items()):
        name = model.name
        absorbing = any((L[2] != 1e-4).any() for L in group)
        magnetic = (len(group[0]) > 3)
        # Note: Use 3 colours per dataset for consistency
        color_index = model_index * 3
        _draw_overplot(group, 1, name + ' rho', fig=fig, color_index=color_index, row=row, col=col)
        if absorbing:
            _draw_overplot(group, 2, name + ' irho', fig=fig, color_index=color_index+1, row=row, col=col)
        if magnetic:
            _draw_overplot(group, 3, name + ' rhoM', fig=fig, color_index=color_index+2, row=row, col=col)
    _profile_labels(fig=fig, row=row, col=col)

def _draw_overplot(group, index, label, fig: go.Figure, color_index: int, row: Optional[int] = None, col: Optional[int] = None):
    color = get_color(color_index)
    for L in group[1:]:
        fig.add_scattergl(x=L[0], y=L[index], opacity=0.1, showlegend=False, mode="lines", line={"color": color}, hoverinfo="skip", row=row, col=col)
    # Plot best
    L = group[0]
    fig.add_scattergl(x=L[0], y=L[index], name=label, mode="lines", line={"color": color}, hoverinfo="skip", row=row, col=col)

def _profiles_contour(profiles, contours, npoints, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    for model_index, (model, group) in enumerate(profiles.items()):
        name = model.name if model.name is not None else f"Model {model_index}"
        absorbing = any((L[2] > 1e-4).any() for L in group)
        magnetic = (len(group[0]) > 3)
        # Find limits of all profiles
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        # Note: Use 3 colours per dataset for consistency
        color_index = model_index * 3
        _draw_contours(group, 1, name + ' rho', zp, contours, fig=fig, color_index=color_index, row=row, col=col)
        if absorbing:
            _draw_contours(group, 2, name + ' irho', zp, contours, fig=fig, color_index=color_index+1, row=row, col=col)
        if magnetic:
            _draw_contours(group, 3, name + ' rhoM', zp, contours, fig=fig, color_index=color_index+2, row=row, col=col)
    _profile_labels(fig=fig, row=row, col=col)

def _draw_contours(group, index, label, zp, contours, fig: go.Figure, color_index: int, row: Optional[int] = None, col: Optional[int] = None):
    # Interpolate on common z
    fp = np.vstack([np.interp(zp, L[0], L[index]) for L in group])
    # Plot the quantiles
    color = get_color(color_index)

    _plot_quantiles(zp, fp, contours, color, alpha=None, fig=fig, row=row, col=col)
    # Plot the best
    fig.add_scattergl(x=zp, y=fp[0], mode="lines", name=label, line=dict(color=color))
    # axes.plot(zp, fp[0], '-', label=label, color=dark(color))

def _profile_labels(fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    fig.update_layout(legend=dict(visible=True), template='simple_white')
    fig.update_xaxes(title_text='z (Å)')
    fig.update_yaxes(title_text='SLD (10⁻⁶/Å²)')

def _residuals_overplot(Q, residuals, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    alpha = 0.4
    shift = 0
    for m_index, (m, r) in enumerate(residuals.items()):
        color_index = m_index * 3
        color = get_color(color_index)
        pop_size = r.shape[1] - 1
        print("residuals shape: ", r.shape)
        qq = np.tile(Q[m], pop_size)
        rr = shift + r[:, 1:].flatten()
        fig.add_scattergl(x=qq, y=rr, showlegend=False, mode="markers", marker=dict(color=color), opacity=alpha, row=row, col=col, hoverinfo="skip")
        # for rr in r[:, 1:]:
            # fig.add_scattergl(x=Q[m], y=shift+rr, showlegend=False, mode="markers", marker=dict(color=color), opacity=alpha, row=row, col=col, hoverinfo="skip")
        fig.add_scattergl(x=Q[m], y=shift+r[:, 0],  name=m.name, mode="markers", marker=dict(color=color), row=row, col=col, hoverinfo="skip")
        shift += 5
    _residuals_labels(fig, row=row, col=col)

def _residuals_contour(Q, residuals, contours=CONTOURS):
    import matplotlib.pyplot as plt
    shift = 0
    for m, r in residuals.items():
        color = next_color()
        _plot_quantiles(Q[m], shift+r.T, contours, color)
        plt.plot(Q[m], shift+r[:, 0], '.', label=m.name, markersize=1, color=dark(color))
        # Use 3 colours from cycle so reflectivity matches rho for each dataset
        next_color()
        next_color()
        shift += 5
    _residuals_labels()

def _residuals_labels(fig, row=None, col=None):
    fig.update_layout(legend=dict(visible=True))
    fig.update_xaxes(title_text='Q (1/Å)', row=row, col=col)
    fig.update_yaxes(title_text='Residuals', row=row, col=col)

def _profiles_draw_align_lines(profiles, slabs, align, axes):
    for i, m in enumerate(profiles.keys()):
        t1_offset = _find_offset(slabs[m][0], align) if align != 'auto' else None
        if t1_offset is not None:
            axes.axvline(x=t1_offset, color='grey', label=f"{m}:{i}")

def _plot_quantiles(x, y, contours, color, alpha, fig: go.Figure, row: Optional[int] = None, col: Optional[int] = None):
    """
    Plot quantile curves for a set of lines.

    *x* is the x coordinates for all lines.

    *y* is the y coordinates, one row for each line.

    *contours* is a list of confidence intervals expressed as percents.

    *color* is the color to use for the quantiles.  Quantiles are draw as
    a filled region with alpha transparency.  Higher probability regions
    will be covered with multiple contours, which will make them lighter
    and more saturated.

    *alpha* is the transparency level to use for all fill regions.  The
    default value, alpha=2./(#contours+1), works pretty well.
    """
    _, q = form_quantiles(y, contours)

    if alpha is None:
        alpha = 2. / (len(q) + 1)
    for lo, hi in q:
        fig.add_scattergl(x=x, y=lo, showlegend=False, mode="lines", line=dict(color=color), opacity=alpha, row=row, col=col, hoverinfo="skip")
        fig.add_scattergl(x=x, y=hi, showlegend=False, mode="lines", line=dict(color=color), fill="tonexty", opacity=alpha, row=row, col=col, hoverinfo="skip")
