from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objs as go

from refl1d.uncertainty import _find_offset, align_profiles, form_quantiles
from .colors import COLORS

ErrorType = tuple[
    dict[str, list[np.ndarray]],  # profiles
    dict[str, np.ndarray],  # slabs
    dict[str, np.ndarray],  # Q
    dict[str, np.ndarray],  # residuals
]

CONTOURS = (68, 95)


def get_color(index: int):
    return COLORS[index % len(COLORS)]


def show_errors(errors: ErrorType, npoints: int, align: bool, residuals: bool = True):
    from plotly.subplots import make_subplots

    specs = (
        [[{"secondary_y": True, "rowspan": 1}], [{}]] if residuals else [[{"secondary_y": True, "rowspan": 2}], [None]]
    )
    fig = make_subplots(rows=2, cols=1, specs=specs)
    fig.update_layout(template=None)
    contour_data = show_profiles(errors, align, CONTOURS, npoints, fig=fig, row=1, col=1)
    if residuals:
        show_residuals(errors, None, fig=fig, row=2, col=1)
    return dict(fig=fig, contour_data=contour_data, contours=CONTOURS)


def show_profiles(
    errors: ErrorType,
    align: bool,
    contours: Optional[tuple[float]],
    npoints: int,
    fig: "go.Figure",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> "go.Figure":
    profiles, slabs, _, _ = errors
    if align is not None:
        profiles = align_profiles(profiles, slabs, align)

    contour_data = None
    if contours is not None:
        contour_data = _profiles_contour(profiles, contours, npoints, fig=fig, row=row, col=col)
    else:
        _profiles_overplot(profiles, fig=fig, row=row, col=col)

    if align != "auto":
        _profiles_draw_align_lines(profiles, slabs, align, fig=fig, row=row, col=col)

    return contour_data


def show_residuals(
    errors: ErrorType,
    contours: Optional[tuple[float]],
    fig: "go.Figure",
    row: Optional[int] = None,
    col: Optional[int] = None,
):
    _, _, Q, residuals = errors

    if contours is not None:
        _residuals_contour(Q, residuals, contours, fig, row, col)
    else:
        _residuals_overplot(Q, residuals, fig=fig, row=row, col=col)


def _profiles_overplot(
    profiles: dict[str, list[np.ndarray]], fig: "go.Figure", row: Optional[int] = None, col: Optional[int] = None
):
    has_magnetism = False
    for model_index, (model, group) in enumerate(profiles.items()):
        name = model.name
        absorbing = any((L[2] != 1e-4).any() for L in group)
        magnetic = len(group[0]) > 3
        # Note: Use 3 colours per dataset for consistency
        color_index = model_index * 4
        _draw_overplot(group, 1, name + " rho", fig=fig, color_index=color_index, row=row, col=col, secondary_y=False)
        if absorbing:
            _draw_overplot(
                group, 2, name + " irho", fig=fig, color_index=color_index + 1, row=row, col=col, secondary_y=False
            )
        if magnetic:
            has_magnetism = True
            _draw_overplot(
                group, 3, name + " rhoM", fig=fig, color_index=color_index + 2, row=row, col=col, secondary_y=False
            )
            _draw_overplot(
                group, 4, name + " thetaM", fig=fig, color_index=color_index + 3, row=row, col=col, secondary_y=True
            )
    _profile_labels(fig=fig, row=row, col=col, magnetic=has_magnetism)


def _draw_overplot(
    group,
    index,
    label,
    fig: "go.Figure",
    color_index: int,
    row: Optional[int] = None,
    col: Optional[int] = None,
    secondary_y: bool = False,
):
    color = get_color(color_index)
    for L in group[1:]:
        fig.add_scattergl(
            x=L[0],
            y=L[index],
            opacity=0.1,
            showlegend=False,
            mode="lines",
            line={"color": color},
            hoverinfo="skip",
            row=row,
            col=col,
            secondary_y=secondary_y,
        )
    # Plot best
    L = group[0]
    fig.add_scattergl(
        x=L[0],
        y=L[index],
        name=label,
        mode="lines",
        line={"color": color},
        hoverinfo="skip",
        row=row,
        col=col,
        secondary_y=secondary_y,
    )


def _profiles_contour(
    profiles: dict[str, list[np.ndarray]],
    contours: Optional[tuple[float]],
    npoints,
    fig: "go.Figure",
    row: Optional[int] = None,
    col: Optional[int] = None,
):
    contour_data = {}
    has_magnetism = False
    for model_index, (model, group) in enumerate(profiles.items()):
        name = model.name if model.name is not None else f"Model {model_index}"
        if name in contour_data:
            name += f" {model_index}"
        contour_data[name] = {}
        absorbing = any((L[2] > 1e-4).any() for L in group)
        magnetic = len(group[0]) > 3
        # Find limits of all profiles
        z = np.hstack([line[0] for line in group])
        zp = np.linspace(np.min(z), np.max(z), npoints)
        contour_data[name]["z"] = zp
        contour_data[name]["data"] = {}
        # Note: Use 4 colours per dataset for consistency
        color_index = model_index * 4
        contour_data[name]["data"]["rho"] = _draw_contours(
            group, 1, name + " rho", zp, contours, fig=fig, color_index=color_index, row=row, col=col, secondary_y=False
        )
        if absorbing:
            contour_data[name]["data"]["irho"] = _draw_contours(
                group,
                2,
                name + " irho",
                zp,
                contours,
                fig=fig,
                color_index=color_index + 1,
                row=row,
                col=col,
                secondary_y=False,
            )
        if magnetic:
            has_magnetism = True
            contour_data[name]["data"]["rhoM"] = _draw_contours(
                group,
                3,
                name + " rhoM",
                zp,
                contours,
                fig=fig,
                color_index=color_index + 2,
                row=row,
                col=col,
                secondary_y=False,
            )
            contour_data[name]["data"]["thetaM"] = _draw_contours(
                group,
                4,
                name + " thetaM",
                zp,
                contours,
                fig=fig,
                color_index=color_index + 3,
                row=row,
                col=col,
                secondary_y=True,
            )

    _profile_labels(fig=fig, row=row, col=col, magnetic=has_magnetism)
    return contour_data


def _draw_contours(
    group,
    index,
    label,
    zp,
    contours,
    fig: "go.Figure",
    color_index: int,
    row: Optional[int] = None,
    col: Optional[int] = None,
    secondary_y: bool = False,
):
    # Interpolate on common z
    fp = np.vstack([np.interp(zp, L[0], L[index]) for L in group])
    # Plot the quantiles
    color = get_color(color_index)
    legendgroup = f"group_{color_index}"
    fig.add_scattergl(
        x=zp,
        y=fp[0],
        mode="lines",
        name=label,
        line=dict(color=color),
        legendgroup=legendgroup,
        row=row,
        col=col,
        secondary_y=secondary_y,
    )
    named_contours = _plot_quantiles(
        zp, fp, contours, color, alpha=None, fig=fig, row=row, col=col, legendgroup=legendgroup, secondary_y=secondary_y
    )
    named_contours["best"] = fp[0]
    # Plot the best
    # axes.plot(zp, fp[0], '-', label=label, color=dark(color))
    return named_contours


def _profile_labels(fig: "go.Figure", row: Optional[int] = None, col: Optional[int] = None, magnetic: bool = False):
    fig.update_xaxes(title_text="z (Å)", row=row, col=col, showline=True, zeroline=False)
    fig.update_yaxes(title_text="SLD (10⁻⁶/Å²)", row=row, col=col, showline=True, secondary_y=False, side="left")
    if magnetic:
        fig.update_yaxes(
            automargin=True,
            showgrid=False,
            title_text="Magnetic Angle θ<sub>M</sub> / °",
            row=row,
            col=col,
            showline=True,
            secondary_y=True,
            side="right",
        )


def _residuals_overplot(Q, residuals, fig: "go.Figure", row: Optional[int] = None, col: Optional[int] = None):
    alpha = 0.6
    shift = 0
    for m_index, (m, r) in enumerate(residuals.items()):
        color_index = m_index * 4
        color = get_color(color_index)
        pop_size = r.shape[1] - 1
        qq = np.tile(Q[m], pop_size)
        rr = shift + r[:, 1:].ravel(order="F")
        fig.add_scattergl(
            x=qq,
            y=rr,
            showlegend=False,
            mode="markers",
            marker=dict(color=color, size=2),
            opacity=alpha,
            row=row,
            col=col,
            hoverinfo="skip",
        )
        fig.add_scattergl(
            x=Q[m],
            y=shift + r[:, 0],
            name=m.name,
            mode="markers",
            marker=dict(color=color, size=3),
            row=row,
            col=col,
            hoverinfo="skip",
        )
        shift += 5
    _residuals_labels(fig, row=row, col=col)


def _residuals_contour(Q, residuals, contours, fig: "go.Figure", row: Optional[int] = None, col: Optional[int] = None):
    shift = 0
    for model_index, (m, r) in enumerate(residuals.items()):
        color_index = model_index * 4
        color = get_color(color_index)
        x = Q[m]
        # residuals x may not be sorted:
        sort_order = np.argsort(x)
        sorted_x = x[sort_order]
        fig.add_scattergl(
            x=sorted_x,
            y=shift + r[:, 0],
            mode="markers",
            name=m.name,
            marker=dict(color=color, size=3),
            row=row,
            col=col,
        )
        _plot_quantiles(sorted_x, shift + r.T[:, sort_order], contours, color, alpha=None, fig=fig, row=row, col=col)
        # Plot the best
        shift += 5
    _residuals_labels(fig, row=row, col=col)


def _residuals_labels(fig, row=None, col=None):
    fig.update_xaxes(title_text="Q (1/Å)", row=row, col=col, showline=True, zeroline=False)
    fig.update_yaxes(title_text="Residuals", row=row, col=col, showline=True)


def _profiles_draw_align_lines(
    profiles, slabs, align, fig: "go.Figure", row: Optional[int] = None, col: Optional[int] = None
):
    for i, m in enumerate(profiles.keys()):
        t1_offset = _find_offset(slabs[m][0], align) if align != "auto" else None
        if t1_offset is not None:
            fig.add_vline(x=t1_offset, line=dict(dash="dash"), row=row, col=col)


def _plot_quantiles(
    x,
    y,
    contours,
    color,
    alpha,
    fig: "go.Figure",
    row: Optional[int] = None,
    col: Optional[int] = None,
    legendgroup: Optional[str] = None,
    secondary_y: bool = False,
):
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

    output = {}
    if alpha is None:
        alpha = 2.0 / (len(q) + 1)
    for contour, (lo, hi) in zip(contours, q):
        output[f"{contour} percent lower"] = lo
        output[f"{contour} percent upper"] = hi
        fig.add_scattergl(
            x=x,
            y=lo,
            showlegend=False,
            mode="lines",
            line=dict(color=color, width=1),
            opacity=alpha,
            row=row,
            col=col,
            hoverinfo="skip",
            legendgroup=legendgroup,
            secondary_y=secondary_y,
        )
        fig.add_scattergl(
            x=x,
            y=hi,
            showlegend=False,
            mode="lines",
            line=dict(color=color, width=1),
            fill="tonexty",
            opacity=alpha,
            row=row,
            col=col,
            hoverinfo="skip",
            legendgroup=legendgroup,
            secondary_y=secondary_y,
        )
    return output
