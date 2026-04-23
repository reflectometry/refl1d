"""Plotting utilities using Plotly, mirroring the matplotlib API.
Exports the same public functions as `matplotlib.py` and returns a
`plotly.graph_objects.Figure` from `plot(...)`.
"""  # noqa: D401

from __future__ import annotations

from typing import List, Optional

import numpy as np
from plotly import graph_objects as go

from .probe_data import ProbeData
# from bumps.plotutil import auto_shift, coordinated_colors

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKER_OPACITY = 0.5


def save(data: ProbeData, filename: str) -> None:
    """Save the arrays present in the ProbeData container to a file.

    This function is identical to the matplotlib version and is kept for
    compatibility with existing code.
    """
    cols: List[np.ndarray] = [data.Q]
    headers = ["Q (1/A)"]

    if data.dQ is not None:
        cols.append(data.dQ)
        headers.append("dQ (1/A)")
    if data.R is not None:
        cols.append(data.R)
        headers.append("R")
    if data.dR is not None:
        cols.append(data.dR)
        headers.append("dR")
    if data.theory is not None:
        cols.append(data.theory)
        headers.append("theory")
    if data.fresnel is not None:
        cols.append(data.fresnel)
        headers.append("fresnel")

    header_str = " ".join([f"{h:>17}" if i == 0 else f"{h:>20}" for i, h in enumerate(headers)])
    header_text = f"# intensity: {data.intensity:.15g}\n# background: {data.background:.15g}\n# {header_str}\n"

    A = np.array(cols)
    with open(filename, "wb") as fid:
        fid.write(header_text.encode())
        np.savetxt(fid, A.T, fmt="%20.15g")


def plot(data: ProbeData | List[ProbeData], view: str = "log", **kwargs) -> go.Figure:
    """Create a Plotly figure for one or more ``ProbeData`` objects.

    The public API matches ``matplotlib.plot`` – the caller supplies a view
    string (``"linear"``, ``"log"``, ``"fresnel"`` …) and any additional
    keyword arguments are forwarded to the view‑specific functions.

    The function returns a ``plotly.graph_objects.Figure`` instance.
    """
    print("I'm plotting from plotly!")
    fig: go.Figure
    if isinstance(data, list):
        # Combine figures for multiple datasets by merging their traces.
        fig = go.Figure()
        for i, d in enumerate(data):
            sub = plot(d, view, plot_index=i, **kwargs)
            for trace in sub.data:
                fig.add_trace(trace)

    elif view == "linear":
        fig = plot_linear(data, **kwargs)
    elif view == "log":
        fig = plot_log(data, **kwargs)
    elif view == "fresnel":
        fig = plot_fresnel(data, **kwargs)
    elif view == "logfresnel":
        fig = plot_logfresnel(data, **kwargs)
    elif view == "q4":
        fig = plot_Q4(data, **kwargs)
    elif view == "resolution":
        fig = plot_resolution(data, **kwargs)
    elif view.startswith("resid"):
        fig = plot_residuals(data, **kwargs)
    elif view == "fft":
        fig = plot_fft(data, **kwargs)
    elif view == "SA":
        # Placeholder – original library did nothing for this view.
        fig = go.Figure()
    else:
        raise TypeError(f"incorrect reflectivity view '{view}'")

    y_islog = view.startswith("log")

    fig.update_xaxes(
        title=dict(text="Q (Å<sup>-1</sup>)"),
        type="linear",
        autorange=True,
        showgrid=True,
    )
    fig.update_yaxes(
        title=dict(text=f"Reflectivity {view}"),
        exponentformat="power",
        showexponent="all",
        autorange=True,
        showgrid=True,
        type="log" if y_islog else "linear",
    )
    fig.update_layout(legend=dict(x=0.95, y=0.95, xanchor="left", yanchor="top"))
    fig.update_layout(template="simple_white")
    return fig


def _get_label(data: ProbeData, prefix: Optional[str] = None, gloss: str = "", suffix: str = "") -> Optional[str]:
    base = prefix if prefix else data.name
    if base:
        return " ".join((base + suffix, gloss)) if gloss else base
    return suffix + " " + gloss if gloss else None


def plot_resolution(data: ProbeData, suffix: str = "", label: Optional[str] = None, **kwargs) -> go.Figure:
    """Plot measurement resolution (dQ vs Q) using Plotly."""
    fig = go.Figure()
    fig.add_scatter(
        x=data.Q,
        y=data.dQ,
        mode="lines",
        name=_get_label(data, prefix=label, suffix=suffix),
    )
    fig.update_xaxes(title_text="Q ($\AA^{-1}$)")
    fig.update_yaxes(title_text="Q resolution (1-σ Å^{-1})")
    fig.update_layout(title="Measurement resolution")
    return fig


def plot_linear(data: ProbeData, **kwargs) -> go.Figure:
    fig = _plot_pair(data, ylabel="Reflectivity", **kwargs)
    fig.update_yaxes(type="linear")
    return fig


def plot_log(data: ProbeData, **kwargs) -> go.Figure:
    fig = _plot_pair(data, ylabel="Reflectivity", yaxis_scale="log", **kwargs)
    return fig


def plot_fresnel(data: ProbeData, **kwargs) -> go.Figure:
    if data.fresnel is None:
        raise ValueError("Fresnel-normalized reflectivity requires fresnel data arrays.")
    sub = data.meta.get("substrate_name", "sub")
    sur = data.meta.get("surface_name", "sur")
    name = sub if data.meta.get("surface_is_vacuum", False) else f"{sub}:{sur}" if sur != "sur" else sub
    ylabel = f"R/(R({name}))"
    return _plot_pair(data, ylabel=ylabel, scale_factor=data.fresnel, **kwargs)


def plot_logfresnel(data: ProbeData, **kwargs) -> go.Figure:
    fig = plot_fresnel(data, **kwargs)
    fig.update_yaxes(type="log")
    return fig


def plot_Q4(data: ProbeData, **kwargs) -> go.Figure:
    Q4 = 1e-8 * data.Q**-4 * data.intensity + data.background
    return _plot_pair(data, ylabel="R (100 Q)^4", scale_factor=Q4, **kwargs)


def _plot_pair(
    data: ProbeData,
    ylabel: str = "",
    yaxis_scale: str = "linear",
    scale_factor: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    suffix: str = "",
    plot_shift: float = 0,
    show_resolution: bool = True,
    plot_index: int = 0,
    **kwargs,
) -> go.Figure:
    """Add data and theory traces to a Plotly figure.

    Mirrors the behaviour of the original matplotlib implementation but builds
    a Plotly ``Figure`` instead of drawing directly.
    """
    fig = go.Figure()
    c = COLORS[plot_index % len(COLORS)]

    Q = data.Q
    dQ = data.dQ if show_resolution else None

    # Data points with error bars
    if data.R is not None:
        R = data.R
        dR = data.dR
        if scale_factor is not None:
            R = R / scale_factor
            if dR is not None:
                dR = dR / scale_factor
                # Prepare error bar specifications, handling None values safely
        error_x = dict(array=dQ, visible=True) if dQ is not None else None
        error_y = dict(array=dR, visible=True) if dR is not None else None
        fig.add_scatter(
            x=Q,
            y=R,
            mode="markers",
            error_x=error_x,
            error_y=error_y,
            marker=dict(color=c),
            opacity=MARKER_OPACITY,
            name=_get_label(data, prefix=label, gloss="data", suffix=suffix),
            **kwargs,
        )

    # Theory line
    if data.theory is not None:
        th = data.theory
        if scale_factor is not None:
            th = th / scale_factor
        fig.add_scatter(
            x=Q,
            y=th,
            mode="lines",
            line=dict(color=c),
            name=_get_label(data, prefix=label, gloss="theory", suffix=suffix),
            **kwargs,
        )

    fig.update_xaxes(title_text="Q (inv Angstroms)")
    fig.update_yaxes(title_text=ylabel)
    fig.update_yaxes(type=yaxis_scale)

    return fig


def plot_residuals(
    data: ProbeData, label: Optional[str] = None, suffix: str = "", residuals_shift: float = 0, plot_index=0, **kwargs
) -> go.Figure:
    if data.theory is None or data.R is None:
        return go.Figure()
    # trans = auto_shift(residuals_shift)
    c = COLORS[plot_index % len(COLORS)]
    residual = (data.theory - data.R) / data.dR
    fig = go.Figure()
    fig.add_scatter(
        x=data.Q,
        y=residual,
        mode="lines+markers",
        line=dict(color=c),
        name=_get_label(data, prefix=label, suffix=suffix),
        **kwargs,
    )
    # Horizontal reference lines at -1, 0, 1
    for y in (-1, 0, 1):
        fig.add_hline(y=y, line_dash="dash" if y != 0 else "solid", line_color="black")
    fig.update_xaxes(title_text="Q (inv A)")
    fig.update_yaxes(title_text="(theory-data)/error")
    return fig


def plot_fft(data: ProbeData, label: Optional[str] = None, suffix: str = "", plot_index=0, **kwargs) -> go.Figure:
    """FFT analysis of reflectivity signal using Plotly."""
    if data.fresnel is None:
        raise TypeError("FFT reflectivity needs fresnel array")
    c = COLORS[plot_index % len(COLORS)]
    Qmax = max(data.Q)
    T = np.linspace(0, Qmax, len(data.Q))
    z = np.linspace(0, 2 * np.pi / Qmax, len(data.Q) // 2)
    fig = go.Figure()
    if data.R is not None:
        signal = np.interp(T, data.Q, data.R / data.fresnel)
        A = np.abs(np.fft.fft(signal - np.average(signal)))[: len(signal) // 2]
        fig.add_scatter(
            x=z,
            y=A,
            mode="lines+markers",
            name=_get_label(data, prefix=label, gloss="data", suffix=suffix),
            line=dict(color=c),
            opacity=MARKER_OPACITY,
        )
    if data.theory is not None:
        signal = np.interp(T, data.Q, data.theory / data.fresnel)
        A = np.abs(np.fft.fft(signal - np.average(signal)))[: len(signal) // 2]
        fig.add_scatter(
            x=z,
            y=A,
            mode="lines",
            name=_get_label(data, prefix=label, gloss="theory", suffix=suffix),
            line=dict(color=c),
        )
    fig.update_xaxes(title_text="w (A)")
    sub = data.meta.get("substrate_name", "sub")
    sur = data.meta.get("surface_name", "sur")
    name = f"{sub}:{sur}" if sur != "sur" else sub
    fig.update_yaxes(title_text=f"|FFT(R/R({name}))|")
    return fig
