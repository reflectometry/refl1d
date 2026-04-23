# refl1d.plotting.reflectivity
# Python version of generate_new_traces (without local_offset)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKER_OPACITY = 0.7
show_resolution = False  # Set True if you want error bars for dQ


def generate_new_traces_no_offset(model_data, view, calculate_residuals=False):
    theory_traces = []
    data_traces = []
    residuals_traces = []
    yaxis_label = "Reflectivity"
    xaxis_label = "Q (Å⁻¹)"

    if not model_data:
        return {
            "theory_traces": [],
            "data_traces": [],
            "xaxis_label": xaxis_label,
            "yaxis_label": yaxis_label,
        }

    if view == "Reflectivity":
        for plot_index, model in enumerate(model_data):
            for xs in model:
                label = f"{xs['label']} {xs['polarization']}"
                color = COLORS[plot_index % len(COLORS)]
                legendgroup = f"group_{plot_index}"
                y = xs["theory"]
                theory_traces.append(
                    {
                        "x": xs["Q"],
                        "y": y,
                        "mode": "lines",
                        "name": label + " theory",
                        "line": {"width": 2, "color": color},
                    }
                )
                if "R" in xs:
                    data_trace = {
                        "x": xs["Q"],
                        "y": xs["R"],
                        "mode": "markers",
                        "name": label + " data",
                        "marker": {"color": color},
                        "opacity": MARKER_OPACITY,
                        "legendgroup": legendgroup,
                    }
                    if show_resolution and "dQ" in xs:
                        data_trace["error_x"] = {"type": "data", "array": xs["dQ"], "visible": True}
                    if "dR" in xs:
                        data_trace["error_y"] = {"type": "data", "array": xs["dR"], "visible": True}
                        if calculate_residuals:
                            residuals = [(r - t) / dr for r, t, dr in zip(xs["R"], xs["theory"], xs["dR"])]
                            residuals_trace = {
                                "x": xs["Q"],
                                "y": residuals,
                                "mode": "markers",
                                "name": label + " residuals",
                                "showlegend": False,
                                "legendgroup": legendgroup,
                                "marker": {"color": color},
                                "opacity": MARKER_OPACITY,
                                "yaxis": "y2",
                            }
                            residuals_traces.append(residuals_trace)
                    data_traces.append(data_trace)
    elif view == "Fresnel (R/R_substrate)":
        yaxis_label = "Fresnel (R/R_substrate)"
        for plot_index, model in enumerate(model_data):
            for xs in model:
                label = f"{xs['label']} {xs['polarization']}"
                color = COLORS[plot_index % len(COLORS)]
                legendgroup = f"group_{plot_index}"
                theory = [y / f for y, f in zip(xs["theory"], xs["fresnel"])]
                theory_traces.append(
                    {
                        "x": xs["Q"],
                        "y": theory,
                        "mode": "lines",
                        "name": label + " theory",
                        "line": {"width": 2, "color": color},
                    }
                )
                if "R" in xs:
                    R = [y / f for y, f in zip(xs["R"], xs["fresnel"])]
                    data_trace = {
                        "x": xs["Q"],
                        "y": R,
                        "mode": "markers",
                        "name": label + " data",
                        "marker": {"color": color},
                        "opacity": MARKER_OPACITY,
                        "legendgroup": legendgroup,
                    }
                    if show_resolution and "dQ" in xs:
                        data_trace["error_x"] = {"type": "data", "array": xs["dQ"], "visible": True}
                    if "dR" in xs:
                        dR = [dy / f for dy, f in zip(xs["dR"], xs["fresnel"])]
                        data_trace["error_y"] = {"type": "data", "array": dR, "visible": True}
                        if calculate_residuals:
                            residuals = [(r - t) / dr for r, t, dr in zip(xs["R"], xs["theory"], xs["dR"])]
                            residuals_trace = {
                                "x": xs["Q"],
                                "y": residuals,
                                "mode": "markers",
                                "name": label + " residuals",
                                "showlegend": False,
                                "legendgroup": legendgroup,
                                "marker": {"color": color},
                                "opacity": MARKER_OPACITY,
                                "yaxis": "y2",
                            }
                            residuals_traces.append(residuals_trace)
                    data_traces.append(data_trace)
    elif view == "RQ^4":
        yaxis_label = "R · Q⁴"
        for plot_index, model in enumerate(model_data):
            for xs in model:
                label = f"{xs['label']} {xs['polarization']}"
                color = COLORS[plot_index % len(COLORS)]
                legendgroup = f"group_{plot_index}"
                intensity = xs.get("intensity_in", 1.0)
                background = xs.get("background_in", 0.0)
                Q4 = [1e-8 * qq**-4 * intensity + background for qq in xs["Q"]]
                theory = [t / q4 for t, q4 in zip(xs["theory"], Q4)]
                theory_traces.append(
                    {
                        "x": xs["Q"],
                        "y": theory,
                        "mode": "lines",
                        "name": label + " theory",
                        "line": {"width": 2, "color": color},
                    }
                )
                if "R" in xs:
                    R = [r / q4 for r, q4 in zip(xs["R"], Q4)]
                    data_trace = {
                        "x": xs["Q"],
                        "y": R,
                        "mode": "markers",
                        "name": label + " data",
                        "marker": {"color": color},
                        "opacity": MARKER_OPACITY,
                        "legendgroup": legendgroup,
                    }
                    if show_resolution and "dQ" in xs:
                        data_trace["error_x"] = {"type": "data", "array": xs["dQ"], "visible": True}
                    if "dR" in xs:
                        dR = [dy / q4 for dy, q4 in zip(xs["dR"], Q4)]
                        data_trace["error_y"] = {"type": "data", "array": dR, "visible": True}
                        if calculate_residuals:
                            residuals = [(r - t) / dr for r, t, dr in zip(xs["R"], xs["theory"], xs["dR"])]
                            residuals_trace = {
                                "x": xs["Q"],
                                "y": residuals,
                                "mode": "markers",
                                "name": label + " residuals",
                                "showlegend": False,
                                "legendgroup": legendgroup,
                                "marker": {"color": color},
                                "opacity": MARKER_OPACITY,
                                "yaxis": "y2",
                            }
                            residuals_traces.append(residuals_trace)
                    data_traces.append(data_trace)
    elif view == "Spin Asymmetry":
        yaxis_label = "Spin Asymmetry (pp - mm) / (pp + mm)"
        for plot_index, model in enumerate(model_data):
            pp = next((xs for xs in model if xs.get("polarization") == "++"), None)
            mm = next((xs for xs in model if xs.get("polarization") == "--"), None)
            if pp is not None and mm is not None:
                label = pp["label"]
                color = COLORS[plot_index % len(COLORS)]
                legendgroup = f"group_{plot_index}"

                # Interpolate mm.Q to pp.Q for theory and R
                def interp(x, xp, fp):
                    # Simple linear interpolation for 1D arrays
                    import numpy as np

                    return np.interp(x, xp, fp)

                Tm = interp(pp["Q"], mm["Q"], mm["theory"])
                TSA = [(p - m) / (p + m) for p, m in zip(pp["theory"], Tm)]
                theory_traces.append(
                    {
                        "x": pp["Q"],
                        "y": TSA,
                        "mode": "lines",
                        "name": label + " theory",
                        "line": {"width": 2, "color": color},
                    }
                )
                if "R" in pp and "R" in mm:
                    Rm = interp(pp["Q"], mm["Q"], mm["R"])
                    SA = [(p - m) / (p + m) for p, m in zip(pp["R"], Rm)]
                    data_trace = {
                        "x": pp["Q"],
                        "y": SA,
                        "mode": "markers",
                        "name": label + " data",
                        "marker": {"color": color},
                        "opacity": MARKER_OPACITY,
                        "legendgroup": legendgroup,
                    }
                    if show_resolution and "dQ" in pp:
                        data_trace["error_x"] = {"type": "data", "array": pp["dQ"], "visible": True}
                    if "dR" in pp and "dR" in mm:
                        dRm = interp(pp["Q"], mm["Q"], mm["dR"])
                        dSA = [
                            ((4 * ((p * dm) ** 2 + (m * dm) ** 2)) / (p + m) ** 4) ** 0.5
                            for p, m, dm in zip(pp["R"], Rm, dRm)
                        ]
                        data_trace["error_y"] = {"type": "data", "array": dSA, "visible": True}
                        if calculate_residuals:
                            residuals = [(v - t) / d for v, t, d in zip(SA, TSA, dSA)]
                            residuals_trace = {
                                "x": pp["Q"],
                                "y": residuals,
                                "mode": "markers",
                                "name": label + " residuals",
                                "showlegend": False,
                                "legendgroup": legendgroup,
                                "marker": {"color": color},
                                "opacity": MARKER_OPACITY,
                                "yaxis": "y2",
                            }
                            residuals_traces.append(residuals_trace)
                    data_traces.append(data_trace)
    data_traces.extend(residuals_traces)
    return {
        "theory_traces": theory_traces,
        "data_traces": data_traces,
        "xaxis_label": xaxis_label,
        "yaxis_label": yaxis_label,
    }
