from typing import Optional

from .profile_data import ProfileData
from .probe_data import ProbeData
from ..sample.reflectivity import BASE_GUIDE_ANGLE as DEFAULT_THETA_M


from bumps.plotutil import auto_shift, coordinated_colors


def save(data: ProbeData, filename: str):
    """
    Save the arrays present in the ProbeData container to a file.
    """
    cols = [data.Q]
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
        fid.write(asbytes(header_text))
        np.savetxt(fid, A.T, fmt="%20.15g")


def plot(
    data: ProbeData | list[ProbeData],
    profile_data: ProfileData,
    step_profile_data: ProfileData | None,
    plot_shift=0,
    profile_shift=0,
    view="log",
):
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plot_reflectivity(data, plot_shift=plot_shift, view=view)

    plt.subplot(212)
    plot_profile(profile_data, step_profile_data, plot_shift=profile_shift)


def plot_reflectivity(data: ProbeData | list[ProbeData], view="log", **kwargs):
    """
    Plot theory against data using the structured ProbeData.
    """
    if isinstance(data, list):
        for d in data:
            plot_reflectivity(d, view, **kwargs)
        return

    if view == "linear":
        plot_linear(data, **kwargs)
    elif view == "log":
        plot_log(data, **kwargs)
    elif view == "fresnel":
        plot_fresnel(data, **kwargs)
    elif view == "logfresnel":
        plot_logfresnel(data, **kwargs)
    elif view == "q4":
        plot_Q4(data, **kwargs)
    elif view == "resolution":
        plot_resolution(data, **kwargs)
    elif view.startswith("resid"):
        plot_residuals(data, **kwargs)
    elif view == "fft":
        plot_fft(data, **kwargs)
    elif view == "SA":
        pass
    else:
        raise TypeError(f"incorrect reflectivity view '{view}'")


def _get_label(data: ProbeData, prefix=None, gloss="", suffix=""):
    base = prefix if prefix else data.name
    if base:
        return " ".join((base + suffix, gloss)) if gloss else base
    return suffix + " " + gloss if gloss else None


def plot_resolution(data: ProbeData, suffix="", label=None, **kwargs):
    import matplotlib.pyplot as plt

    plt.plot(data.Q, data.dQ, label=_get_label(data, prefix=label, suffix=suffix))
    plt.xlabel(r"Q ($\AA^{-1}$)")
    plt.ylabel(r"Q resolution ($1-\sigma \AA^{-1}$)")
    plt.title("Measurement resolution")


def plot_linear(data: ProbeData, **kwargs):
    import matplotlib.pyplot as plt

    _plot_pair(data, ylabel="Reflectivity", **kwargs)
    plt.yscale("linear")


def plot_log(data: ProbeData, **kwargs):
    import matplotlib.pyplot as plt

    _plot_pair(data, ylabel="Reflectivity", **kwargs)
    plt.yscale("log")


def plot_logfresnel(data: ProbeData, **kwargs):
    import matplotlib.pyplot as plt

    plot_fresnel(data, **kwargs)
    plt.yscale("log")


def plot_fresnel(data: ProbeData, **kwargs):
    if data.fresnel is None:
        raise ValueError("Fresnel-normalized reflectivity requires fresnel data arrays.")

    sub = data.meta.get("substrate_name", "sub")
    sur = data.meta.get("surface_name", "sur")

    if data.meta.get("surface_is_vacuum", False):
        name = sub
    else:
        name = f"{sub}:{sur}" if sur != "sur" else sub

    _plot_pair(data, ylabel=f"R/(R({name}))", scale_factor=data.fresnel, **kwargs)


def plot_Q4(data: ProbeData, **kwargs):
    Q4 = 1e-8 * data.Q**-4 * data.intensity + data.background
    _plot_pair(data, ylabel="R (100 Q)^4", scale_factor=Q4, **kwargs)


def _plot_pair(
    data: ProbeData, ylabel="", scale_factor=None, label=None, suffix="", plot_shift=0, show_resolution=True, **kwargs
):
    import matplotlib.pyplot as plt

    c = coordinated_colors()
    trans = auto_shift(plot_shift)

    Q = data.Q
    dQ = data.dQ if show_resolution else None

    if data.R is not None:
        R = data.R
        dR = data.dR
        if scale_factor is not None:
            R = R / scale_factor
            if dR is not None:
                dR = dR / scale_factor

        plt.errorbar(
            Q,
            R,
            yerr=dR,
            xerr=dQ,
            capsize=0,
            fmt=".",
            color=c["light"],
            transform=trans,
            label=_get_label(data, prefix=label, gloss="data", suffix=suffix),
        )

    if data.theory is not None:
        th = data.theory
        if scale_factor is not None:
            th = th / scale_factor

        plt.plot(
            Q,
            th,
            "-",
            color=c["dark"],
            transform=trans,
            label=_get_label(data, prefix=label, gloss="theory", suffix=suffix),
        )

    plt.xlabel("Q (inv Angstroms)")
    plt.ylabel(ylabel)
    h = plt.legend(fancybox=True, numpoints=1)
    h.get_frame().set_alpha(0.5)


def plot_residuals(data: ProbeData, label=None, suffix="", residuals_shift=0, **kwargs):
    import matplotlib.pyplot as plt

    if data.theory is None or data.R is None:
        return

    trans = auto_shift(residuals_shift)
    c = coordinated_colors()

    residual = (data.theory - data.R) / data.dR

    plt.plot(
        data.Q, residual, ".", color=c["light"], transform=trans, label=_get_label(data, prefix=label, suffix=suffix)
    )
    plt.axhline(1, color="black", ls="--", lw=1)
    plt.axhline(0, color="black", lw=1)
    plt.axhline(-1, color="black", ls="--", lw=1)
    plt.xlabel("Q (inv A)")
    plt.ylabel("(theory-data)/error")
    plt.legend(numpoints=1)


def plot_fft(data: ProbeData, label=None, suffix="", **kwargs):
    """
    FFT analysis of reflectivity signal using static arrays.
    """
    import matplotlib.pyplot as plt

    if data.fresnel is None:
        raise TypeError("FFT reflectivity needs fresnel array")

    c = coordinated_colors()
    Qmax = max(data.Q)
    T = np.linspace(0, Qmax, len(data.Q))
    z = np.linspace(0, 2 * pi / Qmax, len(data.Q) // 2)

    if data.R is not None:
        signal = np.interp(T, data.Q, data.R / data.fresnel)
        A = abs(np.fft.fft(signal - np.average(signal)))
        A = A[: len(A) // 2]
        plt.plot(z, A, ".-", color=c["light"], label=_get_label(data, prefix=label, gloss="data", suffix=suffix))

    if data.theory is not None:
        signal = np.interp(T, data.Q, data.theory / data.fresnel)
        A = abs(np.fft.fft(signal - np.average(signal)))
        A = A[: len(A) // 2]
        plt.plot(z, A, "-", color=c["dark"], label=_get_label(data, prefix=label, gloss="theory", suffix=suffix))

    plt.xlabel("w (A)")

    sub = data.meta.get("substrate_name", "sub")
    sur = data.meta.get("surface_name", "sur")
    name = f"{sub}:{sur}" if sur != "sur" else sub
    plt.ylabel(f"|FFT(R/R({name}))|")


def plot_profile(data: ProfileData, step_data: Optional[ProfileData] = None, plot_shift=None):
    import matplotlib.pyplot as plt

    trans = auto_shift(plot_shift)
    if step_data is not None:
        plt.plot(step_data.z, step_data.rho, ":g", transform=trans)
        plt.plot(step_data.z, step_data.irho, ":b", transform=trans)
        if step_data.rhoM is not None:
            plt.plot(step_data.z, step_data.rhoM, ":r", transform=trans)
        if step_data.thetaM is not None and (abs(step_data.thetaM - DEFAULT_THETA_M) > 1e-3).any():
            ax = plt.twinx()
            ax.plot(step_data.z, step_data.thetaM, ":k", transform=trans)
            plt.ylabel("magnetic angle (degrees)")

    handles = [
        plt.plot(data.z, data.rho, "-g", transform=trans)[0],
        plt.plot(data.z, data.irho, "-b", transform=trans)[0],
    ]
    if data.rhoM is not None:
        rhoM_plot = plt.plot(data.z, data.rhoM, "-r", transform=trans)[0]
        handles.append(rhoM_plot)

    if data.thetaM is not None and (abs(data.thetaM - DEFAULT_THETA_M) > 1e-3).any():
        ax = plt.twinx()
        h = ax.plot(data.z, data.thetaM, "-k", transform=trans)
        handles.append(h[0])
        plt.ylabel("magnetic angle (degrees)")

    labels = [h.get_label() for h in handles]
    plt.legend(handles=handles, labels=labels)
    plt.ylabel("SLD (10^6 / A**2)")
    plt.xlabel("depth (A)")
