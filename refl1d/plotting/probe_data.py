from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING
import numpy as np

from ..probe.polarization import Polarization

if TYPE_CHECKING:
    from ..probe.probe import BaseProbe, PolarizedNeutronProbe, ProbeSet, PolarizedQProbe


@dataclass
class ProbeData:
    """
    Container for everything that `BaseProbe` would have written.
    All public fields are NumPy ndarrays; any additional information can be
    stored in the free‑form `meta` mapping.
    """

    Q: np.ndarray  # 1‑D (or 2‑D) momentum transfer
    dQ: np.ndarray | None = None  # Uncertainty on Q
    R: np.ndarray | None = None  # Measured reflectivity
    dR: np.ndarray | None = None  # Uncertainty on R
    theory: np.ndarray | None = None  # Calculated reflectivity (same shape as Q)
    fresnel: np.ndarray | None = None  # Fresnel‑normalized reflectivity

    # Simple scalar metadata that used to be written as header lines
    intensity: float = 1.0
    background: float = 0.0
    name: str | None = None
    label: str | None = None
    polarization: Optional[Polarization] = None

    # Free‑form catch‑all for anything else (e.g. substrate name)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def save(self, filename: str) -> None:
        """
        Save the data to a file.
        """


def get_single_probe_data(theory, probe: "BaseProbe", substrate=None, surface=None, polarization=""):
    fresnel_calculator = probe.fresnel(substrate, surface)
    direction_multiplier = -1.0 if probe.back_reflectivity else 1.0
    calc_Q = probe.calc_Q
    Q, FQ = probe.apply_beam(calc_Q, fresnel_calculator(calc_Q * direction_multiplier))
    Q, R = theory

    assert isinstance(FQ, np.ndarray)
    if len(Q) != len(probe.Q):
        # Saving interpolated data
        output = dict(Q=Q, theory=R, fresnel=np.interp(Q, probe.Q, FQ))
    elif getattr(probe, "R", None) is not None:
        output = dict(
            Q=probe.Q,
            dQ=probe.dQ,
            R=probe.R,
            dR=probe.dR,
            theory=R,
            fresnel=FQ,
        )
    else:
        output = dict(Q=probe.Q, dQ=probe.dQ, theory=R, fresnel=FQ)
    output["background"] = probe.background.value
    output["intensity"] = probe.intensity.value
    output["polarization"] = polarization
    output["label"] = probe.label()
    return output


def get_probe_data(theory, probe, substrate=None, surface=None):
    if isinstance(probe, PolarizedNeutronProbe):
        output = []
        for xsi, xsi_th, suffix in zip(probe.xs, theory, ("--", "-+", "+-", "++")):
            if xsi is not None:
                output.append(get_single_probe_data(xsi_th, xsi, substrate, surface, suffix))
        return output
    elif isinstance(probe, ProbeSet):
        return [get_single_probe_data(t, p, substrate, surface) for p, t in probe.parts(theory)]
    else:
        return [get_single_probe_data(theory, probe, substrate, surface)]
