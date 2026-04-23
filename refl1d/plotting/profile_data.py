from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING
import numpy as np

from ..probe.polarization import Polarization


@dataclass
class ProfileData:
    """
    Container for profile data.
    """

    z: np.ndarray  # 1D depth
    rho: np.ndarray  # Nuclear scattering length density
    irho: np.ndarray  # Imaginary nuclear scattering length density
    rhoM: Optional[np.ndarray] = None  # Magnetic scattering length density
    thetaM: Optional[np.ndarray] = None  # Magnetic angle (degrees)
    meta: Mapping[str, Any] = field(default_factory=dict)  # Additional metadata

    def save(self, filename: str) -> None:
        """
        Save the profile data to a file.
        """
        # Implementation would go here
        pass
