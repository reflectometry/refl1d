from enum import StrEnum, auto


class Polarization(StrEnum):
    """Allowed neutron probe polarizations"""

    mm = auto()
    mp = auto()
    pm = auto()
    pp = auto()
