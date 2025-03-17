from copy import deepcopy
from typing import Optional, Tuple
import re

from bumps.serialize import TYPE_KEY
from .. import __schema_version__ as CURRENT_SCHEMA_VERSION


def migrate(
    serialized: dict, from_version: Optional[str] = None, to_version: Optional[str] = CURRENT_SCHEMA_VERSION
) -> Tuple[str, dict]:
    """
    Migrate a serialized object from one version to another
    By default, the `from_version` is determined by inspection of the serialized object.
    This is overriden by setting the `from_version` keyword argument

    Also by default, the target version is the current schema, which can be overriden with
    the `to_version` keyword argument
    """

    if from_version is None:
        libraries = serialized.get("libraries", {})
        from_version = libraries.get("refl1d", {}).get("schema_version", "0")

    current_version = from_version
    while current_version != to_version:
        print(f"migrating refl1d from schema: {current_version}")
        current_version, serialized = MIGRATIONS[current_version](serialized)

    return current_version, serialized


def _migrate_0_to_1(serialized: dict):
    MAPPINGS = {
        "refl1d.abeles": "refl1d.probe.abeles",
        "refl1d.fresnel": "refl1d.probe.fresnel",
        "refl1d.instrument": "refl1d.probe.instrument",
        "refl1d.oversampling": "refl1d.probe.oversampling",
        "refl1d.probe": "refl1d.probe.probe",
        "refl1d.cheby": "refl1d.sample.cheby",
        "refl1d.model": "refl1d.sample.layers",
        "refl1d.material": "refl1d.sample.material",
        "refl1d.materialdb": "refl1d.sample.materialdb",
        "refl1d.magnetic": "refl1d.sample.magnetic",
        "refl1d.magnetism": "refl1d.sample.magnetism",
        "refl1d.reflectivity": "refl1d.sample.reflectivity",
        "refl1d.polymer": "refl1d.sample.polymer",
        "refl1d.mono": "refl1d.sample.mono",
        "refl1d.flayer": "refl1d.sample.flayer",
        "refl1d.util": "refl1d.utils.util",
        "refl1d.support": "refl1d.utils.support",
    }

    def remap(obj):
        if isinstance(obj, dict):
            if TYPE_KEY in obj:
                classname: str = obj[TYPE_KEY]
                first_dot_pair = ".".join(classname.split(".")[:2])
                if first_dot_pair in MAPPINGS:
                    obj[TYPE_KEY] = classname.replace(first_dot_pair, MAPPINGS[first_dot_pair], 1)
            for v in obj.values():
                remap(v)
        elif isinstance(obj, list):
            for v in obj:
                remap(v)

    output = deepcopy(serialized)
    remap(output)

    return "1", output


def _migrate_1_to_2(serialized: dict):
    """
    Replace refl1d.bumps_interface.fitproblem.FitProblem with bumps.fitproblem.FitProblem
    """

    def remap(obj):
        if isinstance(obj, dict):
            if TYPE_KEY in obj:
                classname: str = obj[TYPE_KEY]
                if classname == "refl1d.bumps_interface.fitproblem.FitProblem":
                    obj[TYPE_KEY] = "bumps.fitproblem.FitProblem"
            for v in obj.values():
                remap(v)
        elif isinstance(obj, list):
            for v in obj:
                remap(v)

    output = deepcopy(serialized)
    remap(output)
    return "2", output


MIGRATIONS = {
    "0": _migrate_0_to_1,
    "1": _migrate_1_to_2,
}
