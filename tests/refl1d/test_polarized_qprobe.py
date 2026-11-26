from pathlib import Path

import numpy as np
import pytest

from refl1d.probe.data_loaders.load4 import PolarizedQProbe, load4
from refl1d.probe.probe import QProbe


def test_load4_polarized_qprobe():
    """Test that load4 correctly loads polarized data and creates a PolarizedQProbe."""
    # Get the path to the example data file (located in the doc/examples/four_column directory)
    test_file = Path(__file__).parent.parent.parent / "doc" / "examples" / "four_column" / "refl.txt"

    # Load the probe
    probe = load4(str(test_file))

    # Verify it's a PolarizedQProbe
    assert isinstance(probe, PolarizedQProbe), f"Expected PolarizedQProbe, got {type(probe)}"

    # Verify the specific cross sections are pp and mm
    assert probe.pp is not None, "Expected pp cross section to be present"
    assert probe.mm is not None, "Expected mm cross section to be present"
    assert probe.pm is None, "Expected pm cross section to be None"
    assert probe.mp is None, "Expected mp cross section to be None"


def _cross_section(q, data, uncertainty):
    q = np.asarray(q, dtype=float)
    return QProbe(q, np.full_like(q, 0.001), R=np.asarray(data, dtype=float), dR=np.asarray(uncertainty, dtype=float))


def _save(probe, path):
    theory = tuple(None if xs is None else (xs.Q, np.ones_like(xs.Q)) for xs in probe.xs)
    probe.save(str(path), theory)
    return path.with_name(path.stem + "-SA.dat")


def test_save_polarized_qprobe_writes_measured_spin_asymmetry(tmp_path):
    pp = _cross_section([0.01, 0.02], [3, 2], [0.1, 0.2])
    mm = _cross_section([0.01, 0.02], [2, -2], [0.2, 0.3])
    probe = PolarizedQProbe(pp=pp, mm=mm)
    path = _save(probe, tmp_path / "scan-refl.dat")

    table = np.loadtxt(path)
    np.testing.assert_allclose(table[0], [0.01, 0.2, 0.05059644256269408])
    np.testing.assert_allclose(table[1], [0.02, np.nan, np.nan], equal_nan=True)
    assert (tmp_path / "scan-refl.datA").is_file()
    assert (tmp_path / "scan-refl.datD").is_file()


def test_save_polarized_qprobe_propagates_interpolation_uncertainty(tmp_path):
    pp = _cross_section([0.005, 0.01, 0.02, 0.03, 0.04], [3, 3, 3, 5, 5], [0.1] * 5)
    mm = _cross_section([0.03, 0.01], [4, 2], [0.4, 0.2])
    path = _save(PolarizedQProbe(pp=pp, mm=mm), tmp_path / "scan-refl.dat")

    table = np.loadtxt(path)
    np.testing.assert_allclose(table[:, 0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(table[:, 1], [0.2, 0, 1 / 9])
    np.testing.assert_allclose(
        table[:, 2], [0.05059644256269408, np.sqrt(0.1**2 + 0.1**2 + 0.2**2) / 6, np.hypot(0.8, 4) / 81]
    )


def test_save_polarized_qprobe_skips_spin_asymmetry_in_theory_only_export(tmp_path):
    pp = _cross_section([0.01, 0.02], [3, 2], [0.1, 0.2])
    mm = _cross_section([0.01, 0.02], [2, 1], [0.2, 0.3])
    theory = (np.array([0.01, 0.015, 0.02]), np.ones(3))
    probe = PolarizedQProbe(pp=pp, mm=mm)
    probe.save(str(tmp_path / "scan-refl-interp.dat"), (theory, None, None, theory))

    assert not (tmp_path / "scan-refl-interp-SA.dat").exists()
    assert (tmp_path / "scan-refl-interp.datA").is_file()
    assert (tmp_path / "scan-refl-interp.datD").is_file()


@pytest.mark.parametrize("case", ["missing", "unmeasured", "disjoint"])
def test_save_polarized_qprobe_requires_overlapping_measured_channels(tmp_path, case):
    pp = _cross_section([0.01, 0.02], [3, 2], [0.1, 0.2])
    if case == "missing":
        mm = None
    elif case == "unmeasured":
        mm = QProbe(pp.Q, pp.dQ)
    else:
        mm = _cross_section([0.03, 0.04], [2, 1], [0.2, 0.3])
    path = _save(PolarizedQProbe(pp=pp, mm=mm), tmp_path / "scan-refl.dat")

    assert not path.exists()
    assert (tmp_path / "scan-refl.datD").is_file()


if __name__ == "__main__":
    test_load4_polarized_qprobe()
