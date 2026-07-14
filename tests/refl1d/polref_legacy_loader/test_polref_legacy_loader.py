import pytest
import numpy as np
from pathlib import Path
from refl1d.names import NeutronProbe, PolarizedNeutronProbe
from refl1d.probe.data_loaders.polref_legacy_data_loader import logstep, TOF_loader, load_probe_polref

def test_logstep():
    # Test basic geometric progression
    start = 0.01
    stop = 0.1
    step = 0.01
    # Expected: 0.01, 0.01*(1.01), 0.01*(1.01)^2 ...
    # Wait, the source code says: point = point + base**(np.log10(step*point)/np.log10(base))
    # which simplifies to point = point + step*point = point * (1+step)

    res = logstep(start, stop, step)
    assert res[0] == start
    assert np.isclose(res[1], start * (1 + step))
    assert res[-1] >= stop

    # Test with different base (though it should not affect the result based on the math)
    res_base = logstep(start, stop, step, base=2.0)
    assert np.allclose(res, res_base)

def test_TOF_loader_simulated():
    # Test loading without a filename (simulation mode)
    T = 0.25
    dQoQ = 0.02
    Q_sim_range = (0.005, 0.2)

    probe = TOF_loader(T=T, dQoQ=dQoQ, Q_sim_range=Q_sim_range)

    assert isinstance(probe, NeutronProbe)
    assert np.allclose(probe.T, T)
    assert np.allclose(probe.dT, T * dQoQ)
    # Check if Q is log-spaced as expected
    assert np.isclose(probe.Q[0], Q_sim_range[0])
    assert np.isclose(probe.Q[1], Q_sim_range[0] * (1 + dQoQ))
    # Check if L is calculated via QT2L (roughly)
    # L = 4pi*sin(T)/Q (in rad)
    expected_L_first = 4 * np.pi * np.sin(np.radians(T)) / probe.Q[0]
    assert np.isclose(probe.L[0], expected_L_first)

def test_TOF_loader_file(tmp_path):
    # Test loading from a temporary file
    d = tmp_path / "test_data.dat"
    # 3 columns: Q, R, dR
    data = np.array([
        [0.01, 1.0, 0.1],
        [0.02, 0.8, 0.08],
        [0.03, 0.6, 0.06]
    ])
    np.savetxt(d, data)

    T = 0.25
    dQoQ = 0.02
    probe = TOF_loader(T=T, dQoQ=dQoQ, filename=str(d), skiprows=0)

    assert isinstance(probe, NeutronProbe)
    assert np.allclose(probe.Q, data[:, 0])
    assert np.allclose(probe.R, data[:, 1]) # R
    assert np.allclose(probe.dR, data[:, 2]) # dR
    assert np.allclose(probe.dT, T * dQoQ)

def test_load_probe_polref_standard(tmp_path):
    # Test non-polarized loading
    d = tmp_path / "test_standard.dat"
    data = np.array([[0.01, 1.0, 0.1], [0.02, 0.8, 0.08]])
    np.savetxt(d, data)

    filename = "test_standard"
    probe = load_probe_polref(filename=filename, angle=0.25, dQoQ=0.02, path=str(tmp_path), pol_mode=None)

    assert isinstance(probe, NeutronProbe)
    assert probe.name == filename
    assert "intensity test_standard" in probe.intensity.name
    assert "inst" in probe.intensity.tags

def test_load_probe_polref_pnr(tmp_path):
    # Test PNR loading (_d.dat, _u.dat)
    d_file = tmp_path / "test_pnr_d.dat"
    u_file = tmp_path / "test_pnr_u.dat"
    data = np.array([[0.01, 1.0, 0.1], [0.02, 0.8, 0.08]])
    np.savetxt(d_file, data)
    np.savetxt(u_file, data)

    filename = "test_pnr"
    # Note: load_probe_polref appends .dat and suffixes
    # Source code: files = dict(data_mm=f"{filepath}_d.dat", data_pp=f"{filepath}_u.dat")
    # filepath = Path(path)/filename
    # So it looks for path/filename_d.dat

    probe = load_probe_polref(filename=filename, angle=0.25, dQoQ=0.02, path=str(tmp_path), pol_mode="pnr")

    assert isinstance(probe, PolarizedNeutronProbe)
    assert probe.mm is not None
    assert probe.pp is not None
    assert probe.mp is None
    assert probe.pm is None

    # Verify parameter sharing
    # lines 129-135: xs.intensity = probe.pp.intensity etc.
    assert probe.mm.intensity is probe.pp.intensity
    assert probe.mm.sample_broadening is probe.pp.sample_broadening
    assert probe.mm.theta_offset is probe.pp.theta_offset
    assert probe.mm.background is probe.pp.background

def test_load_probe_polref_pa(tmp_path):
    # Test PA loading (_dd, _du, _ud, _uu)
    files = ["_dd.dat", "_du.dat", "_ud.dat", "_uu.dat"]
    for f in files:
        data = np.array([[0.01, 1.0, 0.1], [0.02, 0.8, 0.08]])
        np.savetxt(tmp_path / f"test_pa{f}", data)

    filename = "test_pa"
    probe = load_probe_polref(filename=filename, angle=0.25, dQoQ=0.02, path=str(tmp_path), pol_mode="pa")

    assert isinstance(probe, PolarizedNeutronProbe)
    assert probe.mm is not None
    assert probe.mp is not None
    assert probe.pm is not None
    assert probe.pp is not None

    # Verify parameter sharing
    assert probe.mm.intensity is probe.pp.intensity
    assert probe.mp.intensity is probe.pp.intensity
    assert probe.pm.intensity is probe.pp.intensity
