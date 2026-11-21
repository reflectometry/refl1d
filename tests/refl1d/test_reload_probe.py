import asyncio
from copy import deepcopy

from bumps import api as bumps_api
from bumps.fitproblem import FitProblem
from bumps.state import State
import numpy as np
import pytest

from refl1d.names import Experiment, SLD
from refl1d.probe.data_loaders.load4 import load4
from refl1d.webview.server import api


@pytest.fixture
def load_problem(monkeypatch):
    state = State()
    state.shared.selected_fitter = bumps_api.state.shared.selected_fitter
    state.shared.fitter_settings = deepcopy(bumps_api.state.shared.fitter_settings)
    state.shared.autosave_session = state.shared.autosave_history = False
    monkeypatch.setattr(bumps_api, "state", state)
    monkeypatch.setattr(api, "state", state)

    def reload_data(file, probes):
        sample = SLD("substrate", rho=2) | SLD("film", rho=4)(50, 3) | SLD("air", rho=0)
        experiments = [Experiment(sample=sample, probe=probe) for probe in probes]

        async def reload():
            await bumps_api.set_problem(FitProblem(experiments))
            await api.load_probe_from_file([str(file.parent)], file.name, fwhm=False)

        asyncio.run(reload())
        return [model.probe for model in state.problem.fitProblem.models]

    return reload_data


def _measurement(q, r):
    return f"{q} {r} 0.01 0.001\n{q + 0.01} {r / 2} 0.01 0.001\n"


def test_reload_preserves_shared_beam_parameters(load_problem, tmp_path):
    initial = tmp_path / "old.refl"
    replacement = tmp_path / "new.refl"
    initial.write_text(_measurement(0.01, 0.4))
    replacement.write_text(_measurement(0.02, 0.2))
    old, sibling = (load4(str(initial)) for _ in range(2))
    old.intensity.range(0.8, 1.2)
    old.intensity.value = 0.94
    old.background.range(1e-7, 3e-6)
    old.background.value = 2e-6
    sibling.intensity = old.intensity

    reloaded, kept = load_problem(replacement, [old, sibling])

    np.testing.assert_allclose(reloaded.Q, [0.02, 0.03])
    np.testing.assert_allclose(reloaded.R, [0.2, 0.1])
    assert kept is sibling
    assert reloaded.intensity is sibling.intensity
    assert reloaded.background is old.background
    assert reloaded.intensity.value == 0.94
    assert reloaded.intensity.bounds == (0.8, 1.2)
    assert not reloaded.intensity.fixed


def test_reload_matches_polarization_channels_by_label(load_problem, tmp_path):
    initial = tmp_path / "old.refl"
    replacement = tmp_path / "new.refl"
    initial.write_text(
        "# polarization: --\n" + _measurement(0.01, 0.4) + "\n\n# polarization: ++\n" + _measurement(0.02, 0.3)
    )
    replacement.write_text(
        "# polarization: -+\n" + _measurement(0.03, 0.2) + "\n\n# polarization: ++\n" + _measurement(0.04, 0.1)
    )
    old = load4(str(initial))
    old.pp.intensity.range(0.8, 1.2)
    old.mm.intensity.value = 0.5
    old.Aguide.value = 123

    (reloaded,) = load_problem(replacement, [old])

    assert reloaded.mm is None
    assert reloaded.mp.intensity.value == 1
    assert reloaded.pp.intensity is old.pp.intensity
    assert reloaded.Aguide is old.Aguide
    np.testing.assert_allclose(reloaded.pp.R, [0.1, 0.05])
