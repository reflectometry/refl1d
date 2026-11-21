from collections import namedtuple
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest

from refl1d.uncertainty import show_errors


@pytest.fixture
def plot_inputs():
    existing = set(plt.get_fignums())
    model = namedtuple("Model", ["name", "index"])("film", 0)
    z = np.array([-10.0, 0.0, 20.0, 30.0])
    rho = np.array([2.1, 3.5, 3.5, 0.0])
    irho = np.full_like(z, 1e-4)
    profiles = {model: [(z, rho, irho), (z, rho * 1.05, irho), (z, rho * 0.95, irho)]}
    q = {model: np.array([0.02, 0.05, 0.1])}
    residuals = {model: np.array([[0.0, 0.2, -0.1], [0.1, 0.3, 0.0], [-0.1, 0.0, -0.2]])}
    try:
        yield profiles, {}, q, residuals
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


@pytest.mark.parametrize("explicit_figure", [False, True])
def test_save_uncertainty_uses_one_figure(plot_inputs, tmp_path, explicit_figure):
    selected = plt.figure() if explicit_figure else None
    active = plt.figure()
    fit_axes = active.add_subplot()
    fit_axes.plot([0, 1], [0, 1])
    basename = str(tmp_path / "film")

    show_errors(plot_inputs, align=None, npoints=20, save=basename, fig=selected)

    uncertainty = selected if explicit_figure else plt.gcf()
    assert uncertainty is not active
    assert active.axes == [fit_axes]
    assert len(uncertainty.axes) == 2
    assert uncertainty.axes[0].lines
    assert uncertainty.axes[1].get_ylabel() == "Residuals"
    if not explicit_figure:
        uncertainty.canvas.draw()
        label = uncertainty.axes[0].xaxis.label.get_window_extent()
        assert label.y0 >= uncertainty.axes[1].bbox.y1
    with BytesIO() as expected:
        uncertainty.savefig(expected, format="png")
        expected.seek(0)
        np.testing.assert_array_equal(plt.imread(basename + "-err.png"), plt.imread(expected))


def test_without_save_uses_current_figure(plot_inputs):
    active = plt.figure()

    show_errors(plot_inputs, align=None, npoints=20)

    assert plt.gcf() is active
    assert len(active.axes) == 2
