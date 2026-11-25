import numpy as np
import pytest
from bumps.fitproblem import FitProblem
from matplotlib import pyplot as plt

from refl1d.names import Experiment, QProbe, SLD
from refl1d.uncertainty import align_profiles, calc_errors_v2, show_errors
from refl1d.webview.server.profile_uncertainty import show_errors as web_errors


@pytest.fixture
def errors():
    q = np.linspace(0.012, 0.23, 51)
    probe = QProbe(q, np.full_like(q, 0.001), R=np.exp(-q * 80), dR=np.full_like(q, 0.003))
    substrate, vacuum = SLD("substrate", rho=2.07), SLD("vacuum", rho=0)
    film = SLD("film", rho=5.3, irho=0.03)(41.3, 0.7)
    film.thickness.range(10, 80)
    model = Experiment(sample=substrate(0, 1.2) | film | vacuum, probe=probe, name="variable thickness")
    points = np.array([[value] for value in (24.2, 29.9, 40.01, 43.12, 52.7, 60.95)])
    return calc_errors_v2(FitProblem(model), points)


@pytest.mark.parametrize("align", [0, 0.5])
def test_profile_contours_keep_best_grid_in_plot_and_export(errors, tmp_path, align):
    aligned = align_profiles(*errors[:2], align)
    ((model, group),) = aligned.items()
    best_z, best_rho = group[0][:2]
    lower = min(profile[0][0] for profile in group)
    upper = max(profile[0][-1] for profile in group)
    assert lower < best_z[0] or upper > best_z[-1]

    save = str(tmp_path / "auto")
    fig = plt.figure()
    try:
        show_errors(errors, align=align, save=save, fig=fig)
        plotted = fig.axes[0].lines[-1].get_xdata()
    finally:
        plt.close(fig)
    table = np.loadtxt(save + "_rho_contour1.dat")
    np.testing.assert_array_equal(table[:, 0], plotted)
    assert np.isin(best_z, plotted).all()
    assert plotted[0] == lower
    assert plotted[-1] == upper
    assert np.all(np.diff(plotted) > 0)
    assert np.max(np.diff(plotted)) <= 0.1 + 1e-12
    np.testing.assert_array_equal(np.interp(best_z, plotted, table[:, 1]), best_rho)

    web = web_errors(errors, npoints=None, align=align, residuals=False)
    np.testing.assert_array_equal(web["contour_data"][model.name]["z"], plotted)


def test_explicit_count_retains_fixed_spacing(errors):
    align = 0.5
    aligned = align_profiles(*errors[:2], align)
    ((model, group),) = aligned.items()
    lower = min(profile[0][0] for profile in group)
    upper = max(profile[0][-1] for profile in group)
    web = web_errors(errors, npoints=17, align=align, residuals=False)
    np.testing.assert_array_equal(web["contour_data"][model.name]["z"], np.linspace(lower, upper, 17))
