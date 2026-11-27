"""
Regression tests for handling incomplete samples (zero or one layer).

The webview builder lets the user delete layers until the sample is empty so
they can rebuild from the ground up.  Previously an empty or one-layer sample
crashed ``Experiment.__init__`` or ``profile.Microslabs.finalize``, causing
deserialization to return ``None`` models and the server to repeatedly fail
while plotting.

These tests cover the fix for https://github.com/reflectometry/refl1d/issues/225
"""

import asyncio

import numpy as np
from bumps.fitproblem import FitProblem
from bumps.serialize import deserialize, serialize
from bumps.state import deserialize_problem, serialize_problem

from refl1d.names import SLD, Experiment, QProbe, Slab, Stack
from refl1d.webview.server import api as server_api


def _make_probe(q=None):
    """Build a simple QProbe with no measurement data."""
    if q is None:
        q = np.linspace(0.01, 0.1, 20)
    return QProbe(q, np.ones_like(q))


def test_empty_stack_serialization():
    """An empty Stack serializes and deserializes without becoming None."""
    stack = Stack()
    assert len(stack) == 0
    restored = deserialize(serialize(stack))
    assert restored is not None
    assert len(restored) == 0


def test_empty_stack_thickness_is_float():
    """The empty stack thickness calculation must return a float.

    Regression: ``sum([])`` is an int, which made the Calculation slot return a
    non-float from ``__float__`` and crashed FitProblem construction.
    """
    stack = Stack()
    assert isinstance(stack.thickness.value, float)
    assert stack.thickness.value == 0.0
    # The stack must still be usable as a FitProblem model.
    probe = _make_probe()
    problem = FitProblem(Experiment(stack, probe))
    assert len(list(problem.models)) == 1


def test_empty_experiment_is_incomplete():
    """An Experiment with no layers is incomplete and does not crash."""
    probe = _make_probe()
    experiment = Experiment(Stack(), probe)
    assert experiment.is_complete is False
    assert experiment.reflectivity() is None
    assert len(experiment.residuals()) == 0
    assert experiment.nllf() == 0.0
    assert experiment.smooth_profile() is None
    assert experiment.step_profile() is None
    assert experiment.slabs() is None
    assert experiment.ismagnetic is False


def test_one_layer_experiment_is_incomplete():
    """A one-layer sample is incomplete (needs substrate + surface)."""
    probe = _make_probe()
    sample = Stack([Slab(SLD(name="Si", rho=2.07, irho=0.0), thickness=100)])
    experiment = Experiment(sample, probe)
    assert experiment.is_complete is False
    assert experiment.reflectivity() is None
    assert len(experiment.residuals()) == 0
    assert experiment.nllf() == 0.0
    assert experiment.smooth_profile() is None
    assert experiment.slabs() is None


def test_two_layer_experiment_is_complete():
    """A normal two-layer sample still renders as before."""
    probe = _make_probe()
    sample = Stack(
        [
            Slab(SLD(name="Si", rho=2.07, irho=0.0), thickness=0),
            Slab(SLD(name="Vacuum", rho=0.0, irho=0.0), thickness=0),
        ]
    )
    experiment = Experiment(sample, probe)
    assert experiment.is_complete is True
    assert experiment.reflectivity() is not None
    assert experiment.smooth_profile() is not None
    assert experiment.slabs() is not None
    assert experiment.ismagnetic is False


def test_fitproblem_empty_model_roundtrip():
    """A FitProblem containing an empty model round-trips through dataclass serialization."""
    probe = _make_probe()
    problem = FitProblem(Experiment(Stack(), probe))
    serialized = serialize_problem(problem, method="dataclass")
    restored = deserialize_problem(serialized, method="dataclass")
    models = list(restored.models)
    assert len(models) == 1
    model = models[0]
    assert isinstance(model, Experiment)
    assert model.sample is not None
    assert len(model.sample) == 0
    assert model.is_complete is False
    assert model.reflectivity() is None
    assert len(model.residuals()) == 0
    assert restored.residuals().size == 0
    assert restored.nllf() == 0.0


def test_fitproblem_one_layer_roundtrip():
    """A FitProblem containing a one-layer model round-trips without crashing."""
    probe = _make_probe()
    sample = Stack([Slab(SLD(name="Si", rho=2.07, irho=0.0), thickness=100)])
    problem = FitProblem(Experiment(sample, probe))
    serialized = serialize_problem(problem, method="dataclass")
    restored = deserialize_problem(serialized, method="dataclass")
    model = next(iter(restored.models))
    assert model.is_complete is False
    assert model.reflectivity() is None
    assert len(model.residuals()) == 0


def test_get_plot_data_incomplete_returns_none():
    """get_plot_data returns None (clearing the plot) for an incomplete model."""
    probe = _make_probe()
    problem = FitProblem(Experiment(Stack(), probe))
    server_api.state.problem.fitProblem = problem
    try:
        result = asyncio.run(server_api.get_plot_data())
        assert result is None
    finally:
        server_api.state.problem.fitProblem = None


def test_get_plot_data_complete_returns_plotdata():
    """get_plot_data still returns plot data for a complete model."""
    probe = _make_probe()
    sample = Stack(
        [
            Slab(SLD(name="Si", rho=2.07, irho=0.0), thickness=0),
            Slab(SLD(name="Vacuum", rho=0.0, irho=0.0), thickness=0),
        ]
    )
    problem = FitProblem(Experiment(sample, probe))
    server_api.state.problem.fitProblem = problem
    try:
        result = asyncio.run(server_api.get_plot_data())
        assert result is not None
        assert "plotdata" in result
        assert len(result["plotdata"]) == 1
    finally:
        server_api.state.problem.fitProblem = None


def test_create_profile_plots_incomplete_returns_none():
    """create_profile_plots returns None when all selected models are incomplete."""
    probe = _make_probe()
    problem = FitProblem(Experiment(Stack(), probe))
    server_api.state.problem.fitProblem = problem
    try:
        result = asyncio.run(server_api.create_profile_plots([{"model_index": 0, "sample_index": 0}]))
        assert result is None
    finally:
        server_api.state.problem.fitProblem = None
