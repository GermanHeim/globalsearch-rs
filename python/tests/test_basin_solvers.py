"""Basin solver selection, configuration, and callback behavior."""

import numpy as np
import pytest
import pyglobalsearch as gs


SOLVERS = [
    ("basin_lbfgs", "PyBasinLBFGS", False),
    ("basin_gradient_descent", "PyBasinGradientDescent", False),
    ("basin_trust_region", "PyBasinTrustRegion", False),
    ("basin_nelder_mead", "PyBasinNelderMead", False),
    ("basin_lbfgsb", "PyBasinLBFGSB", True),
    ("basin_bounded_nelder_mead", "PyBasinBoundedNelderMead", True),
    ("basin_bobyqa", "PyBasinBOBYQA", True),
]


def problem(bounded=False, constraints=None):
    def objective(x):
        if bounded:
            assert np.all((0 <= x) & (x <= 1))
        return (x[0] - 2) ** 2 + (x[1] + 1) ** 2

    def gradient(x):
        if bounded:
            assert np.all((0 <= x) & (x <= 1))
        return np.array([2 * (x[0] - 2), 2 * (x[1] + 1)])

    return gs.PyProblem(
        objective,
        lambda: np.array([[0.0, 1.0], [0.0, 1.0]]),
        gradient,
        lambda x: 2.0 * np.eye(2),
        constraints=constraints,
    )


def params():
    return gs.PyOQNLPParams(iterations=10, population_size=30, wait_cycle=5)


@pytest.mark.parametrize("name,cls,bounded", SOLVERS)
@pytest.mark.parametrize("selection", ["name", "config", "both"])
def test_selection(name, cls, bounded, selection):
    config = getattr(gs.builders, name)(max_iter=2000)
    assert isinstance(config, getattr(gs.builders, cls))
    assert config.max_iter == 2000
    kwargs = {}
    if selection != "config":
        kwargs["local_solver"] = name.upper().replace("_", "-")
    if selection != "name":
        kwargs["local_solver_config"] = config
    result = gs.optimize(problem(bounded), params(), **kwargs)
    expected = 2.0 if bounded else 0.0
    assert abs(result[0].fun() - expected) < 1e-6


@pytest.mark.parametrize("name,cls,bounded", SOLVERS)
def test_rejects_nonlinear_constraints(name, cls, bounded):
    with pytest.raises(ValueError, match="COBYLA"):
        gs.optimize(
            problem(bounded, lambda x: 1.0),
            params(),
            local_solver=name,
        )


def test_mismatched_backend_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        gs.optimize(problem(), params(), local_solver="LBFGS",
                    local_solver_config=gs.builders.basin_lbfgs())


def test_invalid_settings_and_mutation():
    config = gs.builders.basin_lbfgs(tolerance_cost=None)
    config.history_size = 0
    with pytest.raises(ValueError, match="history_size"):
        gs.optimize(problem(), params(), local_solver_config=config)
    with pytest.raises(ValueError, match="tolerance_grad"):
        gs.optimize(problem(), params(), local_solver_config=gs.builders.basin_lbfgs(tolerance_grad=-1))


def test_default_and_optional_settings():
    config = gs.builders.PyBasinLBFGS()
    assert config.max_iter == 1000
    assert config.history_size == 10
    assert config.tolerance_grad == 1e-6
    assert config.tolerance_cost is None
    assert gs.builders.basin_lbfgs(tolerance_grad=None).tolerance_grad is None
    assert gs.builders.basin_lbfgs(tolerance_grad=0.0).tolerance_grad == 0.0
    assert gs.builders.basin_bobyqa().interpolation_points is None


def test_missing_gradient_and_callback_errors():
    p = gs.PyProblem(lambda x: float(x @ x), lambda: np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="[Gg]radient"):
        gs.optimize(p, params(), local_solver="basin_lbfgs")

    def broken(x):
        raise ValueError("basin callback failure")

    p = gs.PyProblem(broken, lambda: np.array([[-1.0, 1.0]]))
    with pytest.raises(ValueError, match="basin callback failure"):
        gs.optimize(p, params(), local_solver="basin_nelder_mead")


def test_basin_trust_region_cauchy():
    config = gs.builders.basin_trust_region(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.cauchy()
    )
    assert gs.optimize(problem(), params(), local_solver_config=config)[0].fun() < 1e-8
