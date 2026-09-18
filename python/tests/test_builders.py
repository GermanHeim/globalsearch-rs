"""
Tests for the builders module in pyglobalsearch.
"""

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray


# Function, variable bounds and gradient definitions
def obj(x: NDArray[np.float64]) -> float:
    return x[0] ** 2 + x[1] ** 2


def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * x[0], 2 * x[1]])


def hess(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([[2, 0], [0, 2]])


def variable_bounds() -> NDArray[np.float64]:
    return np.array([[-3, 3], [-2, 2]])


# Create optimization parameters
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=500,
    wait_cycle=5,
    threshold_factor=0.75,
    distance_factor=0.1,
)

# Create test problems
problem = gs.PyProblem(obj, variable_bounds)  # No gradient or hessian
problem_grad = gs.PyProblem(obj, variable_bounds, grad)  # With gradient only
problem_full = gs.PyProblem(
    obj, variable_bounds, grad, hess
)  # With gradient and hessian


def test_builders_lbfgs():
    """Test LBFGS created using builders module."""
    lbfgs_config = gs.builders.lbfgs(
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_cost=1e-14,
        history_size=6,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_builders_nelder_mead():
    """Test NelderMead created using builders module."""
    nelder_mead_config = gs.builders.nelder_mead(
        max_iter=250,
        simplex_delta=0.08,
        tolerance_simplex=1e-10,
        tolerance_cost=1e-12,
    )
    result = gs.optimize(
        problem,
        params,
        local_solver="NelderMead",
        local_solver_config=nelder_mead_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-4


def test_builders_trustregion():
    """Test TrustRegion created using builders module."""
    trustregion_config = gs.builders.trust_region(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.steihaug(),
        max_iter=120,
        radius=0.8,
        max_radius=80.0,
        eta=0.15,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=trustregion_config,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_builders_lbfgs_tolerances():
    """Test LBFGS created using builders module with custom tolerances."""
    lbfgs_config = gs.builders.lbfgs(
        max_iter=100,
        tolerance_grad=1e-8,
        tolerance_cost=1e-12,
        history_size=10,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=lbfgs_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_builders_cobyla_default():
    """Test COBYLA created using builders module with default parameters."""
    cobyla_config = gs.builders.cobyla()
    result = gs.optimize(
        problem, params, local_solver="COBYLA", local_solver_config=cobyla_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-3


def test_builders_cobyla_custom():
    """Test COBYLA created using builders module with custom parameters."""
    cobyla_config = gs.builders.cobyla(
        max_iter=200,
        step_size=0.5,
        ftol_rel=1e-6,
        ftol_abs=1e-8,
        xtol_rel=1e-5,
        xtol_abs=[1e-7, 1e-8],
    )
    result = gs.optimize(
        problem, params, local_solver="COBYLA", local_solver_config=cobyla_config
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-3


def test_builders_cobyla_tolerances():
    """Test COBYLA builder parameter setting and retrieval."""
    # Test default parameters
    default_cobyla = gs.builders.cobyla()
    assert default_cobyla.max_iter == 300
    assert default_cobyla.step_size == 1.0
    assert default_cobyla.ftol_rel is None
    assert default_cobyla.ftol_abs is None
    assert default_cobyla.xtol_rel is None
    assert default_cobyla.xtol_abs is None

    # Test custom parameters
    custom_cobyla = gs.builders.cobyla(
        max_iter=500,
        step_size=0.25,
        ftol_rel=1e-8,
        ftol_abs=1e-10,
        xtol_rel=1e-6,
        xtol_abs=[1e-8, 1e-9],
    )
    assert custom_cobyla.max_iter == 500
    assert custom_cobyla.step_size == 0.25
    assert custom_cobyla.ftol_rel == 1e-8
    assert custom_cobyla.ftol_abs == 1e-10
    assert custom_cobyla.xtol_rel == 1e-6
    assert custom_cobyla.xtol_abs == [1e-8, 1e-9]


def test_builders_cobyla_partial_tolerances():
    """Test COBYLA builder with only some tolerance parameters set."""
    partial_cobyla = gs.builders.cobyla(
        max_iter=400,
        step_size=0.8,
        ftol_rel=1e-7,
        xtol_rel=1e-5,
    )
    assert partial_cobyla.max_iter == 400
    assert partial_cobyla.step_size == 0.8
    assert partial_cobyla.ftol_rel == 1e-7
    assert partial_cobyla.ftol_abs is None
    assert partial_cobyla.xtol_rel == 1e-5
    assert partial_cobyla.xtol_abs is None

    # Test that it still works for optimization
    result = gs.optimize(
        problem, params, local_solver="COBYLA", local_solver_config=partial_cobyla
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-3
