"""
Tests for different local solvers in pyglobalsearch.
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


def test_lbfgs_custom_config():
    """Test LBFGS with custom configuration using builders."""
    # Create custom LBFGS configuration
    custom_lbfgs = gs.builders.PyLBFGS(
        max_iter=50,
        tolerance_grad=1e-6,
        tolerance_cost=1e-12,
        history_size=5,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=custom_lbfgs
    )
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert abs(result[0].fun()) < 1e-6  # Should still find minimum


def test_lbfgs_tight_tolerance():
    """Test LBFGS with tight tolerances."""
    custom_lbfgs = gs.builders.PyLBFGS(
        max_iter=100,
        tolerance_grad=1e-8,
        history_size=8,
    )
    result = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=custom_lbfgs
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_nelder_mead_custom_config():
    """Test NelderMead with custom configuration."""
    custom_nelder_mead = gs.builders.PyNelderMead(
        max_iter=200,
        simplex_delta=0.05,
        tolerance_simplex=1e-8,
        tolerance_cost=1e-10,
    )
    result = gs.optimize(
        problem,
        params,
        local_solver="NelderMead",
        local_solver_config=custom_nelder_mead,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-4  # NelderMead may be less precise


def test_trustregion_custom_config():
    """Test TrustRegion with custom configuration."""
    custom_trustregion = gs.builders.PyTrustRegion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.steihaug(),
        max_iter=150,
        radius=0.5,
        max_radius=50.0,
        eta=0.1,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=custom_trustregion,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6


def test_trustregion_cauchy_method():
    """Test TrustRegion with Cauchy method."""
    custom_trustregion = gs.builders.PyTrustRegion(
        trust_region_radius_method=gs.builders.PyTrustRegionRadiusMethod.cauchy(),
        max_iter=100,
    )
    result = gs.optimize(
        problem_full,
        params,
        local_solver="TrustRegion",
        local_solver_config=custom_trustregion,
    )
    assert result is not None, "Optimization returned None"
    assert abs(result[0].fun()) < 1e-6
