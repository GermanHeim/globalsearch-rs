"""
Tests for parameter configuration in pyglobalsearch.
"""

import pyglobalsearch as gs
import numpy as np
from numpy.typing import NDArray


# Function, variable bounds and gradient definitions
def obj(x: NDArray[np.float64]) -> float:
    return x[0] ** 2 + x[1] ** 2


def grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * x[0], 2 * x[1]])


def variable_bounds() -> NDArray[np.float64]:
    return np.array([[-3, 3], [-2, 2]])


# Create test problems
problem_grad = gs.PyProblem(obj, variable_bounds, grad)

# Default parameters for comparison
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=500,
    wait_cycle=5,
    threshold_factor=0.75,
    distance_factor=0.1,
)


def test_configure_params():
    """Test configuring PyOQNLPParams."""
    custom_params = gs.PyOQNLPParams(
        iterations=100,
        population_size=200,
        wait_cycle=3,
        threshold_factor=0.5,
        distance_factor=0.2,
    )
    result = gs.optimize(problem_grad, custom_params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_default_params():
    """Test default PyOQNLPParams."""
    default_params = gs.PyOQNLPParams()
    result = gs.optimize(problem_grad, default_params)  # Use default solver
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0].x() == [0.0, 0.0]
    assert result[0].fun() == 0.0


def test_default_seed():
    """Test optimization with default seed."""
    result1 = gs.optimize(
        problem_grad, params, local_solver="LBFGS"
    )  # No seed specified
    result2 = gs.optimize(
        problem_grad, params, local_solver="LBFGS", seed=0
    )  # Default seed
    assert result1 is not None and result2 is not None
    # Results should be identical with same seed
    assert result1[0].x() == result2[0].x()
    assert result1[0].fun() == result2[0].fun()


def test_different_seeds():
    """Test that different seeds can produce different intermediate behavior."""
    result1 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=42)
    result2 = gs.optimize(problem_grad, params, local_solver="LBFGS", seed=123)
    assert result1 is not None and result2 is not None
    # Both should find the global minimum but potentially through different paths
    assert abs(result1[0].fun()) < 1e-10
    assert abs(result2[0].fun()) < 1e-10


def test_convergence_tolerance():
    """Test that optimization respects different tolerance settings."""
    loose_lbfgs = gs.builders.PyLBFGS(
        max_iter=300,
        tolerance_grad=1e-4,  # Loose tolerance
        tolerance_cost=1e-8,
        history_size=10,
    )
    result_loose = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=loose_lbfgs
    )

    tight_lbfgs = gs.builders.PyLBFGS(
        max_iter=300,
        tolerance_grad=1e-10,  # Tight tolerance
        tolerance_cost=1e-12,
        history_size=10,
    )
    result_tight = gs.optimize(
        problem_grad, params, local_solver="LBFGS", local_solver_config=tight_lbfgs
    )

    assert result_loose is not None and result_tight is not None
    # Both should find the solution, potentially with different precision
    assert abs(result_loose[0].fun()) < 1e-3
    assert abs(result_tight[0].fun()) < 1e-8
