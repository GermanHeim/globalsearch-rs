"""
Tests for error handling and validation in pyglobalsearch.
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


def test_gradient_not_implemented_lbfgs():
    """Test that LBFGS fails when the gradient is not implemented."""
    try:
        gs.optimize(problem, params, local_solver="LBFGS")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed: Local Solver Error: unknown failed to run: Gradient not implemented and needed for local solver."
        )


def test_gradient_not_implemented_steepestdescent():
    """Test that SteepestDescent fails when the gradient is not implemented."""
    try:
        gs.optimize(problem, params, local_solver="SteepestDescent")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed: Local Solver Error: unknown failed to run: Gradient not implemented and needed for local solver."
        )


def test_hessian_not_implemented_newtoncg():
    """Test that NewtonCG fails when the hessian is not implemented."""
    try:
        gs.optimize(problem_grad, params, local_solver="NewtonCG")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed: Local Solver Error: unknown failed to run: Hessian not implemented and needed for local solver."
        )


def test_hessian_not_implemented_trustregion():
    """Test that TrustRegion fails when the hessian is not implemented."""
    try:
        gs.optimize(problem_grad, params, local_solver="TrustRegion")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed: Local Solver Error: unknown failed to run: Hessian not implemented and needed for local solver."
        )


def test_invalid_local_solver():
    """Test that we throw an error when the local solver is not implemented."""
    try:
        gs.optimize(problem, params, local_solver="InvalidLocalSolver")
    except Exception as e:
        assert str(e) == "Invalid solver type."


def test_trustregion_requires_hessian():
    """Test TrustRegion with problem_grad (should fail, needs Hessian)."""
    try:
        gs.optimize(problem_grad, params, local_solver="TrustRegion")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed: Local Solver Error: unknown failed to run: Hessian not implemented and needed for local solver."
        )
