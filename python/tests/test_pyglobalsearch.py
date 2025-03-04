import pyglobalsearch as gs
import numpy as np
from typing import Any

# Create the optimization parameters
params = gs.PyOQNLPParams(
    iterations=150,
    population_size=500,
    wait_cycle=5,
    threshold_factor=0.75,
    distance_factor=0.1,
)


# Function, variable bounds and gradient definitions
# Objective function
def obj(x: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return x[0] ** 2 + x[1] ** 2


# Gradient
def grad(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray:
    return np.array(
        [
            2 * x[0],
            2 * x[1],
        ]
    )


# Hessian
def hess(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray:
    return np.array([[2, 0], [0, 2]])


# Variable bounds
def variable_bounds() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([[-3, 3], [-2, 2]])


# Create a problem without a gradient or hessian
problem = gs.PyProblem(obj, variable_bounds)

# Create a problem with the gradient but not with a hessian
problem_grad = gs.PyProblem(obj, variable_bounds, grad)

# Create a problem with the gradient and hessian
problem_full = gs.PyProblem(obj, variable_bounds, grad, hess)


# Test that LBFGS fails when the gradient is not implemented
def test_gradient_not_implemented_lbfgs():
    try:
        gs.optimize(problem, params, local_solver="LBFGS")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Gradient not implemented and needed for local solver."
        )


# Test that SteepestDescent fails when the gradient is not implemented
def test_gradient_not_implemented_steepestdescent():
    try:
        gs.optimize(problem, params, local_solver="SteepestDescent")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Gradient not implemented and needed for local solver."
        )


# Test that NewtonCG fails when the hessian is not implemented
def test_hessian_not_implemented_newtoncg():
    try:
        gs.optimize(problem_grad, params, local_solver="NewtonCG")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Hessian not implemented and needed for local solver."
        )


# Test that TrustRegion fails when the hessian is not implemented
def test_hessian_not_implemented_trustregion():
    try:
        gs.optimize(problem_grad, params, local_solver="TrustRegion")
    except Exception as e:
        assert (
            str(e)
            == "OQNLP Error: Local solver failed to find a solution. Local Solver Error: Failed to run local solver. Hessian not implemented and needed for local solver."
        )


# Test that we throw an error when the local solver is not implemented
def test_invalid_local_solver():
    try:
        gs.optimize(problem, params, local_solver="InvalidLocalSolver")
    except Exception as e:
        assert str(e) == "Invalid solver type."


# Test successful optimization with LBFGS
def test_optimize_lbfgs():
    result = gs.optimize(problem_grad, params, local_solver="LBFGS")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0


## Test successful optimization with SteepestDescent
# This fails due to a bug in argmin:
# https://github.com/argmin-rs/argmin/issues/400
# It should be easy to solve by changing the local solver configuration
# But that is a current limitation of the Python bindings
# def test_optimize_steepestdescent():
#     assert result is not None, "Optimization returned None"
#     assert len(result) > 0, "Optimization returned empty result"
#     assert result[0]["x"] == [0.0, 0.0]
#     assert result[0]["fun"] == 0.0


## Test successful optimization with NewtonCG
# This fails due to a bug in argmin:
# https://github.com/argmin-rs/argmin/issues/400
# It should be easy to solve by changing the local solver configuration
# But that is a current limitation of the Python bindings
# def test_optimize_newtoncg():
#     assert result is not None, "Optimization returned None"
#     assert len(result) > 0, "Optimization returned empty result"
#     assert result[0]["x"] == [0.0, 0.0]
#     assert result[0]["fun"] == 0.0


# Test successful optimization with TrustRegion
def test_optimize_trustregion():
    result = gs.optimize(problem_full, params, local_solver="TrustRegion")
    assert result is not None, "Optimization returned None"
    assert len(result) > 0, "Optimization returned empty result"
    assert result[0]["x"] == [0.0, 0.0]
    assert result[0]["fun"] == 0.0
