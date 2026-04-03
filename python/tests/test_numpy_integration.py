"""Tests for NumPy integration: callbacks receive ndarray, outputs return ndarray."""

import numpy as np
import pytest
import pyglobalsearch as gs
from numpy.typing import NDArray


def make_sphere_problem(*, record_input_types: bool = False):
    """2-D sphere f(x) = x0^2 + x1^2, minimum at origin"""
    received = []

    def objective(x):
        if record_input_types:
            received.append(type(x))
        return float(x[0] ** 2 + x[1] ** 2)

    def bounds():
        return np.array([[-5.0 + 1.0, 5.0 + 1.0], [-5.0 + 1.0, 5.0 + 1.0]])

    problem = gs.PyProblem(objective, bounds)
    return problem, received


def small_params():
    return gs.PyOQNLPParams(iterations=20, population_size=50, wait_cycle=5)


def test_objective_receives_ndarray():
    """Objective callback must receive a numpy ndarray, not a list"""
    problem, received = make_sphere_problem(record_input_types=True)
    gs.optimize(problem, small_params())
    assert received, "Objective was never called"
    assert all(t is np.ndarray for t in received), (
        f"Expected np.ndarray, got: {set(received)}"
    )


def test_gradient_receives_ndarray():
    """Gradient callback must receive a numpy ndarray"""
    received = []

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        received.append(type(x))
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def bounds():
        return np.array([[-3.0, 3.0], [-3.0, 3.0]])

    problem = gs.PyProblem(objective, bounds, gradient=gradient)
    gs.optimize(problem, small_params(), local_solver="LBFGS")

    assert received, "Gradient was never called"
    assert all(t is np.ndarray for t in received), (
        f"Expected np.ndarray, got: {set(received)}"
    )


def test_hessian_receives_ndarray():
    """Hessian callback must receive a numpy ndarray"""
    received = []

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def hessian(x):
        received.append(type(x))
        return np.array([[2.0, 0.0], [0.0, 2.0]])

    def bounds():
        return np.array([[-3.0, 3.0], [-3.0, 3.0]])

    problem = gs.PyProblem(objective, bounds, gradient=gradient, hessian=hessian)
    gs.optimize(problem, small_params(), local_solver="TrustRegion")

    assert received, "Hessian was never called"
    assert all(t is np.ndarray for t in received), (
        f"Expected np.ndarray, got: {set(received)}"
    )


def test_constraint_receives_ndarray():
    """Constraint callbacks must receive a numpy ndarray"""
    received = []

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def constraint(x):
        received.append(type(x))
        return float(x[0] + x[1])  # x0 + x1 >= 0

    problem = gs.PyProblem(objective, bounds, constraints=[constraint])
    gs.optimize(problem, small_params(), local_solver="COBYLA")

    assert received, "Constraint was never called"
    assert all(t is np.ndarray for t in received), (
        f"Expected np.ndarray, got: {set(received)}"
    )


def test_gradient_returning_ndarray_is_accepted():
    """Gradient that returns np.ndarray (not list) must be accepted."""

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x) -> NDArray[np.float64]:
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def bounds():
        return np.array([[-3.0, 3.0], [-3.0, 3.0]])

    problem = gs.PyProblem(objective, bounds, gradient=gradient)
    result = gs.optimize(problem, small_params(), local_solver="LBFGS")

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    assert abs(best.fun()) < 1e-4


def test_hessian_returning_ndarray_is_accepted():
    """Hessian that returns np.ndarray (not list-of-lists) must be accepted"""

    def objective(x):
        return float(x[0] ** 2 + x[1] ** 2)

    def gradient(x):
        return np.array([2.0 * x[0], 2.0 * x[1]])

    def hessian(x) -> NDArray[np.float64]:
        return np.array([[2.0, 0.0], [0.0, 2.0]])

    def bounds():
        return np.array([[-3.0, 3.0], [-3.0, 3.0]])

    problem = gs.PyProblem(objective, bounds, gradient=gradient, hessian=hessian)
    result = gs.optimize(problem, small_params(), local_solver="TrustRegion")

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    assert abs(best.fun()) < 1e-4


def test_as_array_returns_ndarray():
    """PyLocalSolution.as_array() must return a numpy ndarray."""
    problem, _ = make_sphere_problem()
    result = gs.optimize(problem, small_params())
    assert not result.is_empty()

    sol = result.best_solution()
    assert sol is not None
    arr = sol.as_array()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.dtype == np.float64


def test_as_array_matches_point():
    """as_array() values must equal the point attribute"""
    problem, _ = make_sphere_problem()
    result = gs.optimize(problem, small_params())
    sol = result.best_solution()
    assert sol is not None

    np.testing.assert_array_equal(sol.as_array(), np.array(sol.point))


def test_as_array_usable_with_numpy_ops():
    """as_array() result must be usable directly in numpy operations"""
    problem, _ = make_sphere_problem()
    result = gs.optimize(problem, small_params())
    sol = result.best_solution()
    assert sol is not None

    arr = sol.as_array()
    assert np.linalg.norm(arr) >= 0.0
    assert arr.sum() is not None


def test_bounds_returning_ndarray_is_accepted():
    """variable_bounds callback returning np.ndarray must be accepted"""

    def objective(x):
        return float((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)

    def bounds() -> NDArray[np.float64]:
        return np.array(
            [[-5.0 + 1.0, 5.0 + 1.0], [-5.0 + 1.0, 5.0 + 1.0]], dtype=np.float64
        )

    problem = gs.PyProblem(objective, bounds)
    result = gs.optimize(problem, small_params())

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    assert abs(best.x()[0] - 1.0) < 0.5
    assert abs(best.x()[1] - 1.0) < 0.5
