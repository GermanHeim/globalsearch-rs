"""SLSQP, Barrier, AugmentedLagrangian, and linear constraint blocks."""

import numpy as np
import pytest
import pyglobalsearch as gs


def params():
    return gs.PyOQNLPParams(iterations=10, population_size=30, wait_cycle=5)


def quadratic_with_gradient():
    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] + 1) ** 2

    def gradient(x):
        return np.array([2 * (x[0] - 2), 2 * (x[1] + 1)])

    return objective, gradient


def test_slsqp_box_constrained():
    objective, gradient = quadratic_with_gradient()
    problem = gs.PyProblem(
        objective,
        lambda: np.array([[0.0, 1.0], [0.0, 1.0]]),
        gradient=gradient,
    )
    result = gs.optimize(problem, params(), local_solver="slsqp")
    assert abs(result[0].fun() - 2.0) < 1e-6


def test_slsqp_nonlinear_inequality():
    objective, gradient = quadratic_with_gradient()
    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        gradient=gradient,
        constraints=[lambda x: 0.5 - x[0] - x[1]],
        constraint_jacobian=lambda x: np.array([[-1.0, -1.0]]),
    )
    config = gs.builders.slsqp()
    assert isinstance(config, gs.builders.PySLSQP)
    result = gs.optimize(problem, params(), local_solver_config=config)
    assert abs(result[0].fun() - 0.125) < 1e-3


def test_slsqp_linear_inequality():
    objective, gradient = quadratic_with_gradient()
    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        gradient=gradient,
        linear_inequalities=(np.array([[-1.0, -1.0]]), np.array([-1.5])),
        constraint_jacobian=lambda x: np.zeros((0, 2)),
    )
    # -x - y <= -1.5  <=>  x + y >= 1.5; optimum of (x-2)^2+(y+1)^2 there is (2.25, -0.75).
    result = gs.optimize(problem, params(), local_solver="SLSQP")
    assert abs(result[0].fun() - 0.125) < 1e-3


def test_barrier_linear_inequality():
    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

    def gradient(x):
        return np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])

    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        gradient=gradient,
        linear_inequalities=(np.array([[1.0, 1.0]]), np.array([1.0])),
    )
    config = gs.builders.barrier()
    assert isinstance(config, gs.builders.PyBarrier)
    assert config.mu0 == 1.0
    result = gs.optimize(problem, params(), local_solver_config=config)
    assert abs(result[0].fun() - 4.5) < 1e-2


def test_augmented_lagrangian_linear_equality():
    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return np.array([2 * x[0], 2 * x[1]])

    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        gradient=gradient,
        linear_equalities=(np.array([[1.0, 1.0]]), np.array([1.0])),
    )
    config = gs.builders.augmented_lagrangian()
    assert isinstance(config, gs.builders.PyAugmentedLagrangian)
    result = gs.optimize(problem, params(), local_solver_config=config)
    assert abs(result[0].fun() - 0.5) < 1e-2


def test_cobyla_folds_linear_inequality():
    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        linear_inequalities=(np.array([[1.0, 1.0]]), np.array([1.0])),
    )
    result = gs.optimize(problem, params(), local_solver="cobyla")
    assert abs(result[0].fun() - 4.5) < 0.2


def test_barrier_requires_linear_inequalities():
    objective, gradient = quadratic_with_gradient()
    problem = gs.PyProblem(
        objective,
        lambda: np.array([[0.0, 1.0], [0.0, 1.0]]),
        gradient=gradient,
    )
    with pytest.raises(ValueError, match="linear_inequalities"):
        gs.optimize(problem, params(), local_solver="barrier")


def test_classic_solver_rejects_linear_blocks():
    objective, gradient = quadratic_with_gradient()
    problem = gs.PyProblem(
        objective,
        lambda: np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        gradient=gradient,
        linear_inequalities=(np.array([[1.0, 1.0]]), np.array([1.0])),
    )
    with pytest.raises(ValueError, match="SLSQP"):
        gs.optimize(problem, params(), local_solver="lbfgs")


def test_solver_name_and_config_must_match():
    with pytest.raises(ValueError, match="does not match"):
        gs.optimize(
            gs.PyProblem(lambda x: float(x @ x), lambda: np.array([[-1.0, 1.0]])),
            params(),
            local_solver="SLSQP",
            local_solver_config=gs.builders.lbfgs(),
        )
