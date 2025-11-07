"""Tests for custom starting points (with_points) functionality."""

import pytest
import numpy as np
import pyglobalsearch as gs


def test_with_points_list_of_lists():
    """Test custom points with list of lists format."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def bounds():
        return np.array([[-10.0, 10.0], [-10.0, 10.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=50)

    # Custom points as list of lists
    custom_points = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    result = gs.optimize(problem, params, with_points=custom_points)

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    # Should find solution close to origin
    assert abs(best.x()[0]) < 1.0
    assert abs(best.x()[1]) < 1.0


def test_with_points_numpy_array():
    """Test custom points with numpy array format."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=50)

    # Custom points as numpy array
    custom_points = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [-0.5, -0.5, -0.5]])

    result = gs.optimize(problem, params, with_points=custom_points)

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    # Should find solution close to origin
    assert all(abs(x) < 1.0 for x in best.x())


def test_with_points_wrong_dimension():
    """Test that wrong dimension raises an error."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=100)

    # Points have wrong dimension (3 instead of 2)
    custom_points = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="dimension"):
        gs.optimize(problem, params, with_points=custom_points)


def test_with_points_out_of_bounds():
    """Test that out-of-bounds points raise an error."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=100)

    # Point is out of bounds (10.0 > 5.0)
    custom_points = np.array([[10.0, 2.0]])

    with pytest.raises(ValueError, match="outside variable bounds"):
        gs.optimize(problem, params, with_points=custom_points)


def test_with_points_empty_array():
    """Test that empty array is handled gracefully."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=50)

    # Empty array
    custom_points = np.array([]).reshape(0, 2)

    result = gs.optimize(problem, params, with_points=custom_points)

    # Should still work, just without custom points
    assert not result.is_empty()


def test_with_points_single_point():
    """Test with a single custom point."""

    def objective(x):
        return (x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2

    def bounds():
        return np.array([[-10.0, 10.0], [-10.0, 10.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=20, population_size=50)

    # Single point near optimum (3, -2)
    custom_points = np.array([[2.5, -1.5]])

    result = gs.optimize(problem, params, with_points=custom_points)

    assert not result.is_empty()
    best = result.best_solution()
    assert best is not None
    assert abs(best.x()[0] - 3.0) < 1.0
    assert abs(best.x()[1] + 2.0) < 1.0


def test_with_points_comparison():
    """Compare results with and without custom points."""

    def objective(x):
        return (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2

    def bounds():
        return np.array([[0.0, 10.0], [0.0, 10.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=100, population_size=200)

    # Without custom points
    result_without = gs.optimize(problem, params, seed=42)
    best_without = result_without.best_solution()

    # With custom points near optimum
    custom_points = np.array([[4.5, 4.5], [5.5, 5.5]])
    result_with = gs.optimize(problem, params, with_points=custom_points, seed=42)
    best_with = result_with.best_solution()

    # Both should find good solutions
    assert best_without is not None
    assert best_with is not None

    # Both should be close to optimum (5, 5)
    assert abs(best_without.x()[0] - 5.0) < 1.0
    assert abs(best_without.x()[1] - 5.0) < 1.0
    assert abs(best_with.x()[0] - 5.0) < 1.0
    assert abs(best_with.x()[1] - 5.0) < 1.0


def test_with_points_irregular_shape():
    """Test that irregular/jagged arrays are rejected."""

    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def bounds():
        return np.array([[-5.0, 5.0], [-5.0, 5.0]])

    problem = gs.PyProblem(objective, bounds)
    params = gs.PyOQNLPParams(iterations=50, population_size=20)

    # Irregular list (different row lengths) should fail
    custom_points = [[1.0, 2.0], [3.0]]  # Second row has only 1 element

    with pytest.raises((ValueError, IndexError)):
        gs.optimize(problem, params, with_points=custom_points)
