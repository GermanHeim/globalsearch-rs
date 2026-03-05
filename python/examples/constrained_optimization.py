"""
Example: Constrained optimization using COBYLA solver

This example demonstrates how to use constraint functions with the COBYLA solver
to solve a constrained optimization problem.
"""

import numpy as np
import pyglobalsearch as gs


def objective(x):
    """Six-Hump Camel function to minimize."""
    x1, x2 = x[0], x[1]
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


def variable_bounds():
    """Variable bounds: x1 in [-3, 3], x2 in [-2, 2]."""
    return np.array([[-3.0, 3.0], [-2.0, 2.0]])


def constraint1(x):
    """Constraint: x1 + x2 >= -1 (reformulated as x1 + x2 + 1 >= 0)."""
    return x[0] + x[1] + 1.0


def constraint2(x):
    """Constraint: x1 - x2 >= -2 (reformulated as x1 - x2 + 2 >= 0)."""
    return x[0] - x[1] + 2.0


def main():
    # Create problem with constraints
    problem = gs.PyProblem(
        objective=objective,
        variable_bounds=variable_bounds,
        constraints=[constraint1, constraint2],  # List of constraint functions
    )

    # Configure COBYLA solver
    cobyla_config = gs.builders.cobyla(max_iter=100)

    # Set optimization parameters
    params = gs.PyOQNLPParams()
    params.iterations = 250
    params.population_size = 750
    params.wait_cycle = 15

    print("Solving constrained Six-Hump Camel function...")
    print("Constraints: x1 + x2 >= -1, x1 - x2 >= -2")

    # Run optimization
    result = gs.optimize(
        problem=problem,
        params=params,
        local_solver_config=cobyla_config,
        verbose=True,
    )

    # Display results
    best_sol = result.best_solution()
    if best_sol is not None:
        print("\nBest solution found:")
        print(f"  x = [{best_sol.point[0]:.6f}, {best_sol.point[1]:.6f}]")
        print(f"  f(x) = {best_sol.objective:.6f}")

        # Verify constraints
        x_best = np.array(best_sol.point)
        c1 = constraint1(x_best)
        c2 = constraint2(x_best)

        print("\nConstraint verification:")
        print(f"  Constraint 1: {c1:.6f} {'✓' if c1 >= 0 else '✗'}")
        print(f"  Constraint 2: {c2:.6f} {'✓' if c2 >= 0 else '✗'}")

        print(f"\nTotal solutions found: {len(result.solutions)}")
    else:
        print("No solution found!")


if __name__ == "__main__":
    main()
