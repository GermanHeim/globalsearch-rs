"""
Basic Tutorial - Getting Started with PyGlobalSearch

This example provides a simple introduction to using the pyglobalsearch library
for global optimization. We'll optimize the Rosenbrock function.

The Rosenbrock function is defined as:
f(x, y) = (a - x)² + b(y - x²)²

With a = 1 and b = 100, the function has a global minimum at (1, 1) with f(1, 1) = 0.
This function is challenging because it has a narrow curved valley leading to the minimum.

This tutorial covers:
1. Defining objective functions and variable bounds
2. Setting up basic optimization parameters
3. Running the optimization with COBYLA solver
4. Interpreting and analyzing results

References:
H. H. Rosenbrock, An Automatic Method for Finding the Greatest or Least Value of a Function,
The Computer Journal, Volume 3, Issue 3, 1960, Pages 175-184, https://doi.org/10.1093/comjnl/3.3.175
"""

import numpy as np
import pyglobalsearch as gs
from numpy.typing import NDArray


class RosenbrockProblem:
    """
    The Rosenbrock function class

    The standard Rosenbrock function is defined as:
    f(x, y) = (a - x)² + b(y - x²)²

    With default parameters a=1, b=100, the global minimum is at (1, 1) with f(1, 1) = 0.
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        """
        Initialize the Rosenbrock function with parameters a and b.

        Args:
            a: First parameter (default: 1.0)
            b: Second parameter (default: 100.0)
        """
        self.a = a
        self.b = b

    def objective(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the Rosenbrock function: f(x, y) = (a - x)² + b(y - x²)²

        Args:
            x: Input vector [x, y]

        Returns:
            Function value at point x

        Raises:
            ValueError: If input doesn't have exactly 2 dimensions
        """
        # Validate input dimensions
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires exactly 2 variables")

        x_val, y_val = x[0], x[1]

        term1 = (self.a - x_val) ** 2
        term2 = self.b * (y_val - x_val**2) ** 2

        return term1 + term2

    def variable_bounds(self) -> NDArray[np.float64]:
        """
        Define the search bounds for the optimization variables.

        Returns:
            Array of bounds [[x_min, x_max], [y_min, y_max]]
        """
        return np.array(
            [
                [-5.0, 10.0],  # x bounds
                [-5.0, 10.0],  # y bounds
            ]
        )


def main():
    """Main function demonstrating Rosenbrock function optimization"""

    print("=" * 50)
    print("Optimizing the Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
    print("Global minimum: f(1, 1) = 0")
    print()

    # Step 1: Create the problem instance
    rosenbrock = RosenbrockProblem(a=1.0, b=100.0)

    # Step 2: Create the PyGlobalSearch problem
    problem = gs.PyProblem(
        objective=rosenbrock.objective,
        variable_bounds=rosenbrock.variable_bounds,
    )

    # Step 3: Configure local solver (COBYLA for derivative-free optimization)
    local_solver_config = gs.builders.cobyla(
        max_iter=500,  # Maximum iterations for COBYLA
    )

    # Step 4: Configure global optimization parameters
    params = gs.PyOQNLPParams(
        iterations=750,  # Number of global search iterations
        population_size=1500,  # Size of the scatter search population
        wait_cycle=15,  # Wait before updating search parameters
        threshold_factor=0.65,  # Merit filter sensitivity
        distance_factor=0.10,  # Minimum distance between solutions
    )

    print("Optimization parameters configured")
    print(f"  Stage two iterations: {params.iterations}")
    print(f"  Population size: {params.population_size}")
    print(f"  Wait cycle: {params.wait_cycle}")
    print(f"  Threshold factor: {params.threshold_factor}")
    print(f"  Distance factor: {params.distance_factor}")
    print()

    # Step 5: Run the optimization
    print("Starting optimization...")
    print("-" * 30)

    try:
        result = gs.optimize(
            problem=problem,
            params=params,
            local_solver_config=local_solver_config,
            seed=0,  # For reproducible results
            verbose=True,  # Show progress during optimization
        )

        print("-" * 30)
        print("Optimization completed successfully!")

        # Display solution set
        print(result)

    except Exception as e:
        print(f"Optimization failed: {e}")
        return


if __name__ == "__main__":
    main()
