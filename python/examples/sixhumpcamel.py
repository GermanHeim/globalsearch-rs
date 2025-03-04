# Six-Hump Camel Back Function
# The Six-Hump Camel Back function is defined as follows:
#
# $f(x) = (4 - 2.1 x_1^2 + x_1^4 / 3) x_1^2 + x_1 x_2 + (-4 + 4 x_2^2) x_2^2$
#
# The function is defined on the domain $x_1 \in [-3, 3]$ and $x_2 \in [-2, 2]$.
# The function has two global minima at $f(0.0898, -0.7126) = -1.0316$ and $f(-0.0898, 0.7126) = -1.0316$.
# The function is continuous, differentiable and non-convex.
#
# References:
#
# Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf

import pyglobalsearch as gs
import numpy as np
from typing import Any

# Create the optimization parameters
params = gs.PyOQNLPParams(
    iterations=100,
    population_size=500,
    wait_cycle=10,
    threshold_factor=0.75,
    distance_factor=0.1,
)


# Function, variable bounds and gradient definitions
# Objective function
def obj(x: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return (
        4 * x[0] ** 2
        - 2.1 * x[0] ** 4
        + x[0] ** 6 / 3
        + x[0] * x[1]
        - 4 * x[1] ** 2
        + 4 * x[1] ** 4
    )


# Gradient
def grad(x: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray:
    return np.array(
        [
            8 * x[0] - 8.4 * x[0] ** 3 + x[0] ** 5 + x[1],
            x[0] - 8 * x[1] + 16 * x[1] ** 3,
        ]
    )


# Variable bounds
def variable_bounds() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array([[-3, 3], [-2, 2]])


# Create the problem
#
# The problem is defined by the objective function and variable bounds
# The gradient and hessian are optional and depend on the local solver
problem = gs.PyProblem(obj, variable_bounds, grad)


# Optimization returns a solution set
sol_set = gs.optimize(problem, params, local_solver="LBFGS", seed=0)

# Solution set is a list of dictionaries or None
# Each dictionary contains the solution and its function value
#
# Since we use MyPy type hints, we can access the solution and its function value
# without any runtime error
if sol_set is not None and len(sol_set) > 0:
    for sol in sol_set:
        print("Solution:")
        print(sol["x"])
        print(sol["fun"])
else:
    print("No solution found")
