import numpy as np
from typing import Callable, List, Optional, TypedDict, Any

class Solution(TypedDict):
    """
    # Represents the result of an optimization process.

    Attributes:
        x: The optimal solution point as a list of float values
        fun: The objective function value at the solution
    """

    x: List[float]
    fun: float

# class OptimizeResult(TypedDict):
#     solutions: List[Solution]
#     best: Solution

class PyOQNLPParams:
    """
    # Parameters for the OQNLP global optimization algorithm.

    Controls the behavior of the optimizer including population size,
    number of iterations, wait cycle, threshold and distance factor
    and seed.
    """

    iterations: int
    population_size: int
    wait_cycle: int
    threshold_factor: float
    distance_factor: float

    def __init__(
        self,
        iterations: int = 300,
        population_size: int = 1000,
        wait_cycle: int = 15,
        threshold_factor: float = 0.2,
        distance_factor: float = 0.75,
    ) -> None:
        """
        # Initialize optimization parameters.

        Args:
            iterations: Maximum number of iterations to perform
            population_size: Size of the population for the global search
            wait_cycle: Number of iterations to wait before terminating if no improvement
            threshold_factor: Factor controlling the threshold for local searches
            distance_factor: Factor controlling the minimum distance between solutions
        """
        ...

class PyProblem:
    """
    # Defines an optimization problem to be solved.

    Contains the objective function, variable bounds, and optionally
    gradient and hessian functions, depending on the local solver used.

    The objective function, gradient and hessian should take a single
    argument, a numpy array of float values, and return a float, numpy array or
    numpy matrix, respectively.

    The variable bounds function should take no arguments and return a numpy
    array of float values representing the lower and upper bounds for each
    variable in the optimization problem.
    """

    objective: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float]
    variable_bounds: Callable[[], np.ndarray[Any, np.dtype[np.float64]]]
    gradient: Optional[Callable[[np.ndarray[Any, np.dtype[np.float64]]], np.ndarray]]
    hessian: Optional[
        Callable[
            [np.ndarray[Any, np.dtype[np.float64]]],
            np.ndarray[Any, np.dtype[np.float64]],
        ]
    ]

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        variable_bounds: Callable[[], np.ndarray],
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        # Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them.

        Args:
            objective: Function that computes the objective value to be minimized
            variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
            gradient: Optional function that computes the gradient of the objective
            hessian: Optional function that computes the Hessian of the objective
        """
        ...

def optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Optional[str] = "LBFGS",
    seed: Optional[int] = 0,
) -> Optional[List[Solution]]:
    """
    # Perform global optimization on the given problem.

    This function uses a hybrid global-local search algorithm to find the
    global minimum of the objective function defined in the problem.

    Args:
        problem: The optimization problem to solve
        params: Parameters controlling the optimization process
        local_solver: The algorithm to use for local optimization ("LBFGS" by default)
        seed: Seed for reproducibility (0 by default)

    Returns:
        A list of Solution objects containing multiple solutions found and their
        objective values or none if no solution is found
    """
    ...
