import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Optional, TypedDict, Union, Type, Iterator, Protocol

# Protocol definitions for type checking

class ObjectiveFunctionProtocol(Protocol):
    """
    Protocol for objective functions.

    An objective function takes a parameter vector and returns a scalar value
    to be minimized.

    Example:
        >>> def objective(x: NDArray[np.float64]) -> float:
        ...     return x[0]**2 + x[1]**2
    """
    def __call__(self, x: NDArray[np.float64]) -> float: ...

class GradientFunctionProtocol(Protocol):
    """
    Protocol for gradient functions.

    A gradient function takes a parameter vector and returns the gradient
    (vector of partial derivatives) at that point.

    Example:
        >>> def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...     return np.array([2*x[0], 2*x[1]])
    """
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

class HessianFunctionProtocol(Protocol):
    """
    Protocol for Hessian functions.

    A Hessian function takes a parameter vector and returns the Hessian matrix
    (matrix of second-order partial derivatives) at that point.

    Example:
        >>> def hessian(x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...     return np.array([[2.0, 0.0], [0.0, 2.0]])
    """
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

class ConstraintFunctionProtocol(Protocol):
    """
    Protocol for constraint functions.

    A constraint function takes a parameter vector and returns a scalar value.
    The constraint is satisfied when the returned value is >= 0.

    Example:
        >>> def constraint(x: NDArray[np.float64]) -> float:
        ...     return x[0] + x[1] - 1.0  # x[0] + x[1] >= 1
    """
    def __call__(self, x: NDArray[np.float64]) -> float: ...

class BoundsFunctionProtocol(Protocol):
    """
    Protocol for variable bounds functions.

    A bounds function returns a 2D array of shape (n_variables, 2) where each row
    contains [lower_bound, upper_bound] for the corresponding variable.

    Example:
        >>> def bounds() -> NDArray[np.float64]:
        ...     return np.array([[-5.0, 5.0], [-5.0, 5.0]])
    """
    def __call__(self) -> NDArray[np.float64]: ...

class ProblemProtocol(Protocol):
    """
    Protocol defining the interface for optimization problems.

    This protocol specifies the complete interface that optimization problems
    should implement, including objective function, bounds, and optional
    gradient, Hessian, and constraints.

    Use this protocol for type hints when you need to accept any problem-like object.

    Example:
        >>> def run_optimization(problem: ProblemProtocol, params: PyOQNLPParams):
        ...     result = optimize(problem, params)
        ...     return result
    """

    objective: ObjectiveFunctionProtocol
    variable_bounds: Union[NDArray[np.float64], BoundsFunctionProtocol]
    gradient: Optional[GradientFunctionProtocol]
    hessian: Optional[HessianFunctionProtocol]
    constraints: Optional[List[ConstraintFunctionProtocol]]

class Solution(TypedDict):
    """
    Represents the result of an optimization process.

    This is a compatibility type that matches the format used by SciPy's optimization functions.
    Use `PyLocalSolution` for the more feature-rich solution representation.

    Example:
        >>> result = optimize(problem, params)
        >>> best = result.best_solution()
        >>> solution_dict = {"x": best.x(), "fun": best.fun()}

    .. attribute: `x`
        :type: List[float]

        Parameter values at the solution point

    .. attribute: `fun`
        :type: float

        Objective function value at the solution point
    """

    x: List[float]
    fun: float

class PyLocalSolution:
    """
    A local solution in the parameter space.

    This class represents a solution point found by the optimization algorithm,
    including both the parameter values and the corresponding objective function value.
    Multiple PyLocalSolution objects are typically returned in a PySolutionSet.

    The class provides SciPy-compatible methods (`x()` and `fun()`) alongside
    direct attribute access (`point` and `objective`).

    Example:
        >>> solution = PyLocalSolution([1.0, 2.0], 3.5)
        >>> print(f"Point: {solution.x()}, Value: {solution.fun()}")
        Point: [1.0, 2.0], Value: 3.5

    .. py:attribute: `point`
        :type: List[float]

        Parameter values at the solution point
    .. py:attribute: `objective`
        :type: float

        Objective function value at the solution point
    """

    point: List[float]
    objective: float

    def __init__(self, point: List[float], objective: float) -> None:
        """
        Initialize a local solution.

        :param point: The solution point in the parameter space
        :type point: List[float]
        :param objective: The objective function value at the solution point
        :type objective: float
        """
        ...

    def fun(self) -> float:
        """
        Returns the objective function value at the solution point.

        Same as `objective` field

        This method is similar to the `fun` method in `SciPy.optimize` result

        :return: The objective function value
        :rtype: float
        """
        ...

    def x(self) -> List[float]:
        """
        Returns the solution point as a list of float values.

        Same as `point` field

        This method is similar to the `x` method in `SciPy.optimize` result

        :return: The solution point in parameter space
        :rtype: List[float]
        """
        ...

    def as_array(self) -> NDArray[np.float64]:
        """
        Returns the solution point as a NumPy 1D array.

        :return: The solution coordinates as a ``numpy.ndarray`` of shape ``(n,)``
        :rtype: numpy.ndarray
        """
        ...

class PySolutionSet:
    """
    A set of local solutions.

    This class represents a set of local solutions in the parameter space
    including the solution points and their corresponding objective function values.

    The solutions are stored as a list of `PyLocalSolution` objects.

    The `PySolutionSet` class supports indexing, iteration, and provides methods
    to get the number of solutions and find the best solution.
    """

    solutions: List[PyLocalSolution]

    def __init__(self, solutions: List[PyLocalSolution]) -> None:
        """
        Initialize a solution set.

        :param solutions: List of PyLocalSolution objects
        :type solutions: List[PyLocalSolution]
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of solutions stored in the set.

        :return: Number of solutions
        :rtype: int
        """
        ...

    def is_empty(self) -> bool:
        """
        Returns true if the solution set contains no solutions.

        :return: True if the solution set is empty, False otherwise
        :rtype: bool
        """
        ...

    def best_solution(self) -> Optional[PyLocalSolution]:
        """
        Returns the best solution in the set based on the objective function value.

        If the set is empty, returns None.

        :return: The best PyLocalSolution or None if the set is empty
        :rtype: Optional[PyLocalSolution]
        """
        ...

    def __getitem__(self, index: int) -> PyLocalSolution:
        """
        Returns the solution at the given index.

        :param index: Index of the solution to retrieve
        :type index: int
        :return: The PyLocalSolution at the specified index
        :rtype: PyLocalSolution
        """
        ...

    def __iter__(self) -> Iterator[PyLocalSolution]:
        """
        Returns an iterator over the solutions in the set.

        :return: An iterator over PyLocalSolution objects
        :rtype: Iterator[PyLocalSolution]
        """
        ...

class PyOQNLPParams:
    """
    Parameters for the OQNLP global optimization algorithm.

    Controls the behavior of the optimizer including population size,
    number of iterations, wait cycle, threshold and distance factor
    and seed.

    :param iterations: Maximum number of iterations to perform (default 300)
    :type iterations: int
    :param population_size: Size of the population for the global search (default 1000)
    :type population_size: int
    :param wait_cycle: Number of iterations to wait before terminating if no improvement (default 15)
    :type wait_cycle: int
    :param threshold_factor: Factor controlling the threshold for local searches (default 0.2)
    :type threshold_factor: float
    :param distance_factor: Factor controlling the minimum distance between solutions (default 0.75)
    :type distance_factor: float
    :param abs_tol: Absolute tolerance for comparing objective values (default 1e-8)
    :type abs_tol: float
    :param rel_tol: Relative tolerance for comparing objective values (default 1e-6)
    :type rel_tol: float
    """

    iterations: int
    population_size: int
    wait_cycle: int
    threshold_factor: float
    distance_factor: float
    abs_tol: float
    rel_tol: float
    def __init__(
        self,
        iterations: int = 300,
        population_size: int = 1000,
        wait_cycle: int = 15,
        threshold_factor: float = 0.2,
        distance_factor: float = 0.75,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-6,
    ) -> None:
        """
        Initialize optimization parameters.

        :param iterations: Maximum number of iterations to perform (default 300)
        :type iterations: int
        :param population_size: Size of the population for the global search (default 1000)
        :type population_size: int
        :param wait_cycle: Number of iterations to wait before terminating if no improvement (default 15)
        :type wait_cycle: int
        :param threshold_factor: Factor controlling the threshold for local searches (default 0.2)
        :type threshold_factor: float
        :param distance_factor: Factor controlling the minimum distance between solutions (default 0.75)
        :type distance_factor: float
        :param abs_tol: Absolute tolerance for comparing objective values (default 1e-8)
        :type abs_tol: float
        :param rel_tol: Relative tolerance for comparing objective values (default 1e-6)
        :type rel_tol: float
        """
        ...

class PyProblem:
    """
    Defines an optimization problem to be solved.

    Contains the objective function, variable bounds, and optionally
    gradient, hessian, and constraint functions, depending on the local solver used.

    This class implements the :class:`ProblemProtocol` interface.

    **Function Signatures**

    All functions should accept numpy arrays and return appropriate types:

    - **objective**: ``(x: NDArray[np.float64]) -> float``
        Maps parameter vector to scalar objective value to minimize

    - **gradient**: ``(x: NDArray[np.float64]) -> NDArray[np.float64]``
        Returns gradient vector (partial derivatives) at point x

    - **hessian**: ``(x: NDArray[np.float64]) -> NDArray[np.float64]``
        Returns Hessian matrix (second derivatives) at point x as 2D array

    - **constraints**: ``List[(x: NDArray[np.float64]) -> float]``
        List of constraint functions where ``constraint(x) >= 0`` means satisfied

    - **linear_inequalities**: ``Tuple[NDArray[np.float64], NDArray[np.float64]]``
        ``(A, b)`` with ``A x <= b``; ``A`` has shape ``(m, n)``, ``b`` length ``m``

    - **linear_equalities**: ``Tuple[NDArray[np.float64], NDArray[np.float64]]``
        ``(A, b)`` with ``A x == b``; ``A`` has shape ``(m, n)``, ``b`` length ``m``

    - **nonlinear_equalities**: ``List[(x: NDArray[np.float64]) -> float]``
        List of equality functions where ``h(x) == 0`` means satisfied

    - **constraint_jacobian**: ``(x: NDArray[np.float64]) -> NDArray[np.float64]``
        Jacobian of the nonlinear blocks with shape ``(n_eq + n_ineq, n_vars)``;
        equality rows first, then inequality rows

    - **variable_bounds**: ``NDArray[np.float64]`` *or* ``() -> NDArray[np.float64]``
        Array of shape ``(n_vars, 2)`` with ``[lower, upper]`` bounds per variable,
        or a zero-argument callable returning such an array.

    **Solver Requirements**

    Different local solvers have different requirements:

    - **COBYLA**: Only objective and bounds required (derivative-free)
    - **NelderMead**: Only objective and bounds required (derivative-free)
    - **BoundedNelderMead**: Only objective and bounds required (derivative-free)
    - **BOBYQA**: Only objective and bounds required (derivative-free)
    - **LBFGS**: Requires objective, bounds, and gradient
    - **LBFGSB**: Requires objective, bounds, and gradient
    - **GradientDescent**: Requires objective, bounds, and gradient
    - **TrustRegion**: Requires objective, bounds, gradient, and Hessian
    - **SLSQP**: Requires objective, bounds, gradient, and constraint_jacobian when constrained
    - **Barrier**: Requires objective, bounds, gradient, and linear_inequalities
    - **AugmentedLagrangian**: Requires objective, bounds, gradient, and linear_equalities

    **Examples**

    Basic unconstrained problem::

        >>> def objective(x):
        ...     return x[0]**2 + x[1]**2
        >>> def bounds():
        ...     return np.array([[-5, 5], [-5, 5]])
        >>> problem = PyProblem(objective, bounds)

    Problem with gradient for gradient-based solvers::

        >>> def gradient(x):
        ...     return np.array([2*x[0], 2*x[1]])
        >>> problem = PyProblem(objective, bounds, gradient=gradient)

    Problem with Hessian for second-order solvers::

        >>> def hessian(x):
        ...     return np.array([[2.0, 0.0], [0.0, 2.0]])
        >>> problem = PyProblem(objective, bounds, gradient=gradient, hessian=hessian)

    Constrained problem (use with COBYLA or SLSQP)::

        >>> def constraint(x):
        ...     return x[0] + x[1] - 1  # Constraint: x[0] + x[1] >= 1
        >>> problem = PyProblem(objective, bounds, constraints=[constraint])

    Linearly constrained problem (use with SLSQP, Barrier, or AugmentedLagrangian)::

        >>> import numpy as np
        >>> A = np.array([[1.0, 1.0]])
        >>> b = np.array([1.0])  # x[0] + x[1] <= 1
        >>> problem = PyProblem(objective, bounds, linear_inequalities=(A, b))

    Multiple constraints::

        >>> def constraint1(x):
        ...     return x[0] + x[1] - 1
        >>> def constraint2(x):
        ...     return x[0] - x[1]
        >>> problem = PyProblem(objective, bounds, constraints=[constraint1, constraint2])

    **See Also**

    - :class:`ProblemProtocol`: Protocol interface for type checking
    - :class:`ObjectiveFunctionProtocol`: Type hint for objective functions
    - :class:`GradientFunctionProtocol`: Type hint for gradient functions
    - :class:`HessianFunctionProtocol`: Type hint for Hessian functions
    - :class:`ConstraintFunctionProtocol`: Type hint for constraint functions
    """

    objective: Callable[[NDArray[np.float64]], float]
    variable_bounds: Union[NDArray[np.float64], Callable[[], NDArray[np.float64]]]
    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]
    linear_inequalities: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]
    linear_equalities: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]
    nonlinear_equalities: Optional[List[Callable[[NDArray[np.float64]], float]]]
    constraint_jacobian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        variable_bounds: Union[NDArray[np.float64], Callable[[], NDArray[np.float64]]],
        gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        constraints: Optional[List[Callable[[NDArray[np.float64]], float]]] = None,
        linear_inequalities: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        linear_equalities: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
        nonlinear_equalities: Optional[List[Callable[[NDArray[np.float64]], float]]] = None,
        constraint_jacobian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    ) -> None:
        """
        Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them (see class docstring for solver requirements).

        The constraints are optional and should be provided as a list of constraint
        functions if the local solver supports constraints (e.g., COBYLA, SLSQP).

        :param objective: Function that computes the objective value to be minimized
        :type objective: Callable[[NDArray[np.float64]], float]
        :param variable_bounds: Bounds array of shape (n_vars, 2), or a zero-argument
            callable returning one.
        :type variable_bounds: NDArray[np.float64] or Callable[[], NDArray[np.float64]]
        :param gradient: Optional function that computes the gradient of the objective
        :type gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param hessian: Optional function that computes the Hessian of the objective
        :type hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param constraints: Optional list of constraint functions. Each constraint is satisfied when constraint(x) >= 0
        :type constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]

        :raises ValueError: If variable_bounds has wrong shape or the callable fails
        """
        ...

class PyLBFGS:
    """
    Configuration for the L-BFGS solver (Basin-backed, unconstrained).

    Examples
    --------
        >>> lbfgs_config = PyLBFGS(max_iter=500, history_size=20)
    """

    max_iter: int
    tolerance_grad: Optional[float]
    tolerance_cost: Optional[float]
    history_size: int
    def __init__(
        self,
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
        history_size: int = 10,
    ) -> None:
        """
        Initialize L-BFGS solver configuration.
        """
        ...

class PyGradientDescent:
    """
    Configuration for the gradient descent solver (Basin-backed, unconstrained).
    """

    max_iter: int
    tolerance_grad: Optional[float]
    tolerance_cost: Optional[float]
    def __init__(
        self,
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
    ) -> None:
        ...

class PyNelderMead:
    """
    Configuration for the Nelder-Mead simplex solver (Basin-backed, unconstrained).
    """

    max_iter: int
    simplex_delta: float
    tolerance_simplex: Optional[float]
    tolerance_cost: Optional[float]
    def __init__(
        self,
        max_iter: int = 1000,
        simplex_delta: float = 0.1,
        tolerance_simplex: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = 1e-8,
    ) -> None:
        ...

class PyTrustRegionRadiusMethod:
    """
    Trust region radius computation methods.

    This class provides factory methods for different approaches to computing
    the trust region radius in trust region optimization methods.

    Available methods:
        - Cauchy: Uses Cauchy point for trust region radius computation
        - Steihaug: Uses Steihaug's conjugate gradient approach

    Examples
    --------
        >>> cauchy_method = PyTrustRegionRadiusMethod.cauchy()
        >>> steihaug_method = PyTrustRegionRadiusMethod.steihaug()
    """
    @staticmethod
    def cauchy() -> "PyTrustRegionRadiusMethod": ...
    @staticmethod
    def steihaug() -> "PyTrustRegionRadiusMethod": ...

class PyTrustRegion:
    """
    Configuration for the trust region optimization solver (Basin-backed).
    """

    max_iter: int
    tolerance_grad: Optional[float]
    trust_region_radius_method: PyTrustRegionRadiusMethod
    radius: float
    max_radius: float
    eta: float
    def __init__(
        self,
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.steihaug(),
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> None:
        ...

class PyCOBYLA:
    """
    Configuration for the COBYLA (Constrained Optimization BY Linear Approximations) solver.

    This configuration uses Basin's derivative-free COBYLA implementation. It is particularly
    useful when gradients are unavailable or when dealing with noisy objective functions and
    inequality constraints.

    Examples
    --------
    Basic usage::

        >>> cobyla_config = PyCOBYLA(max_iter=500, step_size=0.1)

    With per-variable tolerances::

        >>> # Different tolerance for each variable
        >>> cobyla_config = PyCOBYLA(xtol_abs=[1e-6, 1e-8])

    Using builder pattern::

        >>> cobyla_config = gs.builders.cobyla(
        ...     max_iter=1000,
        ...     xtol_abs=[1e-8] * n_vars  # Same tolerance for all variables
        ... )

    **Attributes**

    max_iter
        Maximum number of objective evaluations
    step_size
        Initial step size for the algorithm
    ftol_rel
        Relative tolerance for function value convergence
    ftol_abs
        Absolute tolerance for function value convergence
    xtol_rel
        Relative tolerance for parameter convergence
    xtol_abs
        Per-variable absolute tolerances for parameter convergence
    """

    max_iter: int
    step_size: float
    ftol_rel: Optional[float]
    ftol_abs: Optional[float]
    xtol_rel: Optional[float]
    xtol_abs: Optional[List[float]]
    def __init__(
        self,
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[List[float]] = None,
    ) -> None: ...

class PyLBFGSB:
    """Box-constrained basin L-BFGS-B with the default More-Thuente line search."""
    max_iter: int
    tolerance_projected_grad: Optional[float]
    tolerance_cost: Optional[float]
    history_size: int

    def __init__(
        self,
        max_iter: int = 1000,
        tolerance_projected_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
        history_size: int = 10,
    ) -> None: ...

class PyBoundedNelderMead:
    """Box-constrained basin Nelder-Mead with projected trial vertices and standard coefficients."""
    max_iter: int
    simplex_delta: float
    tolerance_simplex: Optional[float]
    tolerance_cost: Optional[float]

    def __init__(
        self,
        max_iter: int = 1000,
        simplex_delta: float = 0.1,
        tolerance_simplex: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = 1e-8,
    ) -> None: ...

class PyBOBYQA:
    """Box-constrained basin BOBYQA. Basin reduces the radii automatically for narrow boxes."""
    max_iter: int
    initial_radius: float
    final_radius: float
    interpolation_points: Optional[int]

    def __init__(
        self,
        max_iter: int = 1000,
        initial_radius: float = 1.0,
        final_radius: float = 1e-6,
        interpolation_points: Optional[int] = None,
    ) -> None: ...

class PySLSQP:
    """Gradient-based SLSQP with box, linear, and nonlinear constraints."""
    max_iter: int
    accuracy: Optional[float]
    max_subproblem_iter: Optional[int]

    def __init__(
        self,
        max_iter: int = 1000,
        accuracy: Optional[float] = 1e-6,
        max_subproblem_iter: Optional[int] = None,
    ) -> None: ...

class PyBarrier:
    """Log-barrier method over a BFGS inner solver for linear inequalities ``A x <= b``."""
    max_iter: int
    mu0: float
    reduction: float
    duality_gap_tol: float
    inner_max_iter: int

    def __init__(
        self,
        max_iter: int = 100,
        mu0: float = 1.0,
        reduction: float = 10.0,
        duality_gap_tol: float = 1e-8,
        inner_max_iter: int = 50,
    ) -> None: ...

class PyAugmentedLagrangian:
    """Augmented-Lagrangian method over a BFGS inner solver for linear equalities ``A x = b``."""
    max_iter: int
    rho0: float
    rho_increase: float
    feasibility_decrease: float
    feasibility_tol: float
    inner_max_iter: int

    def __init__(
        self,
        max_iter: int = 100,
        rho0: float = 10.0,
        rho_increase: float = 10.0,
        feasibility_decrease: float = 0.25,
        feasibility_tol: float = 1e-8,
        inner_max_iter: int = 50,
    ) -> None: ...

class builders:
    PyLBFGS: type[PyLBFGS]

    @staticmethod
    def lbfgs(
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
        history_size: int = 10,
    ) -> PyLBFGS:
        """Unconstrained L-BFGS with the default More-Thuente line search."""
        ...

    PyGradientDescent: type[PyGradientDescent]

    @staticmethod
    def gradient_descent(
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
    ) -> PyGradientDescent:
        """Unconstrained gradient descent with the default More-Thuente line search and no momentum."""
        ...

    PyTrustRegion: type[PyTrustRegion]

    @staticmethod
    def trust_region(
        max_iter: int = 1000,
        tolerance_grad: Optional[float] = 1e-6,
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.steihaug(),
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> PyTrustRegion:
        """Unconstrained trust-region optimization using the supplied gradient and Hessian."""
        ...

    PyNelderMead: type[PyNelderMead]

    @staticmethod
    def nelder_mead(
        max_iter: int = 1000,
        simplex_delta: float = 0.1,
        tolerance_simplex: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = 1e-8,
    ) -> PyNelderMead:
        """Unconstrained Nelder-Mead with standard coefficients."""
        ...

    PyLBFGSB: type[PyLBFGSB]

    @staticmethod
    def lbfgsb(
        max_iter: int = 1000,
        tolerance_projected_grad: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = None,
        history_size: int = 10,
    ) -> PyLBFGSB:
        """Box-constrained L-BFGS-B with the default More-Thuente line search."""
        ...

    PyBoundedNelderMead: type[PyBoundedNelderMead]

    @staticmethod
    def bounded_nelder_mead(
        max_iter: int = 1000,
        simplex_delta: float = 0.1,
        tolerance_simplex: Optional[float] = 1e-6,
        tolerance_cost: Optional[float] = 1e-8,
    ) -> PyBoundedNelderMead:
        """Box-constrained Nelder-Mead with projected trial vertices and standard coefficients."""
        ...

    PyBOBYQA: type[PyBOBYQA]

    @staticmethod
    def bobyqa(
        max_iter: int = 1000,
        initial_radius: float = 1.0,
        final_radius: float = 1e-6,
        interpolation_points: Optional[int] = None,
    ) -> PyBOBYQA:
        """Box-constrained BOBYQA. Basin reduces the radii automatically for narrow boxes."""
        ...

    PySLSQP: type[PySLSQP]

    @staticmethod
    def slsqp(
        max_iter: int = 1000,
        accuracy: Optional[float] = 1e-6,
        max_subproblem_iter: Optional[int] = None,
    ) -> PySLSQP:
        """Gradient-based SLSQP with box, linear, and nonlinear constraints."""
        ...

    PyBarrier: type[PyBarrier]

    @staticmethod
    def barrier(
        max_iter: int = 100,
        mu0: float = 1.0,
        reduction: float = 10.0,
        duality_gap_tol: float = 1e-8,
        inner_max_iter: int = 50,
    ) -> PyBarrier:
        """Log-barrier method over a BFGS inner solver for linear inequalities ``A x <= b``."""
        ...

    PyAugmentedLagrangian: type[PyAugmentedLagrangian]

    @staticmethod
    def augmented_lagrangian(
        max_iter: int = 100,
        rho0: float = 10.0,
        rho_increase: float = 10.0,
        feasibility_decrease: float = 0.25,
        feasibility_tol: float = 1e-8,
        inner_max_iter: int = 50,
    ) -> PyAugmentedLagrangian:
        """Augmented-Lagrangian method over a BFGS inner solver for linear equalities ``A x = b``."""
        ...

    @staticmethod
    def cobyla(
        max_iter: int = 300,
        step_size: float = 1.0,
        ftol_rel: Optional[float] = None,
        ftol_abs: Optional[float] = None,
        xtol_rel: Optional[float] = None,
        xtol_abs: Optional[List[float]] = None,
    ) -> PyCOBYLA:
        """
        Create a COBYLA solver configuration.

        This builder function allows easy creation of a COBYLA configuration
        with custom tolerances and parameters.

        Examples
        --------
            >>> cobyla_config = gs.builders.cobyla(max_iter=500, step_size=0.5)

        :param max_iter: Maximum number of objective evaluations (default 300)
        :type max_iter: int
        :param step_size: Initial step size (default 1.0)
        :type step_size: float
        :param ftol_rel: Relative tolerance for function value convergence (optional)
        :type ftol_rel: float
        :param ftol_abs: Absolute tolerance for function value convergence (optional)
        :type ftol_abs: float
        :param xtol_rel: Relative tolerance for parameter convergence (optional)
        :type xtol_rel: float
        :param xtol_abs: Per-variable absolute tolerances for parameter convergence (optional)
        :type xtol_abs: List[float]
        :return: Configured COBYLA solver
        :rtype: PyCOBYLA


        """
        ...

    # Aliases to global class definitions
    PyLBFGS: Type[PyLBFGS]
    PyGradientDescent: Type[PyGradientDescent]
    PyNelderMead: Type[PyNelderMead]
    PyLBFGSB: Type[PyLBFGSB]
    PyBoundedNelderMead: Type[PyBoundedNelderMead]
    PyBOBYQA: Type[PyBOBYQA]
    PySLSQP: Type[PySLSQP]
    PyBarrier: Type[PyBarrier]
    PyAugmentedLagrangian: Type[PyAugmentedLagrangian]
    PyTrustRegionRadiusMethod: Type[PyTrustRegionRadiusMethod]
    PyTrustRegion: Type[PyTrustRegion]
    PyCOBYLA: Type[PyCOBYLA]

class PyObserverMode:
    """
    Observer mode determines which stages to track during optimization.

    This enum controls which phases of the OQNLP algorithm are monitored
    by the observer, allowing fine-grained control over tracking scope.
    """

    Stage1Only: "PyObserverMode"
    Stage2Only: "PyObserverMode"
    Both: "PyObserverMode"

class PyStage1State:
    """
    State tracker for Stage 1 of the OQNLP algorithm.

    Tracks comprehensive metrics during the scatter search phase that builds
    the initial reference set. This includes reference set construction,
    trial point generation, function evaluations, and substage progression.

    Access current Stage 1 state during optimization using observer.stage1().
    Access final Stage 1 statistics after completion using observer.stage1_final().
    """

    reference_set_size: int
    """Current number of solutions in the reference set."""

    best_objective: float
    """Best (lowest) objective function value found so far in Stage 1."""

    current_substage: str
    """String identifier for the current phase of Stage 1 execution."""

    total_time: Optional[float]
    """Total elapsed time since Stage 1 started (seconds)."""

    function_evaluations: int
    """Cumulative count of objective function evaluations during Stage 1."""

    trial_points_generated: int
    """Total number of trial points generated during intensification."""

    best_point: Optional[List[float]]
    """Coordinates of the best solution found so far in Stage 1, or None if no solution evaluated yet."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PyStage2State:
    """
    State tracker for Stage 2 of the OQNLP algorithm.

    Tracks comprehensive metrics during the iterative refinement phase that
    improves the solution set through merit filtering and local optimization.
    This phase focuses on intensifying search around high-quality regions.

    Access current Stage 2 state during optimization using observer.stage2().
    """

    best_objective: float
    """Best (lowest) objective function value found across all solutions."""

    solution_set_size: int
    """Current number of solutions maintained in the working solution set."""

    current_iteration: int
    """Current iteration number in Stage 2."""

    threshold_value: float
    """Current merit filter threshold value."""

    local_solver_calls: int
    """Total number of times local optimization algorithms have been invoked."""

    improved_local_calls: int
    """Number of local solver calls that successfully improved the solution set."""

    function_evaluations: int
    """Cumulative count of objective function evaluations during Stage 2."""

    unchanged_cycles: int
    """Number of consecutive iterations where the solution set has not improved."""

    total_time: Optional[float]
    """Time elapsed since Stage 2 began (seconds)."""

    best_point: Optional[List[float]]
    """Coordinates of the best solution found so far in Stage 2, or None if no solution evaluated yet."""

    last_added_point: Optional[List[float]]
    """Coordinates of the most recently added solution to the solution set, or None if no solution added yet."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class PyObserver:
    """
    Main observer struct that tracks OQNLP algorithm state.

    The observer can be configured to track different metrics during
    Stage 1 (reference set construction) and Stage 2 (iterative improvement).
    It supports real-time monitoring through callbacks and provides detailed
    statistics about algorithm performance and convergence.

    Examples
    --------
    Basic observer with default logging::

        >>> observer = gs.observers.Observer()
        >>> observer.with_default_callback()
        >>> result = gs.optimize(problem, params, observer=observer)

    Observer with custom configuration::

        >>> observer = gs.observers.Observer()
        >>> observer.with_stage1_tracking()
        >>> observer.with_stage2_tracking()
        >>> observer.with_timing()
        >>> observer.with_callback_frequency(5)  # Callback every 5 iterations
        >>> result = gs.optimize(problem, params, observer=observer)

    Accessing state during optimization::

        >>> # In a callback function
        >>> def my_callback(observer):
        ...     if observer.stage1():
        ...         print(f"Stage 1: {observer.stage1().best_objective}")
        ...     if observer.stage2():
        ...         print(f"Stage 2: {observer.stage2().current_iteration}")

    Accessing final statistics::

        >>> # After optimization completes
        >>> stage1_final = observer.stage1_final()
        >>> stage2_final = observer.stage2()
        >>> print(f"Total function evaluations: {stage1_final.function_evaluations + stage2_final.function_evaluations}")
    """

    @property
    def should_observe_stage1(self) -> bool:
        """
        True if Stage 1 tracking is enabled and mode allows Stage 1 observation.
        """
        ...

    @property
    def should_observe_stage2(self) -> bool:
        """
        True if Stage 2 tracking is enabled and mode allows Stage 2 observation.
        """
        ...

    @property
    def is_timing_enabled(self) -> bool:
        """
        True if the observer is configured to track timing information.
        """
        ...

    @property
    def elapsed_time(self) -> Optional[float]:
        """
        Time elapsed since timer started (seconds), or None if timing disabled.
        """
        ...

    def __init__(self) -> None:
        """
        Create a new observer with no tracking enabled.

        Returns a minimal observer that tracks nothing by default.
        Use the builder methods to enable specific tracking features.
        """
        ...

    def with_stage1_tracking(self) -> None:
        """
        Enable Stage 1 tracking.

        Enables tracking of scatter search metrics including reference set size,
        best objective values, function evaluation counts, trial point generation,
        and sub-stage progression.

        Stage 1 tracking is required for stage1() and stage1_final() to return data.
        """
        ...

    def with_stage2_tracking(self) -> None:
        """
        Enable Stage 2 tracking.

        Enables tracking of iterative refinement metrics including current iteration,
        solution set size, best objective values, local solver statistics,
        function evaluations, threshold values, and convergence metrics.

        Stage 2 tracking is required for stage2() to return data.
        """
        ...

    def with_timing(self) -> None:
        """
        Enable timing tracking for stages.

        When enabled, tracks elapsed time for total Stage 1 and Stage 2 duration.
        Timing data is accessible via the total_time properties on state objects.
        """
        ...

    def with_mode(self, mode: PyObserverMode) -> None:
        """
        Set observer mode.

        Controls which stages of the optimization algorithm are monitored.
        This allows fine-grained control over tracking scope and performance.

        :param mode: The observer mode determining which stages to track
        :type mode: PyObserverMode
        """
        ...

    def with_callback_frequency(self, frequency: int) -> None:
        """
        Set the frequency for callback invocation.

        Controls how often the callback is invoked during Stage 2.
        For example, a frequency of 10 means the callback is called every 10 iterations.

        :param frequency: Number of iterations between callback calls
        :type frequency: int
        """
        ...

    def with_callback(
        self,
        callback: Callable[[Optional[PyStage1State], Optional[PyStage2State]], None],
    ) -> None:
        """
        Set a custom callback function for monitoring optimization progress.

        The callback function will be called during optimization with the current
        stage states. This allows real-time monitoring and custom logging.

        The callback receives the current Stage 1 and Stage 2 states, which may be None
        if the corresponding stage is not active or tracking is disabled.

        Examples
        --------
            >>> def my_callback(stage1, stage2):
            ...     if stage1:
            ...         print(f"Stage 1: {stage1.function_evaluations} evaluations")
            ...     if stage2:
            ...         print(f"Stage 2: Iteration {stage2.current_iteration}")
            >>> observer.with_callback(my_callback)

        :param callback: Function to call during optimization progress
        :type callback: Callable[[Optional[PyStage1State], Optional[PyStage2State]], None]
        """
        ...

    def with_default_callback(self) -> None:
        """
        Use a default console logging callback for both Stage 1 and Stage 2.

        This is a convenience method that provides sensible default logging
        for both stages of the optimization. The default callback prints progress
        information to stderr.
        """
        ...

    def with_stage1_callback(self) -> None:
        """
        Use a default console logging callback for Stage 1 only.

        This prints updates during scatter search and local optimization in Stage 1.
        """
        ...

    def with_stage2_callback(self) -> None:
        """
        Use a default console logging callback for Stage 2 only.

        This prints iteration progress during Stage 2. Use with_callback_frequency()
        to control how often updates are printed.
        """
        ...

    def unique_updates(self) -> None:
        """
        Enable filtering of Stage 2 callback messages to only show unique updates.

        When enabled, Stage 2 callback messages will only be printed when
        there is an actual change in the optimization state (other than just
        the iteration number). This reduces log verbosity by filtering out
        identical consecutive messages.

        # Changes that trigger printing:
        - Best objective value changes
        - Solution set size changes
        - Threshold value changes
        - Local solver call counts change
        - Function evaluation counts change

        # Example

        ```python
        observer = PyObserver()
        observer.with_stage2_tracking()
        observer.with_default_callback()
        observer.unique_updates()  # Only print when state changes
        ```
        """
        ...

    def stage1(self) -> Optional[PyStage1State]:
        """
        Get current Stage 1 state reference.

        Returns the current Stage 1 state if Stage 1 tracking is enabled and
        Stage 1 is still active. Returns None after Stage 1 completes.

        For final Stage 1 statistics after completion, use stage1_final().
        """
        ...

    def stage1_final(self) -> Optional[PyStage1State]:
        """
        Get Stage 1 state reference even after completion.

        Returns the final Stage 1 state regardless of whether Stage 1 is still
        active. This method should be used for accessing final statistics after
        optimization completes.
        """
        ...

    def stage2(self) -> Optional[PyStage2State]:
        """
        Get current Stage 2 state reference.

        Returns the current Stage 2 state if Stage 2 tracking is enabled and
        Stage 2 has started. Returns None before Stage 2 begins.
        """
        ...

    def flush_messages(self) -> List[str]:
        """
        Get and clear all buffered messages.

        Returns all messages that have been buffered since the last flush.
        The buffer is cleared after this call.

        This is useful in parallel mode where default callbacks buffer messages
        instead of printing them directly. However, the default callback now
        prints messages directly in parallel mode for real-time output, so this
        method may return an empty list.

        :return: A list of buffered messages
        :rtype: List[str]
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class observers:
    """
    Observers module for monitoring OQNLP optimization progress.

    This module provides classes for tracking and monitoring the progress
    of global optimization algorithms. It allows real-time observation of
    algorithm state, performance metrics, and convergence behavior.

    The main components are:

    - Observer: Main class for configuring and accessing optimization state
    - ObserverMode: Enum controlling which optimization stages to monitor
    - Stage1State: Metrics from the scatter search phase (reference set construction)
    - Stage2State: Metrics from the iterative refinement phase

    Examples
    --------
    Basic usage with default logging::

        >>> import pyglobalsearch as gs
        >>> observer = gs.observers.Observer()
        >>> observer.with_default_callback()
        >>> result = gs.optimize(problem, params, observer=observer)

    Custom observer configuration::

        >>> observer = gs.observers.Observer()
        >>> observer.with_stage1_tracking()
        >>> observer.with_stage2_tracking()
        >>> observer.with_timing()
        >>> observer.with_mode(gs.observers.ObserverMode.Both)
        >>> result = gs.optimize(problem, params, observer=observer)

    Accessing final statistics::

        >>> stage1_stats = observer.stage1_final()
        >>> stage2_stats = observer.stage2()
        >>> print(f"Total evaluations: {stage1_stats.function_evaluations + stage2_stats.function_evaluations}")
    """

    Observer: Type[PyObserver]
    ObserverMode: Type[PyObserverMode]
    Stage1State: Type[PyStage1State]
    Stage2State: Type[PyStage2State]

def optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Optional[str] = None,
    local_solver_config: Optional[
        Union[
            PyLBFGS,
            PyGradientDescent,
            PyTrustRegion,
            PyNelderMead,
            PyLBFGSB,
            PyBoundedNelderMead,
            PyBOBYQA,
            PyCOBYLA,
            PySLSQP,
            PyBarrier,
            PyAugmentedLagrangian,
        ]
    ] = None,
    seed: Optional[int] = 0,
    target_objective: Optional[float] = None,
    max_time: Optional[float] = None,
    verbose: Optional[bool] = False,
    exclude_out_of_bounds: Optional[bool] = False,
    parallel: Optional[bool] = False,
    observer: Optional[PyObserver] = None,
    with_points: Optional[Union[List[List[float]], NDArray[np.float64]]] = None,
) -> PySolutionSet:
    """
    Perform global optimization on the given problem.

    This function implements the OQNLP (OptQuest/NLP) algorithm, which combines
    scatter search metaheuristics with local optimization to find global minima
    of nonlinear problems. It's particularly effective for multi-modal functions
    with multiple local minima.

    The algorithm works in two stages:
    1. Scatter search to explore the parameter space and identify promising regions
    2. Local optimization from multiple starting points to refine solutions

    **Examples**

    Basic optimization::

        >>> result = gs.optimize(problem, params)
        >>> best = result.best_solution()

    With custom solver configuration::

        >>> cobyla_config = gs.builders.cobyla(max_iter=1000)
        >>> result = gs.optimize(problem, params, local_solver_config=cobyla_config)

    Or using just the solver name for a default configuration::

        >>> result = gs.optimize(problem, params, local_solver="LBFGS")

    With observer for progress monitoring::

        >>> observer = gs.observers.Observer()
        >>> observer.with_default_callback()
        >>> result = gs.optimize(problem, params, observer=observer)

    With early stopping::

        >>> result = gs.optimize(problem, params,
        ...                     target_objective=-1.0316,  # Stop when reached
        ...                     max_time=60.0,             # Max 60 seconds
        ...                     verbose=True)              # Show progress

    With custom starting points::

        >>> import numpy as np
        >>> custom_points = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> result = gs.optimize(problem, params, with_points=custom_points)

    :param problem: The optimization problem to solve (objective, bounds, constraints, etc.)
    :type problem: PyProblem
    :param params: Parameters controlling the optimization algorithm behavior
    :type params: PyOQNLPParams
    :param local_solver: Local optimization algorithm to use with its default configuration.
                        One of: ``"COBYLA"`` (default when neither argument is given), ``"LBFGS"``,
                        ``"LBFGSB"``, ``"GradientDescent"``, ``"TrustRegion"``, ``"NelderMead"``,
                        ``"BoundedNelderMead"``, ``"BOBYQA"``, ``"SLSQP"``, ``"Barrier"``,
                        ``"AugmentedLagrangian"``.
                        When passed alongside ``local_solver_config``, must match the config type.
    :type local_solver: str, optional
    :param local_solver_config: Custom configuration for the local solver. The solver type is inferred
                               from the config object's type (e.g. ``PyCOBYLA``, ``PyLBFGS``, ``PySLSQP``).
                               When passed alongside ``local_solver``, both must refer to the same solver type.
    :type local_solver_config: Union[PyLBFGS, PyGradientDescent, PyTrustRegion, PyNelderMead, PyLBFGSB, PyBoundedNelderMead, PyBOBYQA, PyCOBYLA, PySLSQP, PyBarrier, PyAugmentedLagrangian], optional
    :param seed: Random seed for reproducible results (0 by default)
    :type seed: int
    :param target_objective: Stop optimization when this objective value is reached (None by default = no target)
    :type target_objective: float
    :param max_time: Maximum time in seconds for Stage 2 optimization (None by default = unlimited)
    :type max_time: float
    :param verbose: Print progress information during optimization (False by default)
    :type verbose: bool
    :param exclude_out_of_bounds: Filter out solutions that violate bounds (False by default)
    :type exclude_out_of_bounds: bool
    :param parallel: Enable parallel processing using rayon (False by default)
    :type parallel: bool
    :param observer: Observer for monitoring optimization progress and metrics (None by default = no observation)
    :type observer: Optional[PyObserver]
    :param with_points: Custom starting points to include in the reference set (None by default).
                        Can be a 2D numpy array of shape (n_points, n_dimensions) or a list of lists.
                        Points must be within the variable bounds.
    :type with_points: Optional[Union[List[List[float]], NDArray[np.float64]]]
    :return: A set of local solutions found during optimization
    :rtype: PySolutionSet
    :raises ValueError: If solver configuration doesn't match the specified solver type,
                        or if the problem is not properly defined.
    """
    ...
