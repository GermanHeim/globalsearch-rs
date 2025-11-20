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
    variable_bounds: BoundsFunctionProtocol
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

    - **variable_bounds**: ``() -> NDArray[np.float64]``
        Returns array of shape ``(n_vars, 2)`` with ``[lower, upper]`` bounds per variable

    **Solver Requirements**

    Different local solvers have different requirements:

    - **COBYLA**: Only objective and bounds required (derivative-free)
    - **NelderMead**: Only objective and bounds required (derivative-free)
    - **LBFGS**: Requires objective, bounds, and gradient
    - **SteepestDescent**: Requires objective, bounds, and gradient
    - **NewtonCG**: Requires objective, bounds, gradient, and Hessian
    - **TrustRegion**: Requires objective, bounds, gradient, and Hessian

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

    Constrained problem (use with COBYLA)::

        >>> def constraint(x):
        ...     return x[0] + x[1] - 1  # Constraint: x[0] + x[1] >= 1
        >>> problem = PyProblem(objective, bounds, constraints=[constraint])

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
    variable_bounds: Callable[[], NDArray[np.float64]]
    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
    constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]
    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], float],
        variable_bounds: Callable[[], NDArray[np.float64]],
        gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
        constraints: Optional[List[Callable[[NDArray[np.float64]], float]]] = None,
    ) -> None:
        """
        Initialize an optimization problem.

        The objective function and the variable bounds are required.

        The gradient and hessian functions are optional, but should be provided
        if the local solver requires them (see class docstring for solver requirements).

        The constraints are optional and should be provided as a list of constraint
        functions if the local solver supports constraints (e.g., COBYLA).

        :param objective: Function that computes the objective value to be minimized
        :type objective: Callable[[NDArray[np.float64]], float]
        :param variable_bounds: Function that returns an array of [lower, upper] bounds for each variable
        :type variable_bounds: Callable[[], NDArray[np.float64]]
        :param gradient: Optional function that computes the gradient of the objective
        :type gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param hessian: Optional function that computes the Hessian of the objective
        :type hessian: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        :param constraints: Optional list of constraint functions. Each constraint is satisfied when constraint(x) >= 0
        :type constraints: Optional[List[Callable[[NDArray[np.float64]], float]]]

        :raises ValueError: If functions have incorrect signatures or bounds have wrong shape
        """
        ...

class PyLineSearchMethod:
    """
    Base class for line search methods.

    Line search methods are used in gradient-based optimization algorithms
    to determine the step size along the search direction. This class provides
    factory methods for creating specific line search configurations.

    Available methods:
        - Hager-Zhang: Robust line search with strong Wolfe conditions
        - More-Thuente: Efficient line search with cubic interpolation

    Examples
    --------
        # Using factory methods
        hz_method = PyLineSearchMethod.hagerzhang()
        mt_method = PyLineSearchMethod.morethunte()
    """
    @staticmethod
    def hagerzhang() -> "PyLineSearchMethod": ...
    @staticmethod
    def morethunte() -> "PyLineSearchMethod": ...

class HagerZhang(PyLineSearchMethod):
    """
    Hager-Zhang line search configuration.

    Implements the Hager-Zhang line search algorithm, which is a robust
    line search method that satisfies the strong Wolfe conditions and
    provides good performance for gradient-based optimization methods.

    Examples
    --------
        >>> hagerzhang_config = HagerZhang(delta=0.05, sigma=0.95)
    """

    delta: float
    sigma: float
    epsilon: float
    theta: float
    gamma: float
    eta: float
    bounds: List[float]
    def __init__(
        self,
        delta: float = 0.1,
        sigma: float = 0.9,
        epsilon: float = 1e-6,
        theta: float = 0.5,
        gamma: float = 0.66,
        eta: float = 0.01,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> None:
        """
        Initialize Hager-Zhang line search configuration.

        :param delta: Armijo parameter for sufficient decrease condition (default 0.1)
        :type delta: float
        :param sigma: Wolfe parameter for curvature condition (default 0.9)
        :type sigma: float
        :param epsilon: Tolerance for the line search termination (default 1e-6)
        :type epsilon: float
        :param theta: Parameter controlling the bracketing phase (default 0.5)
        :type theta: float
        :param gamma: Expansion factor for the bracketing phase (default 0.66)
        :type gamma: float
        :param eta: Contraction factor for the sectioning phase (default 0.01)
        :type eta: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        """
        ...

class MoreThuente(PyLineSearchMethod):
    """
    More-Thuente line search configuration.

    Implements the More-Thuente line search algorithm, which uses cubic
    interpolation to efficiently find step sizes that satisfy the Wolfe
    conditions. This method is widely used in optimization algorithms.

    Examples
    --------
        >>> morethuente_config = MoreThuente(c1=1e-3, c2=0.8)
    """

    c1: float
    c2: float
    width_tolerance: float
    bounds: List[float]
    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> None:
        """
        Initialize More-Thuente line search configuration.

        :param c1: Armijo parameter for sufficient decrease condition (default 1e-4)
        :type c1: float
        :param c2: Wolfe parameter for curvature condition (default 0.9)
        :type c2: float
        :param width_tolerance: Tolerance for the interval width (default 1e-10)
        :type width_tolerance: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        """
        ...

class PyLineSearchParams(PyLineSearchMethod):
    """
    Wrapper for line search parameters.

    This class provides a unified interface for different line search
    parameter configurations (HagerZhang or MoreThuente).

    Examples
    --------
        >>> hagerzhang_params = HagerZhang(delta=0.1)
        >>> line_search = PyLineSearchParams(hagerzhang_params)
    """

    params: Union[HagerZhang, MoreThuente]
    def __init__(self, params: Union[HagerZhang, MoreThuente]) -> None:
        """
        Initialize line search parameters wrapper.

        :param params: Line search configuration (HagerZhang or MoreThuente)
        :type params: Union[HagerZhang, MoreThuente]
        """
        ...

class PyLBFGS:
    """
    Configuration for the L-BFGS (Limited-memory BFGS) solver.

    L-BFGS is a quasi-Newton optimization method that approximates the
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using limited memory.
    It's efficient for large-scale optimization problems requiring gradients.

    Examples
    --------
        >>> lbfgs_config = PyLBFGS(max_iter=500, history_size=20)
    """

    max_iter: int
    tolerance_grad: float
    tolerance_cost: float
    history_size: int
    l1_coefficient: Optional[float]
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        tolerance_grad: float = 1.490116119384766e-8,
        tolerance_cost: float = 2.220446049250313e-16,
        history_size: int = 10,
        l1_coefficient: Optional[float] = None,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = MoreThuente(),
    ) -> None:
        """
        Initialize L-BFGS solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param tolerance_grad: Gradient tolerance for convergence (default 1.49e-8)
        :type tolerance_grad: float
        :param tolerance_cost: Cost function tolerance for convergence (default 2.22e-16)
        :type tolerance_cost: float
        :param history_size: Number of previous gradients to store (default 10)
        :type history_size: int
        :param l1_coefficient: L1 regularization coefficient (optional)
        :type l1_coefficient: Optional[float]
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
        ...

class PyNelderMead:
    """
    Configuration for the Nelder-Mead simplex solver.

    Nelder-Mead is a derivative-free optimization method that uses a simplex
    (a geometric figure with n+1 vertices in n dimensions) to iteratively
    search for the minimum. It's particularly useful when gradients are not available.

    Examples
    --------
        >>> nelder_mead_config = PyNelderMead(max_iter=1000, alpha=1.5)
    """

    simplex_delta: float
    sd_tolerance: float
    max_iter: int
    alpha: float
    gamma: float
    rho: float
    sigma: float
    def __init__(
        self,
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> None:
        """
        Initialize Nelder-Mead solver configuration.

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        """
        ...

class PySteepestDescent:
    """
    Configuration for the steepest descent (gradient descent) solver.

    Steepest descent is a first-order optimization algorithm that iteratively
    moves in the direction of steepest descent (negative gradient) to find
    local minima. It requires gradient information.

    Examples
    --------
        >>> steepest_config = PySteepestDescent(max_iter=1000)
    """

    max_iter: int
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = PyLineSearchMethod.morethunte(),
    ) -> None:
        """
        Initialize steepest descent solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
        ...

class PyNewtonCG:
    """
    Configuration for the Newton-CG (Newton Conjugate Gradient) solver.

    Newton-CG is a second-order optimization method that uses the conjugate
    gradient algorithm to approximately solve the Newton step. It's efficient
    for problems where the Hessian is large but can be computed or approximated.

    Examples
    --------
        >>> newton_cg_config = PyNewtonCG(max_iter=500, tolerance=1e-10)
    """

    max_iter: int
    curvature_tolerance: float
    tolerance: float
    line_search_params: Union[
        PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
    ]
    def __init__(
        self,
        max_iter: int = 300,
        curvature_tolerance: float = 0.0,
        tolerance: float = 1.490116119384766e-8,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams
        ] = PyLineSearchMethod.morethunte(),
    ) -> None:
        """
        Initialize Newton-CG solver configuration.

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param curvature_tolerance: Tolerance for negative curvature detection (default 0.0)
        :type curvature_tolerance: float
        :param tolerance: Convergence tolerance for the Newton step (default 1.49e-8)
        :type tolerance: float
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        """
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
    Configuration for the trust region optimization solver.

    Trust region methods solve optimization problems by restricting steps to
    within a "trust region" where the quadratic model is considered reliable.
    The method adjusts the trust region size based on the agreement between
    the model and the actual function.

    Examples
    --------
        >>> trustregion_config = PyTrustRegion(radius=2.0, max_radius=50.0)
    """

    trust_region_radius_method: PyTrustRegionRadiusMethod
    max_iter: int
    radius: float
    max_radius: float
    eta: float
    def __init__(
        self,
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.cauchy(),
        max_iter: int = 300,
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> None:
        """
        Initialize trust region solver configuration.

        :param trust_region_radius_method: Method for computing trust region radius (default Cauchy)
        :type trust_region_radius_method: PyTrustRegionRadiusMethod
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param radius: Initial trust region radius (default 1.0)
        :type radius: float
        :param max_radius: Maximum allowed trust region radius (default 100.0)
        :type max_radius: float
        :param eta: Threshold for accepting/rejecting steps (default 0.125)
        :type eta: float
        """
        ...

class PyCOBYLA:
    """
    Configuration for the COBYLA (Constrained Optimization BY Linear Approximations) solver.

    COBYLA is a derivative-free optimization algorithm that can handle inequality constraints.
    It's particularly useful when gradients are not available or when dealing with noisy
    objective functions.

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
        Maximum number of iterations
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

class builders:
    @staticmethod
    def hagerzhang(
        delta: float = 0.1,
        sigma: float = 0.9,
        epsilon: float = 1e-6,
        theta: float = 0.5,
        gamma: float = 0.66,
        eta: float = 0.01,
        bounds: List[float] = [1.490116119384766e-8, 10e20],
    ) -> HagerZhang:
        """
        Create a Hager-Zhang line search configuration.

        This builder function allows easy creation of a Hager-Zhang line search
        configuration with custom parameters for gradient-based optimization methods.

        Examples
        --------
            >>> hagerzhang_config = gs.builders.hagerzhang(delta=0.05, sigma=0.95)

        :param delta: Armijo parameter for sufficient decrease condition (default 0.1)
        :type delta: float
        :param sigma: Wolfe parameter for curvature condition (default 0.9)
        :type sigma: float
        :param epsilon: Tolerance for the line search termination (default 1e-6)
        :type epsilon: float
        :param theta: Parameter controlling the bracketing phase (default 0.5)
        :type theta: float
        :param gamma: Expansion factor for the bracketing phase (default 0.66)
        :type gamma: float
        :param eta: Contraction factor for the sectioning phase (default 0.01)
        :type eta: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        :return: Configured Hager-Zhang line search
        :rtype: HagerZhang
        """
        ...
    @staticmethod
    def morethuente(
        c1: float = 1e-4,
        c2: float = 0.9,
        width_tolerance: float = 1e-10,
        bounds: List[float] = [1.490116119384766e-8, 1e20],
    ) -> MoreThuente:
        """
        Create a Moré-Thuente line search configuration.

        This builder function allows easy creation of a Moré-Thuente line search
        configuration with custom parameters for gradient-based optimization methods.

        Examples
        --------
            >>> morethuente_config = gs.builders.morethuente(c1=1e-3, c2=0.8)

        :param c1: Armijo parameter for sufficient decrease condition (default 1e-4)
        :type c1: float
        :param c2: Wolfe parameter for curvature condition (default 0.9)
        :type c2: float
        :param width_tolerance: Tolerance for the interval width (default 1e-10)
        :type width_tolerance: float
        :param bounds: Step size bounds [min, max] (default [1.49e-8, 1e20])
        :type bounds: List[float]
        :return: Configured Moré-Thuente line search
        :rtype: MoreThuente
        """
        ...
    @staticmethod
    def lbfgs(
        max_iter: int = 300,
        tolerance_grad: float = 1.490116119384766e-8,
        tolerance_cost: float = 2.220446049250313e-16,
        history_size: int = 10,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = MoreThuente(),
        l1_coefficient: Optional[float] = None,
    ) -> "PyLBFGS":
        """
        Create an L-BFGS solver configuration.

        This builder function allows easy creation of an L-BFGS (Limited-memory
        Broyden-Fletcher-Goldfarb-Shanno) configuration with custom parameters.

        Examples
        --------
            >>> lbfgs_config = gs.builders.lbfgs(max_iter=500, history_size=20)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param tolerance_grad: Gradient tolerance for convergence (default 1.49e-8)
        :type tolerance_grad: float
        :param tolerance_cost: Cost function tolerance for convergence (default 2.22e-16)
        :type tolerance_cost: float
        :param history_size: Number of previous gradients to store (default 10)
        :type history_size: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :param l1_coefficient: L1 regularization coefficient (optional)
        :type l1_coefficient: float
        :return: Configured L-BFGS solver
        :rtype: PyLBFGS
        """
        ...
    @staticmethod
    def nelder_mead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead":
        """
        Create a Nelder-Mead solver configuration.

        This builder function allows easy creation of a Nelder-Mead simplex algorithm
        configuration.

        Examples
        --------
            >>> nelder_mead_config = gs.builders.nelder_mead(max_iter=1000, alpha=1.5)

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        :return: Configured Nelder-Mead solver
        :rtype: PyNelderMead
        """
        ...
    @staticmethod
    def neldermead(
        simplex_delta: float = 0.1,
        sd_tolerance: float = 2.220446049250313e-16,
        max_iter: int = 300,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> "PyNelderMead":
        """
        Create a Nelder-Mead solver configuration.

        This builder function allows easy creation of a Nelder-Mead simplex algorithm
        configuration.

        Examples
        --------
            >>> neldermead_config = gs.builders.neldermead(max_iter=1000, alpha=1.5)

        :param simplex_delta: Initial simplex size (default 0.1)
        :type simplex_delta: float
        :param sd_tolerance: Standard deviation tolerance for convergence (default 2.22e-16)
        :type sd_tolerance: float
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param alpha: Reflection coefficient (default 1.0)
        :type alpha: float
        :param gamma: Expansion coefficient (default 2.0)
        :type gamma: float
        :param rho: Contraction coefficient (default 0.5)
        :type rho: float
        :param sigma: Shrink coefficient (default 0.5)
        :type sigma: float
        :return: Configured Nelder-Mead solver
        :rtype: PyNelderMead
        """
        ...
    @staticmethod
    def steepest_descent(
        max_iter: int = 300,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PySteepestDescent":
        """
        Create a steepest descent solver configuration.

        This builder function allows easy creation of a steepest descent (gradient descent)
        configuration.

        Examples
        --------
            >>> steepest_config = gs.builders.steepest_descent(max_iter=1000)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :return: Configured steepest descent solver
        :rtype: PySteepestDescent
        """
        ...
    @staticmethod
    def newton_cg(
        max_iter: int = 300,
        curvature_tolerance: float = 0.0,
        tolerance: float = 1.490116119384766e-8,
        line_search_params: Union[
            PyLineSearchMethod, HagerZhang, MoreThuente, "PyLineSearchParams"
        ] = PyLineSearchMethod.morethunte(),
    ) -> "PyNewtonCG":
        """
        Create a Newton-CG solver configuration.

        This builder function allows easy creation of a Newton-CG (Newton Conjugate Gradient)
        configuration.

        Examples
        --------
            >>> newton_cg_config = gs.builders.newton_cg(max_iter=500, tolerance=1e-10)

        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param curvature_tolerance: Tolerance for negative curvature detection (default 0.0)
        :type curvature_tolerance: float
        :param tolerance: Convergence tolerance for the Newton step (default 1.49e-8)
        :type tolerance: float
        :param line_search_params: Line search configuration (default MoreThuente)
        :type line_search_params: Union[PyLineSearchMethod, HagerZhang, MoreThuente, PyLineSearchParams]
        :return: Configured Newton-CG solver
        :rtype: PyNewtonCG
        """
        ...
    @staticmethod
    def trustregion(
        trust_region_radius_method: PyTrustRegionRadiusMethod = PyTrustRegionRadiusMethod.cauchy(),
        max_iter: int = 300,
        radius: float = 1.0,
        max_radius: float = 100.0,
        eta: float = 0.125,
    ) -> PyTrustRegion:
        """
        Create a trust region solver configuration.

        This builder function allows easy creation of a trust region method
        configuration.

        Examples
        --------
            >>> trustregion_config = gs.builders.trustregion(radius=2.0, max_radius=50.0)

        :param trust_region_radius_method: Method for computing trust region radius (default Cauchy)
        :type trust_region_radius_method: PyTrustRegionRadiusMethod
        :param max_iter: Maximum number of iterations (default 300)
        :type max_iter: int
        :param radius: Initial trust region radius (default 1.0)
        :type radius: float
        :param max_radius: Maximum allowed trust region radius (default 100.0)
        :type max_radius: float
        :param eta: Threshold for accepting/rejecting steps (default 0.125)
        :type eta: float
        :return: Configured trust region solver
        :rtype: PyTrustRegion
        """
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

        :param max_iter: Maximum number of iterations (default 300)
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
    PyHagerZhang: Type[HagerZhang]
    PyMoreThuente: Type[MoreThuente]
    PyLineSearchParams: Type[PyLineSearchParams]

    PyLBFGS: Type[PyLBFGS]
    PyNelderMead: Type[PyNelderMead]
    PySteepestDescent: Type[PySteepestDescent]
    PyNewtonCG: Type[PyNewtonCG]
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
    local_solver: Optional[str] = "COBYLA",
    local_solver_config: Optional[
        Union[
            PyLBFGS,
            PyNelderMead,
            PySteepestDescent,
            PyNewtonCG,
            PyTrustRegion,
            PyCOBYLA,
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
        >>> result = gs.optimize(problem, params,
        ...                     local_solver="COBYLA",
        ...                     local_solver_config=cobyla_config)

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
    :param local_solver: Local optimization algorithm ("COBYLA", "LBFGS", "NewtonCG",
                        "TrustRegion", "NelderMead", "SteepestDescent")
    :type local_solver: str
    :param local_solver_config: Custom configuration for the local solver (None for defaults)
    :type local_solver_config: Union[PyLBFGS, PyNelderMead, PySteepestDescent, PyNewtonCG, PyTrustRegion, PyCOBYLA]
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
