mod builders;
mod observers;

use crate::observers::PyObserver;
use globalsearch::local_solver::builders::{
    COBYLABuilder, LBFGSBuilder, NelderMeadBuilder, NewtonCGBuilder, SteepestDescentBuilder,
    TrustRegionBuilder,
};
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, LocalSolverType, OQNLPParams};
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use seq_macro::seq;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Global constraint registry for COBYLA constraints
type ConstraintRegistry = Arc<Mutex<HashMap<usize, Vec<Py<pyo3::PyAny>>>>>;
static CONSTRAINT_REGISTRY: std::sync::OnceLock<ConstraintRegistry> = std::sync::OnceLock::new();
static PROBLEM_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Global mutex to serialize Python calls from parallel threads
static PYTHON_CALL_MUTEX: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();

// Thread-local storage for current problem ID and constraint index during evaluation
thread_local! {
    static CURRENT_PROBLEM_ID: Cell<usize> = const { Cell::new(0) };
    static CURRENT_CONSTRAINT_INDEX: Cell<usize> = const { Cell::new(0) };
}

fn get_python_call_mutex() -> &'static Mutex<()> {
    PYTHON_CALL_MUTEX.get_or_init(|| Mutex::new(()))
}

fn get_constraint_registry() -> &'static ConstraintRegistry {
    CONSTRAINT_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

fn get_next_problem_id() -> usize {
    PROBLEM_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

fn set_current_problem_id(id: usize) {
    CURRENT_PROBLEM_ID.with(|current| current.set(id));
}

fn get_current_problem_id() -> usize {
    CURRENT_PROBLEM_ID.with(|current| current.get())
}

// Helper function to evaluate all constraints for a given problem ID
fn evaluate_constraints_for_id(problem_id: usize, x: &[f64]) -> Vec<f64> {
    let registry = get_constraint_registry();
    let registry_lock = registry.lock().unwrap();

    if let Some(constraints) = registry_lock.get(&problem_id) {
        Python::attach(|py| {
            constraints
                .iter()
                .map(|constraint| {
                    let args = (x.to_vec(),);
                    constraint
                        .call1(py, args)
                        .expect("Failed to call constraint function")
                        .extract::<f64>(py)
                        .expect("Constraint function must return a float")
                })
                .collect()
        })
    } else {
        Vec::new()
    }
}

// Generate constraint functions dynamically using macro (up to 1000 constraints)
const MAX_CONSTRAINTS: usize = 1000;

seq!(N in 0..1000 {
    #[allow(clippy::get_first)]
    fn constraint_fn_~N(x: &[f64], _user_data: &mut ()) -> f64 {
        let problem_id = get_current_problem_id();
        let constraints = evaluate_constraints_for_id(problem_id, x);
        constraints.get(N).copied().unwrap_or(0.0)
    }
});

// Helper to get constraint function pointers dynamically
fn get_constraint_functions(num_constraints: usize) -> Vec<fn(&[f64], &mut ()) -> f64> {
    if num_constraints > MAX_CONSTRAINTS {
        panic!(
            "Too many constraints! Maximum supported: {}, requested: {}",
            MAX_CONSTRAINTS, num_constraints
        );
    }

    let mut functions = Vec::new();

    seq!(N in 0..1000 {
        if N < num_constraints {
            functions.push(constraint_fn_~N as fn(&[f64], &mut ()) -> f64);
        }
    });

    functions
}

#[pyclass]
#[derive(Debug, Clone)]
/// Parameters for the OQNLP global optimization algorithm.
///
/// The OQNLP algorithm combines scatter search metaheuristics with local optimization
/// to find global minima in nonlinear optimization problems. These parameters control
/// the behavior of the algorithm.
///
/// :param iterations: Maximum number of global iterations
/// :type iterations: int
/// :param population_size: Size of the scatter search population
/// :type population_size: int
/// :param wait_cycle: Iterations to wait without improvement before termination
/// :type wait_cycle: int
/// :param threshold_factor: Controls acceptance threshold for new solutions
/// :type threshold_factor: float
/// :param distance_factor: Controls minimum distance between solutions
/// :type distance_factor: float
/// :param abs_tol: Absolute tolerance for comparing objective values (default: 1e-8)
/// :type abs_tol: float
/// :param rel_tol: Relative tolerance for comparing objective values (default: 1e-6)
/// :type rel_tol: float
///
/// Examples
/// --------
/// Default parameters:
///
/// >>> params = gs.PyOQNLPParams()
///
/// Custom parameters for difficult problems:
///
/// >>> params = gs.PyOQNLPParams(
/// ...     iterations=500,
/// ...     population_size=2000,
/// ...     wait_cycle=25,
/// ...     threshold_factor=0.1,  # More exploration
/// ...     distance_factor=0.2    # Allow closer solutions
/// ... )
pub struct PyOQNLPParams {
    #[pyo3(get, set)]
    /// Maximum number of stage two iterations
    pub iterations: usize,
    #[pyo3(get, set)]
    /// Size of the scatter search population
    pub population_size: usize,
    #[pyo3(get, set)]
    /// Iterations to wait without improvement before termination
    pub wait_cycle: usize,
    #[pyo3(get, set)]
    /// Controls acceptance threshold for new solutions
    pub threshold_factor: f64,
    #[pyo3(get, set)]
    /// Controls minimum distance between solutions
    pub distance_factor: f64,
    #[pyo3(get, set)]
    /// Absolute tolerance for comparing objective values (default: 1e-8)
    pub abs_tol: f64,
    #[pyo3(get, set)]
    /// Relative tolerance for comparing objective values (default: 1e-6)
    pub rel_tol: f64,
}

#[pyclass]
#[derive(Debug, Clone)]
/// A local solution found by the optimization algorithm.
///
/// Represents a single solution point in parameter space along with its
/// objective function value. This class provides both direct attribute access
/// and SciPy-compatible methods for accessing solution data.
///
/// :ivar point: The solution coordinates as a list of float values
/// :vartype point: list[float]
/// :ivar objective: The objective function value at this solution point
/// :vartype objective: float
///
/// Examples
/// --------
/// Create and access a solution:
///
/// >>> solution = PyLocalSolution([1.0, 2.0], 3.5)
/// >>> # Access via attributes
/// >>> x_coords = solution.point
/// >>> f_value = solution.objective
/// >>> # Access via SciPy-compatible methods
/// >>> x_coords = solution.x()
/// >>> f_value = solution.fun()
pub struct PyLocalSolution {
    #[pyo3(get, set)]
    /// The solution coordinates as a list of float values
    ///
    /// :returns: The solution coordinates as a list of float values (same as `x()` method)
    /// :rtype: list[float]
    pub point: Vec<f64>,
    #[pyo3(get, set)]
    /// The objective function value at this solution point
    ///
    /// :returns: The objective function value at this solution point (same as `fun()` method)
    /// :rtype: float
    pub objective: f64,
}

#[pymethods]
impl PyLocalSolution {
    #[new]
    fn new(point: Vec<f64>, objective: f64) -> Self {
        PyLocalSolution { point, objective }
    }

    /// Returns the objective function value at the solution point.
    ///
    /// :returns: The objective function value at this solution point (same as `objective` attribute)
    /// :rtype: float
    ///
    /// .. note::
    ///    This method is similar to the `fun` method in SciPy.optimize results.
    fn fun(&self) -> f64 {
        self.objective
    }

    /// Returns the solution point as a list of float values.
    ///
    /// :returns: The solution coordinates as a list of float values (same as `point` attribute)
    /// :rtype: list[float]
    ///
    /// .. note::
    ///    This method is similar to the `x` method in SciPy.optimize results.
    fn x(&self) -> Vec<f64> {
        self.point.clone()
    }

    fn __repr__(&self) -> String {
        format!("PyLocalSolution(point={:?}, objective={})", self.point, self.objective)
    }

    fn __str__(&self) -> String {
        format!("Solution(x={:?}, fun={})", self.point, self.objective)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
/// A collection of local solutions found by the optimization algorithm.
///
/// The OQNLP algorithm typically finds multiple local solutions during its search.
/// This class stores all solutions and provides methods to access them, find the
/// best solution, and iterate over the results.
///
/// Solutions are automatically sorted by objective function value (best first).
///
/// Examples
/// --------
/// Get optimization results and access solutions:
///
/// >>> result = gs.optimize(problem, params)
/// >>> # Access best solution
/// >>> best = result.best_solution()
/// >>> if best:
/// ...     print(f"Best: x = {best.x()}, f(x) = {best.fun()}")
/// >>> # Check number of solutions
/// >>> print(f"Found {len(result)} solutions")
/// >>> # Iterate over all solutions
/// >>> for i, solution in enumerate(result):
/// ...     print(f"Solution {i}: f(x) = {solution.fun()}")
/// >>> # Access specific solution by index
/// >>> second_best = result[1] if len(result) > 1 else None
pub struct PySolutionSet {
    #[pyo3(get)]
    /// The list of local solutions found by the optimization algorithm.
    ///
    /// :returns: The list of local solutions found by the optimization algorithm
    /// :rtype: list[PyLocalSolution]
    pub solutions: Vec<PyLocalSolution>,
}

#[pymethods]
impl PySolutionSet {
    #[new]
    fn new(solutions: Vec<PyLocalSolution>) -> Self {
        PySolutionSet { solutions }
    }

    /// Returns the number of solutions stored in the set.
    ///
    /// :returns: Number of solutions in the set
    /// :rtype: int
    fn __len__(&self) -> usize {
        self.solutions.len()
    }

    /// Returns true if the solution set contains no solutions.
    ///
    /// :returns: True if the solution set is empty, False otherwise
    /// :rtype: bool
    fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Returns the best solution in the set based on the objective function value.
    ///
    /// :returns: The solution with the lowest objective function value, or None if the set is empty
    /// :rtype: PyLocalSolution or None
    fn best_solution(&self) -> Option<PyLocalSolution> {
        self.solutions.iter().min_by(|a, b| a.objective.partial_cmp(&b.objective).unwrap()).cloned()
    }

    /// Returns the solution at the given index.
    ///
    /// :param index: Index of the solution to retrieve
    /// :type index: int
    /// :returns: The solution at the specified index
    /// :rtype: PyLocalSolution
    /// :raises IndexError: If the index is out of range
    fn __getitem__(&self, index: usize) -> PyResult<PyLocalSolution> {
        self.solutions
            .get(index)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of range"))
    }

    /// Returns an iterator over the solutions in the set.
    ///
    /// :returns: An iterator over all solutions in the set
    /// :rtype: PySolutionSetIterator
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySolutionSetIterator>> {
        let iter = PySolutionSetIterator { inner: slf.solutions.clone(), index: 0 };
        Py::new(slf.py(), iter)
    }

    fn __repr__(&self) -> String {
        format!("PySolutionSet(solutions={:?})", self.solutions)
    }

    fn __str__(&self) -> String {
        let mut result = String::from("Solution Set\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        result.push_str(&format!("Total solutions: {}\n", self.solutions.len()));

        if !self.solutions.is_empty() {
            if let Some(best) = self.best_solution() {
                result.push_str(&format!("Best objective value: {:.8e}\n", best.objective));
            }
        }
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (i, solution) in self.solutions.iter().enumerate() {
            result.push_str(&format!("Solution #{}\n", i + 1));
            result.push_str(&format!("  Objective: {:.8e}\n", solution.objective));
            result.push_str(&format!("  Parameters: {:?}\n", solution.point));

            if i < self.solutions.len() - 1 {
                result.push_str("――――――――――――――――――――――――――――――――――――\n");
            }
        }

        result
    }
}

#[pyclass]
struct PySolutionSetIterator {
    inner: Vec<PyLocalSolution>,
    index: usize,
}

#[pymethods]
impl PySolutionSetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyLocalSolution> {
        if slf.index < slf.inner.len() {
            let result = slf.inner[slf.index].clone();
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[pymethods]
impl PyOQNLPParams {
    #[new]
    #[pyo3(signature = (
        iterations = 300,
        population_size = 1000,
        wait_cycle = 15,
        threshold_factor = 0.2,
        distance_factor = 0.75,
        abs_tol = 1e-8,
        rel_tol = 1e-6,
    ))]
    #[pyo3(
        text_signature = "(iterations=300, population_size=1000, wait_cycle=15, threshold_factor=0.2, distance_factor=0.75, abs_tol=1e-8, rel_tol=1e-6)"
    )]
    fn new(
        iterations: usize,
        population_size: usize,
        wait_cycle: usize,
        threshold_factor: f64,
        distance_factor: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Self {
        PyOQNLPParams {
            iterations,
            population_size,
            wait_cycle,
            threshold_factor,
            distance_factor,
            abs_tol,
            rel_tol,
        }
    }
}

#[pyclass]
#[derive(Debug)]
/// Defines an optimization problem to be solved.
///
/// A PyProblem encapsulates all the mathematical components needed for optimization:
/// the objective function to minimize, variable bounds, and optional gradient, Hessian,
/// and constraint functions.
///
/// :param objective: Function that takes x (array-like) and returns the value to minimize (float)
/// :type objective: callable
/// :param variable_bounds: Function that returns bounds array of shape (n_vars, 2) with [lower, upper] bounds
/// :type variable_bounds: callable
/// :param gradient: Function that takes x and returns gradient array (for gradient-based solvers)
/// :type gradient: callable, optional
/// :param hessian: Function that takes x and returns Hessian matrix (for Newton-type solvers)
/// :type hessian: callable, optional
/// :param constraints: List of constraint functions where constraint(x) >= 0 means satisfied
/// :type constraints: list[callable], optional
///
/// Examples
/// --------
/// Basic unconstrained problem:
///
/// >>> def objective(x): return x[0]**2 + x[1]**2
/// >>> def bounds(): return np.array([[-5, 5], [-5, 5]])
/// >>> problem = gs.PyProblem(objective, bounds)
///
/// Problem with gradient for faster convergence:
///
/// >>> def gradient(x): return np.array([2*x[0], 2*x[1]])
/// >>> problem = gs.PyProblem(objective, bounds, gradient=gradient)
///
/// Constrained problem (requires COBYLA solver):
///
/// >>> def constraint(x): return x[0] + x[1] - 1  # x[0] + x[1] >= 1
/// >>> problem = gs.PyProblem(objective, bounds, constraints=[constraint])
pub struct PyProblem {
    #[pyo3(get, set)]
    /// Objective function to minimize
    ///
    /// :param x: Input parameters as a list or array of floats
    /// :return: Objective function value (float)
    objective: Py<pyo3::PyAny>,

    #[pyo3(get, set)]
    /// Function returning variable bounds
    ///
    /// :returns: 2D array-like of shape (n_vars, 2) with [lower, upper] bounds for each variable
    /// :rtype: array-like
    variable_bounds: Py<pyo3::PyAny>,

    #[pyo3(get, set)]
    /// Function returning the gradient
    ///
    /// :param x: Input parameters as a list or array of floats
    /// :return: Gradient as a list or array of floats
    /// :raises ValueError: If the gradient length does not match the number of variables
    gradient: Option<Py<pyo3::PyAny>>,
    #[pyo3(get, set)]
    /// Function returning the Hessian matrix
    ///
    /// :param x: Input parameters as a list or array of floats
    /// :return: Hessian matrix as a 2D list or array of floats
    /// :raises ValueError: If the Hessian is not square or does not match the number of variables
    hessian: Option<Py<pyo3::PyAny>>,

    #[pyo3(get, set)]
    /// List of constraint functions
    ///
    /// :param x: Input parameters as a list or array of floats
    /// :return: Constraint value (float), should be >= 0 to be satisfied
    /// :raises ValueError: If any constraint function does not return a float
    constraints: Option<Py<pyo3::PyAny>>,

    /// Unique ID for this problem instance (used for constraint management)
    ///
    /// :return: Unique problem ID (integer)
    /// :rtype: int
    problem_id: usize,
}

impl Clone for PyProblem {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            objective: self.objective.clone_ref(py),
            variable_bounds: self.variable_bounds.clone_ref(py),
            gradient: self.gradient.as_ref().map(|g| g.clone_ref(py)),
            hessian: self.hessian.as_ref().map(|h| h.clone_ref(py)),
            constraints: self.constraints.as_ref().map(|c| c.clone_ref(py)),
            problem_id: self.problem_id,
        })
    }
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (objective, variable_bounds, gradient=None, hessian=None, constraints=None))]
    fn new(
        objective: Py<pyo3::PyAny>,
        variable_bounds: Py<pyo3::PyAny>,
        gradient: Option<Py<pyo3::PyAny>>,
        hessian: Option<Py<pyo3::PyAny>>,
        constraints: Option<Py<pyo3::PyAny>>,
    ) -> Self {
        PyProblem {
            objective,
            variable_bounds,
            gradient,
            hessian,
            constraints,
            problem_id: get_next_problem_id(),
        }
    }
}

impl PyProblem {
    /// Evaluate Python constraint functions at a given point
    pub fn evaluate_constraints(&self, x: &[f64]) -> Vec<f64> {
        if let Some(constraints_fn) = &self.constraints {
            Python::attach(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .unwrap_or_else(|_| panic!("Failed to convert x to Python object"));

                // Call the Python constraints function which should return a list of constraint values
                let result = constraints_fn
                    .call1(py, (x_py,))
                    .unwrap_or_else(|_| panic!("Failed to call Python constraints function"));

                // Extract the constraint values as a Vec<f64>
                result.extract::<Vec<f64>>(py).unwrap_or_else(|_| {
                    panic!("Python constraints function must return a list of floats")
                })
            })
        } else {
            vec![]
        }
    }
}

impl Problem for PyProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Serialize Python calls from parallel threads to avoid GIL deadlocks
        let _guard = get_python_call_mutex().lock().unwrap();
        Python::attach(|py| {
            let x_py = x
                .to_vec()
                .into_pyobject(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            let result = self
                .objective
                .call1(py, (x_py,))
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            result.extract(py).map_err(|e: PyErr| EvaluationError::InvalidInput(e.to_string()))
        })
    }

    fn variable_bounds(&self) -> Array2<f64> {
        Python::attach(|py| {
            let result = self
                .variable_bounds
                .call0(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                .and_then(|res| {
                    res.extract::<Vec<Vec<f64>>>(py)
                        .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                });

            match result {
                Ok(bounds) => {
                    let rows = bounds.len();
                    let cols = if rows > 0 { bounds[0].len() } else { 0 };
                    Array2::from_shape_vec((rows, cols), bounds.into_iter().flatten().collect())
                        .unwrap()
                }
                Err(_) => panic!("Variable bounds must be a 2D array of floats"),
            }
        })
    }

    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        if let Some(grad_fn) = &self.gradient {
            Python::attach(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = grad_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let grad_vec: Vec<f64> = result
                    .extract(py)
                    .map_err(|e: PyErr| EvaluationError::InvalidInput(e.to_string()))?;

                Ok(Array1::from(grad_vec))
            })
        } else {
            Err(EvaluationError::GradientNotImplemented)
        }
    }

    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        if let Some(hess_fn) = &self.hessian {
            Python::attach(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = hess_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let hess_vec: Vec<Vec<f64>> = result
                    .extract(py)
                    .map_err(|e: PyErr| EvaluationError::InvalidInput(e.to_string()))?;

                let size = hess_vec.len();
                let flat_hess: Vec<f64> = hess_vec.into_iter().flatten().collect();

                Array2::from_shape_vec((size, size), flat_hess).map_err(|_| {
                    EvaluationError::InvalidInput("Hessian shape mismatch".to_string())
                })
            })
        } else {
            Err(EvaluationError::HessianNotImplemented)
        }
    }

    fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
        if let Some(constraint_funcs) = &self.constraints {
            // Register Python constraints in the global registry
            let registry = get_constraint_registry();
            let mut registry_lock = registry.lock().unwrap();

            Python::attach(|py| {
                // Extract individual constraint functions from the Python list/tuple
                let constraint_list: Vec<Py<pyo3::PyAny>> = if let Ok(list) =
                    constraint_funcs.cast_bound::<pyo3::types::PyList>(py)
                {
                    list.iter().map(|item| item.unbind()).collect()
                } else if let Ok(tuple) = constraint_funcs.cast_bound::<pyo3::types::PyTuple>(py) {
                    tuple.iter().map(|item| item.unbind()).collect()
                } else {
                    // Single constraint function
                    vec![constraint_funcs.clone_ref(py)]
                };

                let num_constraints = constraint_list.len();
                registry_lock.insert(self.problem_id, constraint_list);
                drop(registry_lock); // Release the lock

                // Set the current problem ID for constraint evaluation
                set_current_problem_id(self.problem_id);

                // Return appropriate number of function pointers
                get_constraint_functions(num_constraints)
            })
        } else {
            Vec::new()
        }
    }
}

/// Perform global optimization on the given problem.
///
/// This function implements the OQNLP (OptQuest/NLP) algorithm, which combines
/// scatter search metaheuristics with local optimization to find global minima
/// of nonlinear problems. It's particularly effective for multi-modal functions
/// with multiple local minima.
///
/// The algorithm works in two stages:
///
/// 1. Scatter search to explore the parameter space and identify promising regions
/// 2. Local optimization from multiple starting points to refine solutions
///
/// :param problem: The optimization problem to solve (objective, bounds, constraints, etc.)
/// :type problem: PyProblem
/// :param params: Parameters controlling the optimization algorithm behavior
/// :type params: PyOQNLPParams
/// :param observer: Optional observer for tracking algorithm progress and metrics
/// :type observer: PyObserver, optional
/// :param local_solver: Local optimization algorithm ("COBYLA", "LBFGS", "NewtonCG", "TrustRegion", "NelderMead", "SteepestDescent")
/// :type local_solver: str, optional
/// :param local_solver_config: Custom configuration for the local solver (must match solver type)
/// :type local_solver_config: object, optional
/// :param seed: Random seed for reproducible results
/// :type seed: int, optional
/// :param target_objective: Stop optimization when this objective value is reached
/// :type target_objective: float, optional
/// :param max_time: Maximum time in seconds for Stage 2 optimization (unlimited if None)
/// :type max_time: float, optional
/// :param verbose: Print progress information during optimization
/// :type verbose: bool, optional
/// :param exclude_out_of_bounds: Filter out solutions that violate bounds
/// :type exclude_out_of_bounds: bool, optional
/// :param parallel: Enable parallel processing using rayon (default: False)
/// :type parallel: bool, optional
/// :param with_points: Custom starting points to seed the reference set (2D array-like of shape (n_points, n_vars))
/// :type with_points: array-like, optional
/// :returns: A set of local solutions found during optimization
/// :rtype: PySolutionSet
/// :raises ValueError: If solver configuration doesn't match the specified solver type, or if the problem is not properly defined
///
/// Examples
/// --------
/// Basic optimization:
///
/// >>> result = gs.optimize(problem, params)
/// >>> best = result.best_solution()
///
/// With observer for progress tracking:
///
/// >>> observer = gs.observers.Observer().with_stage1_tracking().with_stage2_tracking().with_default_callback()
/// >>> result = gs.optimize(problem, params, observer=observer)
///
/// With custom solver configuration:
///
/// >>> cobyla_config = gs.builders.cobyla(max_iter=1000)
/// >>> result = gs.optimize(problem, params,
/// ...                     local_solver="COBYLA",
/// ...                     local_solver_config=cobyla_config)
///
/// With early stopping:
///
/// >>> result = gs.optimize(problem, params,
/// ...                     target_objective=-1.0316,  # Stop when reached
/// ...                     max_time=60.0,             # Max 60 seconds
/// ...                     verbose=True)              # Show progress
///
/// Enable parallel processing:
///
/// >>> result = gs.optimize(problem, params, parallel=True)
///
/// With custom starting points:
///
/// >>> import numpy as np
/// >>> custom_points = np.array([[1.0, 2.0], [-1.0, -1.0]])
/// >>> result = gs.optimize(problem, params, with_points=custom_points)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    problem,
    params,
    observer = None,
    local_solver = None,
    local_solver_config = None,
    seed = None,
    target_objective = None,
    max_time = None,
    verbose = None,
    exclude_out_of_bounds = None,
    parallel = None,
    with_points = None,
))]
fn optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    observer: Option<&PyObserver>,
    local_solver: Option<&str>,
    local_solver_config: Option<Py<pyo3::PyAny>>,
    seed: Option<u64>,
    target_objective: Option<f64>,
    max_time: Option<f64>,
    verbose: Option<bool>,
    exclude_out_of_bounds: Option<bool>,
    parallel: Option<bool>,
    with_points: Option<Vec<Vec<f64>>>,
) -> PyResult<PySolutionSet> {
    Python::attach(|py| {
        // Convert local_solver string to enum
        let solver_type = LocalSolverType::from_string(local_solver.unwrap_or("cobyla"))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let seed = seed.unwrap_or(0);

        // Create local solver configuration (default)
        let local_solver_config = if let Some(config) = local_solver_config {
            match solver_type {
                LocalSolverType::LBFGS => {
                    if let Ok(lbfgs_config) = config.extract::<crate::builders::PyLBFGS>(py) {
                        lbfgs_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyLBFGS for LBFGS solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::NelderMead => {
                    if let Ok(neldermead_config) =
                        config.extract::<crate::builders::PyNelderMead>(py)
                    {
                        neldermead_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyNelderMead for NelderMead solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::SteepestDescent => {
                    if let Ok(steepest_descent_config) =
                        config.extract::<crate::builders::PySteepestDescent>(py)
                    {
                        steepest_descent_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PySteepestDescent for SteepestDescent solver type"
                                .to_string(),
                        ));
                    }
                }
                LocalSolverType::NewtonCG => {
                    if let Ok(newtoncg_config) = config.extract::<crate::builders::PyNewtonCG>(py) {
                        newtoncg_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyNewtonCG for NewtonCG solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::TrustRegion => {
                    if let Ok(trustregion_config) =
                        config.extract::<crate::builders::PyTrustRegion>(py)
                    {
                        trustregion_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyTrustRegion for TrustRegion solver type".to_string(),
                        ));
                    }
                }
                LocalSolverType::COBYLA => {
                    if let Ok(cobyla_config) = config.extract::<crate::builders::PyCOBYLA>(py) {
                        cobyla_config.to_builder().build()
                    } else {
                        return Err(PyValueError::new_err(
                            "Expected PyCOBYLA for COBYLA solver type".to_string(),
                        ));
                    }
                }
            }
        } else {
            // Create default local solver configuration
            match solver_type {
                LocalSolverType::LBFGS => LBFGSBuilder::default().build(),
                LocalSolverType::NewtonCG => NewtonCGBuilder::default().build(),
                LocalSolverType::TrustRegion => TrustRegionBuilder::default().build(),
                LocalSolverType::NelderMead => NelderMeadBuilder::default().build(),
                LocalSolverType::SteepestDescent => SteepestDescentBuilder::default().build(),
                LocalSolverType::COBYLA => COBYLABuilder::default().build(),
            }
        };

        let oqnlp_params: OQNLPParams = OQNLPParams {
            iterations: params.iterations,
            population_size: params.population_size,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
            distance_factor: params.distance_factor,
            seed,
            local_solver_type: solver_type,
            local_solver_config,
        };

        let mut optimizer =
            OQNLP::new(problem, oqnlp_params).map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Add custom points if provided
        if let Some(points) = with_points {
            let rows = points.len();
            let cols = if rows > 0 { points[0].len() } else { 0 };
            let flat_points: Vec<f64> = points.into_iter().flatten().collect();
            let points_array = Array2::from_shape_vec((rows, cols), flat_points)
                .map_err(|e| PyValueError::new_err(format!("Invalid points array shape: {}", e)))?;
            optimizer = optimizer
                .with_points(points_array)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let parallel_enabled = parallel.unwrap_or(false);

        // For Python problems with parallel processing, we need special GIL handling
        // When parallel=True, we allow threads during parallel computation but ensure
        // Python calls are serialized with the mutex
        let should_detach_gil = !parallel_enabled;

        // Add observer if provided
        let optimizer = if let Some(py_observer) = observer {
            let cloned_observer = py_observer.clone_inner();
            optimizer.add_observer(cloned_observer)
        } else {
            optimizer
        };

        // Set parallel mode
        let optimizer = optimizer.parallel(parallel_enabled);

        // Apply optional configurations
        let optimizer = if let Some(target) = target_objective {
            optimizer.target_objective(target)
        } else {
            optimizer
        };

        let optimizer =
            if let Some(max_secs) = max_time { optimizer.max_time(max_secs) } else { optimizer };

        let optimizer = if verbose.unwrap_or(false) { optimizer.verbose() } else { optimizer };

        let optimizer = if exclude_out_of_bounds.unwrap_or(false) {
            optimizer.exclude_out_of_bounds()
        } else {
            optimizer
        };

        let mut optimizer = optimizer.with_tolerance(params.abs_tol, params.rel_tol);

        // For parallel execution, we need to allow threads to run without the GIL
        // but ensure Python calls are properly synchronized
        let solution_set = if parallel_enabled {
            // Allow threads during parallel computation
            py.detach(|| optimizer.run())
        } else {
            // For serial execution, detach GIL as before
            if should_detach_gil {
                py.detach(|| optimizer.run())
            } else {
                optimizer.run()
            }
        };

        let binding = solution_set.map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Update the observer in place if provided
        if let Some(py_observer) = observer {
            // For parallel execution, we need to handle observer updates carefully
            if parallel_enabled {
                // Observer updates happen in the main thread after parallel execution
                if let Some(updated_observer) = py.detach(|| optimizer.observer().cloned()) {
                    *py_observer.inner.write().unwrap() = updated_observer;
                } else if let Some(updated_observer) = optimizer.observer() {
                    *py_observer.inner.write().unwrap() = updated_observer.clone();
                }
            } else {
                // For sequential execution, update observer directly
                if let Some(updated_observer) = optimizer.observer() {
                    *py_observer.inner.write().unwrap() = updated_observer.clone();
                }
            }
        }
        let py_solutions: Vec<PyLocalSolution> = binding
            .solutions()
            .map(|sol| PyLocalSolution { point: sol.point.to_vec(), objective: sol.objective })
            .collect();

        Ok(PySolutionSet::new(py_solutions))
    })
}

#[pymodule]
/// PyGlobalSearch: Python bindings for globalsearch-rs.
///
/// PyGlobalSearch provides a Python interface to the `globalsearch-rs` Rust crate,
/// which implements the OQNLP (OptQuest/NLP) algorithm for global optimization.
///
/// The OQNLP algorithm combines scatter search metaheuristics with local optimization
/// to effectively find global minima in nonlinear optimization problems. It's particularly
/// effective for:
///
/// * Multi-modal functions with multiple minima
/// * Nonlinear optimization problems
/// * Problems where derivative information may be unavailable or unreliable
/// * Constrained optimization (using COBYLA solver)
///
/// Quick Start
/// -----------
/// ::
///
///     import pyglobalsearch as gs
///     import numpy as np
///
///     # Define your problem
///     def objective(x): return x[0]**2 + x[1]**2
///     def bounds(): return np.array([[-5, 5], [-5, 5]])
///
///     # Create problem and parameters
///     problem = gs.PyProblem(objective, bounds)
///     params = gs.PyOQNLPParams()
///
///     # Optimize
///     result = gs.optimize(problem, params)
///     best = result.best_solution()
///     print(f"Best: x = {best.x()}, f(x) = {best.fun()}")
///
/// Key Features
/// ------------
/// * **Multiple Solvers**: COBYLA, L-BFGS, Newton-CG, Trust Region, Nelder-Mead, Steepest Descent
/// * **Constraint Support**: Inequality constraints via COBYLA solver
/// * **Builder Pattern**: Flexible solver configuration using builder functions
/// * **Multiple Solutions**: Returns all global minima found
/// * **Early Stopping**: Target objectives and time limits for efficiency
///
/// Main Classes
/// ------------
/// * `PyProblem`: Defines the optimization problem (objective, bounds, constraints)
/// * `PyOQNLPParams`: Controls algorithm behavior (iterations, population size, etc.)
/// * `PySolutionSet`: Contains all solutions found by the optimizer
/// * `PyLocalSolution`: Represents a single solution point and objective value
///
/// Algorithm Reference
/// -------------------
/// Based on: Ugray, Z., Lasdon, L., Plummer, J., Glover, F., Kelly, J., & Martí, R. (2007).
/// "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization."
/// INFORMS Journal on Computing, 19(3), 328-340.
fn pyglobalsearch(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_class::<PyOQNLPParams>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyLocalSolution>()?;
    m.add_class::<PySolutionSet>()?;

    // Builders submodule
    let builders = PyModule::new(_py, "builders")?;
    crate::builders::init_module(_py, &builders)?;
    m.add_submodule(&builders)?;

    // Observers submodule
    let observers = PyModule::new(_py, "observers")?;
    crate::observers::init_module(_py, &observers)?;
    m.add_submodule(&observers)?;

    Ok(())
}
