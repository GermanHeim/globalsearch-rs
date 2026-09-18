//! Python configurations for the Basin-backed local solvers.

use super::PyTrustRegionRadiusMethod;
use globalsearch::local_solver::builders::*;
use pyo3::prelude::*;

/// Unconstrained L-BFGS with the default More-Thuente line search.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLBFGS {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    #[pyo3(get, set)]
    pub tolerance_grad: Option<f64>,
    /// Absolute change in cost between iterates. Disabled by default.
    #[pyo3(get, set)]
    pub tolerance_cost: Option<f64>,
    /// Number of correction pairs retained by L-BFGS. Must be positive.
    #[pyo3(get, set)]
    pub history_size: usize,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
    fn new(
        max_iter: u64,
        tolerance_grad: Option<f64>,
        tolerance_cost: Option<f64>,
        history_size: usize,
    ) -> Self {
        Self { max_iter, tolerance_grad, tolerance_cost, history_size }
    }
}

impl PyLBFGS {
    pub fn to_builder(&self) -> LBFGSBuilder {
        LBFGSBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .tolerance_cost(self.tolerance_cost)
            .history_size(self.history_size)
    }
}

/// Unconstrained L-BFGS with the default More-Thuente line search.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
pub fn lbfgs(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
) -> PyLBFGS {
    PyLBFGS::new(max_iter, tolerance_grad, tolerance_cost, history_size)
}

/// Unconstrained gradient descent with the default More-Thuente line search and no momentum.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyGradientDescent {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    #[pyo3(get, set)]
    pub tolerance_grad: Option<f64>,
    /// Absolute change in cost between iterates. Disabled by default.
    #[pyo3(get, set)]
    pub tolerance_cost: Option<f64>,
}

#[pymethods]
impl PyGradientDescent {
    #[new]
    #[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None))]
    fn new(max_iter: u64, tolerance_grad: Option<f64>, tolerance_cost: Option<f64>) -> Self {
        Self { max_iter, tolerance_grad, tolerance_cost }
    }
}

impl PyGradientDescent {
    pub fn to_builder(&self) -> GradientDescentBuilder {
        GradientDescentBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Unconstrained gradient descent with the default More-Thuente line search and no momentum.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None))]
pub fn gradient_descent(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyGradientDescent {
    PyGradientDescent::new(max_iter, tolerance_grad, tolerance_cost)
}

/// Unconstrained trust-region optimization using the supplied gradient and Hessian.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyTrustRegion {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    #[pyo3(get, set)]
    pub tolerance_grad: Option<f64>,
    /// Trust-region subproblem method, defaulting to Steihaug.
    #[pyo3(get, set)]
    pub trust_region_radius_method: PyTrustRegionRadiusMethod,
    /// Positive initial trust-region radius.
    #[pyo3(get, set)]
    pub radius: f64,
    /// Maximum trust-region radius, at least the initial radius.
    #[pyo3(get, set)]
    pub max_radius: f64,
    /// Step acceptance threshold, in [0, 0.25).
    #[pyo3(get, set)]
    pub eta: f64,
}

#[pymethods]
impl PyTrustRegion {
    #[new]
    #[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), trust_region_radius_method = PyTrustRegionRadiusMethod::Steihaug, radius = 1.0, max_radius = 100.0, eta = 0.125))]
    fn new(
        max_iter: u64,
        tolerance_grad: Option<f64>,
        trust_region_radius_method: PyTrustRegionRadiusMethod,
        radius: f64,
        max_radius: f64,
        eta: f64,
    ) -> Self {
        Self { max_iter, tolerance_grad, trust_region_radius_method, radius, max_radius, eta }
    }
}

impl PyTrustRegion {
    pub fn to_builder(&self) -> TrustRegionBuilder {
        TrustRegionBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .method(self.trust_region_radius_method.clone().into())
            .radius(self.radius)
            .max_radius(self.max_radius)
            .eta(self.eta)
    }
}

/// Unconstrained trust-region optimization using the supplied gradient and Hessian.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), trust_region_radius_method = PyTrustRegionRadiusMethod::Steihaug, radius = 1.0, max_radius = 100.0, eta = 0.125))]
pub fn trust_region(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    trust_region_radius_method: PyTrustRegionRadiusMethod,
    radius: f64,
    max_radius: f64,
    eta: f64,
) -> PyTrustRegion {
    PyTrustRegion::new(
        max_iter,
        tolerance_grad,
        trust_region_radius_method,
        radius,
        max_radius,
        eta,
    )
}

/// Unconstrained Nelder-Mead with standard coefficients.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyNelderMead {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Positive absolute coordinate step for the initial simplex.
    #[pyo3(get, set)]
    pub simplex_delta: f64,
    /// Maximum simplex distance from its best vertex in the infinity norm.
    #[pyo3(get, set)]
    pub tolerance_simplex: Option<f64>,
    /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
    #[pyo3(get, set)]
    pub tolerance_cost: Option<f64>,
}

#[pymethods]
impl PyNelderMead {
    #[new]
    #[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
    fn new(
        max_iter: u64,
        simplex_delta: f64,
        tolerance_simplex: Option<f64>,
        tolerance_cost: Option<f64>,
    ) -> Self {
        Self { max_iter, simplex_delta, tolerance_simplex, tolerance_cost }
    }
}

impl PyNelderMead {
    pub fn to_builder(&self) -> NelderMeadBuilder {
        NelderMeadBuilder::default()
            .max_iter(self.max_iter)
            .simplex_delta(self.simplex_delta)
            .tolerance_simplex(self.tolerance_simplex)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Unconstrained Nelder-Mead with standard coefficients.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
pub fn nelder_mead(
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyNelderMead {
    PyNelderMead::new(max_iter, simplex_delta, tolerance_simplex, tolerance_cost)
}

/// Box-constrained L-BFGS-B with the default More-Thuente line search.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLBFGSB {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Absolute projected-gradient infinity-norm tolerance. None disables.
    #[pyo3(get, set)]
    pub tolerance_projected_grad: Option<f64>,
    /// Absolute change in cost between iterates. Disabled by default.
    #[pyo3(get, set)]
    pub tolerance_cost: Option<f64>,
    /// Number of correction pairs retained by L-BFGS. Must be positive.
    #[pyo3(get, set)]
    pub history_size: usize,
}

#[pymethods]
impl PyLBFGSB {
    #[new]
    #[pyo3(signature = (max_iter = 1000, tolerance_projected_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
    fn new(
        max_iter: u64,
        tolerance_projected_grad: Option<f64>,
        tolerance_cost: Option<f64>,
        history_size: usize,
    ) -> Self {
        Self { max_iter, tolerance_projected_grad, tolerance_cost, history_size }
    }
}

impl PyLBFGSB {
    pub fn to_builder(&self) -> LBFGSBBuilder {
        LBFGSBBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_projected_grad(self.tolerance_projected_grad)
            .tolerance_cost(self.tolerance_cost)
            .history_size(self.history_size)
    }
}

/// Box-constrained L-BFGS-B with the default More-Thuente line search.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_projected_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
pub fn lbfgsb(
    max_iter: u64,
    tolerance_projected_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
) -> PyLBFGSB {
    PyLBFGSB::new(max_iter, tolerance_projected_grad, tolerance_cost, history_size)
}

/// Box-constrained Nelder-Mead with projected trial vertices and standard coefficients.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBoundedNelderMead {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Positive absolute coordinate step for the initial simplex.
    #[pyo3(get, set)]
    pub simplex_delta: f64,
    /// Maximum simplex distance from its best vertex in the infinity norm.
    #[pyo3(get, set)]
    pub tolerance_simplex: Option<f64>,
    /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
    #[pyo3(get, set)]
    pub tolerance_cost: Option<f64>,
}

#[pymethods]
impl PyBoundedNelderMead {
    #[new]
    #[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
    fn new(
        max_iter: u64,
        simplex_delta: f64,
        tolerance_simplex: Option<f64>,
        tolerance_cost: Option<f64>,
    ) -> Self {
        Self { max_iter, simplex_delta, tolerance_simplex, tolerance_cost }
    }
}

impl PyBoundedNelderMead {
    pub fn to_builder(&self) -> BoundedNelderMeadBuilder {
        BoundedNelderMeadBuilder::default()
            .max_iter(self.max_iter)
            .simplex_delta(self.simplex_delta)
            .tolerance_simplex(self.tolerance_simplex)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Box-constrained Nelder-Mead with projected trial vertices and standard coefficients.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
pub fn bounded_nelder_mead(
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyBoundedNelderMead {
    PyBoundedNelderMead::new(max_iter, simplex_delta, tolerance_simplex, tolerance_cost)
}

/// Box-constrained BOBYQA. Basin reduces the radii automatically for narrow boxes.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBOBYQA {
    /// Maximum number of basin executor iterations, excluding initialization.
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Positive initial trust-region radius, larger than the final radius.
    #[pyo3(get, set)]
    pub initial_radius: f64,
    /// Positive final trust-region radius.
    #[pyo3(get, set)]
    pub final_radius: f64,
    /// Interpolation-set size, between 2n+1 and (n+1)(n+2)/2. None selects 2n+1.
    #[pyo3(get, set)]
    pub interpolation_points: Option<usize>,
}

#[pymethods]
impl PyBOBYQA {
    #[new]
    #[pyo3(signature = (max_iter = 1000, initial_radius = 1.0, final_radius = 1e-6, interpolation_points = None))]
    fn new(
        max_iter: u64,
        initial_radius: f64,
        final_radius: f64,
        interpolation_points: Option<usize>,
    ) -> Self {
        Self { max_iter, initial_radius, final_radius, interpolation_points }
    }
}

impl PyBOBYQA {
    pub fn to_builder(&self) -> BOBYQABuilder {
        BOBYQABuilder::default()
            .max_iter(self.max_iter)
            .initial_radius(self.initial_radius)
            .final_radius(self.final_radius)
            .interpolation_points(self.interpolation_points)
    }
}

/// Box-constrained BOBYQA. Basin reduces the radii automatically for narrow boxes.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, initial_radius = 1.0, final_radius = 1e-6, interpolation_points = None))]
pub fn bobyqa(
    max_iter: u64,
    initial_radius: f64,
    final_radius: f64,
    interpolation_points: Option<usize>,
) -> PyBOBYQA {
    PyBOBYQA::new(max_iter, initial_radius, final_radius, interpolation_points)
}

/// Gradient-based SLSQP with box, linear, and nonlinear constraints.
///
/// Requires `gradient` and `constraint_jacobian` on the problem.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PySLSQP {
    /// Maximum number of executor iterations (outer SLSQP steps).
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Kraft composite accuracy tolerance. None disables convergence tests.
    #[pyo3(get, set)]
    pub accuracy: Option<f64>,
    /// NNLS active-set iteration limit per QP subproblem. None selects default.
    #[pyo3(get, set)]
    pub max_subproblem_iter: Option<usize>,
}

#[pymethods]
impl PySLSQP {
    #[new]
    #[pyo3(signature = (max_iter = 1000, accuracy = Some(1e-6), max_subproblem_iter = None))]
    fn new(max_iter: u64, accuracy: Option<f64>, max_subproblem_iter: Option<usize>) -> Self {
        Self { max_iter, accuracy, max_subproblem_iter }
    }
}

impl PySLSQP {
    pub fn to_builder(&self) -> SLSQPBuilder {
        SLSQPBuilder::default()
            .max_iter(self.max_iter)
            .accuracy(self.accuracy)
            .max_subproblem_iter(self.max_subproblem_iter)
    }
}

/// Gradient-based SLSQP with box, linear, and nonlinear constraints.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, accuracy = Some(1e-6), max_subproblem_iter = None))]
pub fn slsqp(max_iter: u64, accuracy: Option<f64>, max_subproblem_iter: Option<usize>) -> PySLSQP {
    PySLSQP::new(max_iter, accuracy, max_subproblem_iter)
}

/// Log-barrier method over a BFGS inner solver for linear inequalities `A x <= b`.
///
/// Requires `gradient` and `linear_inequalities` on the problem. Box bounds
/// are folded into the inequality system and enforced exactly.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBarrier {
    /// Maximum number of outer barrier iterations (safety budget).
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Initial barrier parameter. Must be positive.
    #[pyo3(get, set)]
    pub mu0: f64,
    /// Per-iteration shrink factor. Must exceed 1.
    #[pyo3(get, set)]
    pub reduction: f64,
    /// Outer duality-gap tolerance: stop once `m * mu <= tol`.
    #[pyo3(get, set)]
    pub duality_gap_tol: f64,
    /// Iteration budget for each inner barrier-subproblem solve.
    #[pyo3(get, set)]
    pub inner_max_iter: u64,
}

#[pymethods]
impl PyBarrier {
    #[new]
    #[pyo3(signature = (max_iter = 100, mu0 = 1.0, reduction = 10.0, duality_gap_tol = 1e-8, inner_max_iter = 50))]
    fn new(
        max_iter: u64,
        mu0: f64,
        reduction: f64,
        duality_gap_tol: f64,
        inner_max_iter: u64,
    ) -> Self {
        Self { max_iter, mu0, reduction, duality_gap_tol, inner_max_iter }
    }
}

impl PyBarrier {
    pub fn to_builder(&self) -> BarrierBuilder {
        BarrierBuilder::default()
            .max_iter(self.max_iter)
            .mu0(self.mu0)
            .reduction(self.reduction)
            .duality_gap_tol(self.duality_gap_tol)
            .inner_max_iter(self.inner_max_iter)
    }
}

/// Log-barrier method over a BFGS inner solver for linear inequalities `A x <= b`.
#[pyfunction]
#[pyo3(signature = (max_iter = 100, mu0 = 1.0, reduction = 10.0, duality_gap_tol = 1e-8, inner_max_iter = 50))]
pub fn barrier(
    max_iter: u64,
    mu0: f64,
    reduction: f64,
    duality_gap_tol: f64,
    inner_max_iter: u64,
) -> PyBarrier {
    PyBarrier::new(max_iter, mu0, reduction, duality_gap_tol, inner_max_iter)
}

/// Augmented-Lagrangian method over a BFGS inner solver for linear equalities `A x = b`.
///
/// Requires `gradient` and `linear_equalities` on the problem. Box bounds
/// are not enforced during local search (as with the unconstrained solvers);
/// use SLSQP when box and equalities must both hold strictly.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyAugmentedLagrangian {
    /// Maximum number of outer iterations (safety budget).
    #[pyo3(get, set)]
    pub max_iter: u64,
    /// Initial penalty parameter. Must be positive.
    #[pyo3(get, set)]
    pub rho0: f64,
    /// Penalty growth factor when feasibility stalls. Must exceed 1.
    #[pyo3(get, set)]
    pub rho_increase: f64,
    /// Required feasibility-decrease ratio in (0, 1) for multiplier updates.
    #[pyo3(get, set)]
    pub feasibility_decrease: f64,
    /// Outer feasibility tolerance: stop once `||A x - b|| <= tol`.
    #[pyo3(get, set)]
    pub feasibility_tol: f64,
    /// Iteration budget for each inner subproblem solve.
    #[pyo3(get, set)]
    pub inner_max_iter: u64,
}

#[pymethods]
impl PyAugmentedLagrangian {
    #[new]
    #[pyo3(signature = (max_iter = 100, rho0 = 10.0, rho_increase = 10.0, feasibility_decrease = 0.25, feasibility_tol = 1e-8, inner_max_iter = 50))]
    fn new(
        max_iter: u64,
        rho0: f64,
        rho_increase: f64,
        feasibility_decrease: f64,
        feasibility_tol: f64,
        inner_max_iter: u64,
    ) -> Self {
        Self { max_iter, rho0, rho_increase, feasibility_decrease, feasibility_tol, inner_max_iter }
    }
}

impl PyAugmentedLagrangian {
    pub fn to_builder(&self) -> AugmentedLagrangianBuilder {
        AugmentedLagrangianBuilder::default()
            .max_iter(self.max_iter)
            .rho0(self.rho0)
            .rho_increase(self.rho_increase)
            .feasibility_decrease(self.feasibility_decrease)
            .feasibility_tol(self.feasibility_tol)
            .inner_max_iter(self.inner_max_iter)
    }
}

/// Augmented-Lagrangian method over a BFGS inner solver for linear equalities `A x = b`.
#[pyfunction]
#[pyo3(signature = (max_iter = 100, rho0 = 10.0, rho_increase = 10.0, feasibility_decrease = 0.25, feasibility_tol = 1e-8, inner_max_iter = 50))]
pub fn augmented_lagrangian(
    max_iter: u64,
    rho0: f64,
    rho_increase: f64,
    feasibility_decrease: f64,
    feasibility_tol: f64,
    inner_max_iter: u64,
) -> PyAugmentedLagrangian {
    PyAugmentedLagrangian::new(
        max_iter,
        rho0,
        rho_increase,
        feasibility_decrease,
        feasibility_tol,
        inner_max_iter,
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLBFGS>()?;
    m.add_function(wrap_pyfunction!(lbfgs, m)?)?;
    m.add_class::<PyGradientDescent>()?;
    m.add_function(wrap_pyfunction!(gradient_descent, m)?)?;
    m.add_class::<PyTrustRegion>()?;
    m.add_function(wrap_pyfunction!(trust_region, m)?)?;
    m.add_class::<PyNelderMead>()?;
    m.add_function(wrap_pyfunction!(nelder_mead, m)?)?;
    m.add_class::<PyLBFGSB>()?;
    m.add_function(wrap_pyfunction!(lbfgsb, m)?)?;
    m.add_class::<PyBoundedNelderMead>()?;
    m.add_function(wrap_pyfunction!(bounded_nelder_mead, m)?)?;
    m.add_class::<PyBOBYQA>()?;
    m.add_function(wrap_pyfunction!(bobyqa, m)?)?;
    m.add_class::<PySLSQP>()?;
    m.add_function(wrap_pyfunction!(slsqp, m)?)?;
    m.add_class::<PyBarrier>()?;
    m.add_function(wrap_pyfunction!(barrier, m)?)?;
    m.add_class::<PyAugmentedLagrangian>()?;
    m.add_function(wrap_pyfunction!(augmented_lagrangian, m)?)?;
    Ok(())
}
