//! Python configurations for the optional basin solvers.

use super::PyTrustRegionRadiusMethod;
use globalsearch::local_solver::builders::*;
use pyo3::prelude::*;

/// Unconstrained basin L-BFGS with the default More-Thuente line search.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinLBFGS {
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
impl PyBasinLBFGS {
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

impl PyBasinLBFGS {
    pub fn to_builder(&self) -> BasinLBFGSBuilder {
        BasinLBFGSBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .tolerance_cost(self.tolerance_cost)
            .history_size(self.history_size)
    }
}

/// Unconstrained basin L-BFGS with the default More-Thuente line search.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
pub fn basin_lbfgs(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
) -> PyBasinLBFGS {
    PyBasinLBFGS::new(max_iter, tolerance_grad, tolerance_cost, history_size)
}

/// Unconstrained basin gradient descent with the default More-Thuente line search and no momentum.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinGradientDescent {
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
impl PyBasinGradientDescent {
    #[new]
    #[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None))]
    fn new(max_iter: u64, tolerance_grad: Option<f64>, tolerance_cost: Option<f64>) -> Self {
        Self { max_iter, tolerance_grad, tolerance_cost }
    }
}

impl PyBasinGradientDescent {
    pub fn to_builder(&self) -> BasinGradientDescentBuilder {
        BasinGradientDescentBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Unconstrained basin gradient descent with the default More-Thuente line search and no momentum.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), tolerance_cost = None))]
pub fn basin_gradient_descent(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyBasinGradientDescent {
    PyBasinGradientDescent::new(max_iter, tolerance_grad, tolerance_cost)
}

/// Unconstrained basin trust-region optimization using the supplied gradient and Hessian.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinTrustRegion {
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
impl PyBasinTrustRegion {
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

impl PyBasinTrustRegion {
    pub fn to_builder(&self) -> BasinTrustRegionBuilder {
        BasinTrustRegionBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_grad(self.tolerance_grad)
            .method(self.trust_region_radius_method.clone().into())
            .radius(self.radius)
            .max_radius(self.max_radius)
            .eta(self.eta)
    }
}

/// Unconstrained basin trust-region optimization using the supplied gradient and Hessian.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_grad = Some(1e-6), trust_region_radius_method = PyTrustRegionRadiusMethod::Steihaug, radius = 1.0, max_radius = 100.0, eta = 0.125))]
pub fn basin_trust_region(
    max_iter: u64,
    tolerance_grad: Option<f64>,
    trust_region_radius_method: PyTrustRegionRadiusMethod,
    radius: f64,
    max_radius: f64,
    eta: f64,
) -> PyBasinTrustRegion {
    PyBasinTrustRegion::new(
        max_iter,
        tolerance_grad,
        trust_region_radius_method,
        radius,
        max_radius,
        eta,
    )
}

/// Unconstrained basin Nelder-Mead with standard coefficients.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinNelderMead {
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
impl PyBasinNelderMead {
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

impl PyBasinNelderMead {
    pub fn to_builder(&self) -> BasinNelderMeadBuilder {
        BasinNelderMeadBuilder::default()
            .max_iter(self.max_iter)
            .simplex_delta(self.simplex_delta)
            .tolerance_simplex(self.tolerance_simplex)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Unconstrained basin Nelder-Mead with standard coefficients.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
pub fn basin_nelder_mead(
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyBasinNelderMead {
    PyBasinNelderMead::new(max_iter, simplex_delta, tolerance_simplex, tolerance_cost)
}

/// Box-constrained basin L-BFGS-B with the default More-Thuente line search.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinLBFGSB {
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
impl PyBasinLBFGSB {
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

impl PyBasinLBFGSB {
    pub fn to_builder(&self) -> BasinLBFGSBBuilder {
        BasinLBFGSBBuilder::default()
            .max_iter(self.max_iter)
            .tolerance_projected_grad(self.tolerance_projected_grad)
            .tolerance_cost(self.tolerance_cost)
            .history_size(self.history_size)
    }
}

/// Box-constrained basin L-BFGS-B with the default More-Thuente line search.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, tolerance_projected_grad = Some(1e-6), tolerance_cost = None, history_size = 10))]
pub fn basin_lbfgsb(
    max_iter: u64,
    tolerance_projected_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
) -> PyBasinLBFGSB {
    PyBasinLBFGSB::new(max_iter, tolerance_projected_grad, tolerance_cost, history_size)
}

/// Box-constrained basin Nelder-Mead with projected trial vertices and standard coefficients.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinBoundedNelderMead {
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
impl PyBasinBoundedNelderMead {
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

impl PyBasinBoundedNelderMead {
    pub fn to_builder(&self) -> BasinBoundedNelderMeadBuilder {
        BasinBoundedNelderMeadBuilder::default()
            .max_iter(self.max_iter)
            .simplex_delta(self.simplex_delta)
            .tolerance_simplex(self.tolerance_simplex)
            .tolerance_cost(self.tolerance_cost)
    }
}

/// Box-constrained basin Nelder-Mead with projected trial vertices and standard coefficients.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, simplex_delta = 0.1, tolerance_simplex = Some(1e-6), tolerance_cost = Some(1e-8)))]
pub fn basin_bounded_nelder_mead(
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
) -> PyBasinBoundedNelderMead {
    PyBasinBoundedNelderMead::new(max_iter, simplex_delta, tolerance_simplex, tolerance_cost)
}

/// Box-constrained basin BOBYQA. Basin reduces the radii automatically for narrow boxes.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyBasinBOBYQA {
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
impl PyBasinBOBYQA {
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

impl PyBasinBOBYQA {
    pub fn to_builder(&self) -> BasinBOBYQABuilder {
        BasinBOBYQABuilder::default()
            .max_iter(self.max_iter)
            .initial_radius(self.initial_radius)
            .final_radius(self.final_radius)
            .interpolation_points(self.interpolation_points)
    }
}

/// Box-constrained basin BOBYQA. Basin reduces the radii automatically for narrow boxes.
#[pyfunction]
#[pyo3(signature = (max_iter = 1000, initial_radius = 1.0, final_radius = 1e-6, interpolation_points = None))]
pub fn basin_bobyqa(
    max_iter: u64,
    initial_radius: f64,
    final_radius: f64,
    interpolation_points: Option<usize>,
) -> PyBasinBOBYQA {
    PyBasinBOBYQA::new(max_iter, initial_radius, final_radius, interpolation_points)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasinLBFGS>()?;
    m.add_function(wrap_pyfunction!(basin_lbfgs, m)?)?;
    m.add_class::<PyBasinGradientDescent>()?;
    m.add_function(wrap_pyfunction!(basin_gradient_descent, m)?)?;
    m.add_class::<PyBasinTrustRegion>()?;
    m.add_function(wrap_pyfunction!(basin_trust_region, m)?)?;
    m.add_class::<PyBasinNelderMead>()?;
    m.add_function(wrap_pyfunction!(basin_nelder_mead, m)?)?;
    m.add_class::<PyBasinLBFGSB>()?;
    m.add_function(wrap_pyfunction!(basin_lbfgsb, m)?)?;
    m.add_class::<PyBasinBoundedNelderMead>()?;
    m.add_function(wrap_pyfunction!(basin_bounded_nelder_mead, m)?)?;
    m.add_class::<PyBasinBOBYQA>()?;
    m.add_function(wrap_pyfunction!(basin_bobyqa, m)?)?;
    Ok(())
}
