mod basin;
pub use basin::*;

use globalsearch::local_solver::builders::{COBYLABuilder, TrustRegionRadiusMethod};
use pyo3::prelude::*;

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyTrustRegionRadiusMethod {
    Cauchy,
    Steihaug,
}

#[pymethods]
impl PyTrustRegionRadiusMethod {
    #[staticmethod]
    fn cauchy() -> Self {
        PyTrustRegionRadiusMethod::Cauchy
    }

    #[staticmethod]
    fn steihaug() -> Self {
        PyTrustRegionRadiusMethod::Steihaug
    }
}

impl From<PyTrustRegionRadiusMethod> for TrustRegionRadiusMethod {
    fn from(method: PyTrustRegionRadiusMethod) -> Self {
        match method {
            PyTrustRegionRadiusMethod::Cauchy => TrustRegionRadiusMethod::Cauchy,
            PyTrustRegionRadiusMethod::Steihaug => TrustRegionRadiusMethod::Steihaug,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
/// COBYLA (Constrained Optimization BY Linear Approximations) solver configuration.
///
/// This configuration uses Basin's derivative-free COBYLA implementation, which can
/// handle inequality constraints. It builds linear approximations to the objective function
/// and constraints, making it suitable for problems where gradients are unavailable
/// or unreliable.
///
/// :param max_iter: Maximum number of objective evaluations
/// :type max_iter: int
/// :param step_size: Initial trust region radius
/// :type step_size: float
/// :param ftol_rel: Relative tolerance for function convergence
/// :type ftol_rel: float, optional
/// :param ftol_abs: Absolute tolerance for function convergence
/// :type ftol_abs: float, optional
/// :param xtol_rel: Relative tolerance for parameter convergence
/// :type xtol_rel: float, optional
/// :param xtol_abs: Per-variable absolute tolerances for parameters
/// :type xtol_abs: list[float], optional
///
/// .. rubric:: Key Features
///
/// - No gradient information required
/// - Handles inequality constraints (constraint(x) ≥ 0)
/// - Robust for noisy or discontinuous functions
/// - Good for problems with expensive function evaluations
///
/// .. rubric:: Convergence Criteria
///
/// COBYLA stops when any of these conditions are met:
///
/// - Maximum objective evaluations reached
/// - Function tolerance satisfied
/// - The trust-region radius reaches the resolution derived from the parameter tolerances
///
/// Examples
/// --------
/// Default configuration:
///
/// >>> cobyla_config = gs.builders.cobyla()
///
/// High precision optimization:
///
/// >>> precise = gs.builders.cobyla(
/// ...     max_iter=1000,
/// ...     xtol_abs=[1e-10] * n_vars,  # Very tight parameter tolerance.
/// ... )
///
/// For expensive function evaluations:
///
/// >>> efficient = gs.builders.cobyla(
/// ...     max_iter=100,
/// ...     ftol_rel=1e-4,  # Looser function tolerance.
/// ...     step_size=0.1,  # Smaller initial steps.
/// ... )
pub struct PyCOBYLA {
    #[pyo3(get, set)]
    /// Maximum number of objective evaluations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Initial step size
    ///
    /// :type: float
    pub step_size: f64,

    #[pyo3(get, set)]
    /// Relative tolerance for function value convergence
    ///
    /// :type: float, optional
    pub ftol_rel: Option<f64>,

    #[pyo3(get, set)]
    /// Absolute tolerance for function value convergence
    ///
    /// :type: float, optional
    pub ftol_abs: Option<f64>,

    #[pyo3(get, set)]
    /// Relative tolerance for parameter convergence
    ///
    /// :type: float, optional
    pub xtol_rel: Option<f64>,

    #[pyo3(get, set)]
    /// Per-variable absolute tolerances for parameter convergence
    ///
    /// :type: list[float], optional
    pub xtol_abs: Option<Vec<f64>>,
}

#[pymethods]
impl PyCOBYLA {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        step_size = 1.0,
        ftol_rel = None,
        ftol_abs = None,
        xtol_rel = None,
        xtol_abs = None,
    ))]
    fn new(
        max_iter: u64,
        step_size: f64,
        ftol_rel: Option<f64>,
        ftol_abs: Option<f64>,
        xtol_rel: Option<f64>,
        xtol_abs: Option<Vec<f64>>,
    ) -> Self {
        PyCOBYLA { max_iter, step_size, ftol_rel, ftol_abs, xtol_rel, xtol_abs }
    }
}

impl PyCOBYLA {
    pub fn to_builder(&self) -> COBYLABuilder {
        let mut builder = COBYLABuilder::new(self.max_iter, self.step_size);

        if let Some(ftol_rel) = self.ftol_rel {
            builder = builder.ftol_rel(ftol_rel);
        }
        if let Some(ftol_abs) = self.ftol_abs {
            builder = builder.ftol_abs(ftol_abs);
        }
        if let Some(xtol_rel) = self.xtol_rel {
            builder = builder.xtol_rel(xtol_rel);
        }
        if let Some(xtol_abs) = &self.xtol_abs {
            builder = builder.xtol_abs(xtol_abs.clone());
        }

        builder
    }
}

#[pyfunction]
/// Create a COBYLA solver configuration.
///
/// COBYLA (Constrained Optimization BY Linear Approximations) handles
/// nonlinear/linear constraints without derivatives. It's also an excellent
/// choice for derivative-free optimization. For gradient-based constrained
/// problems, see ``slsqp``, ``barrier``, and ``augmented_lagrangian``.
///
/// :param max_iter: Maximum number of objective evaluations
/// :type max_iter: int
/// :param step_size: Initial trust region radius (larger = more exploration)
/// :type step_size: float
/// :param ftol_rel: Relative tolerance for function value convergence
/// :type ftol_rel: float, optional
/// :param ftol_abs: Absolute tolerance for function value convergence
/// :type ftol_abs: float, optional
/// :param xtol_rel: Relative tolerance for parameter convergence
/// :type xtol_rel: float, optional
/// :param xtol_abs: Per-variable absolute tolerances (length must match problem dimension)
/// :type xtol_abs: list[float], optional
/// :returns: Configured COBYLA solver instance
/// :rtype: PyCOBYLA
///
/// .. note::
///    - If ``xtol_abs`` is provided, its length must match the problem dimension
///    - For constrained problems, use COBYLA (derivative-free), SLSQP (gradient-based),
///      Barrier (linear inequalities), or AugmentedLagrangian (linear equalities)
///    - Larger ``step_size`` values encourage more exploration but may slow convergence.
///
/// Examples
/// --------
/// Default COBYLA (good starting point):
///
/// >>> config = gs.builders.cobyla()
///
/// Conservative settings for reliable convergence:
///
/// >>> config = gs.builders.cobyla(
/// ...     max_iter=1000,
/// ...     step_size=0.1,
/// ...     xtol_abs=[1e-8, 1e-8],  # Same tolerance for both variables.
/// ... )
#[pyo3(signature = (
    max_iter = 300,
    step_size = 1.0,
    ftol_rel = None,
    ftol_abs = None,
    xtol_rel = None,
    xtol_abs = None,
))]
#[pyo3(
    text_signature = "(max_iter: int = 300, step_size: float = 1.0, ftol_rel: Optional[float] = None, ftol_abs: Optional[float] = None, xtol_rel: Optional[float] = None, xtol_abs: Optional[List[float]] = None)"
)]
fn cobyla(
    max_iter: u64,
    step_size: f64,
    ftol_rel: Option<f64>,
    ftol_abs: Option<f64>,
    xtol_rel: Option<f64>,
    xtol_abs: Option<Vec<f64>>,
) -> PyCOBYLA {
    PyCOBYLA { max_iter, step_size, ftol_rel, ftol_abs, xtol_rel, xtol_abs }
}

/// Initialize the builders module
pub fn init_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    basin::register(m)?;

    m.add_class::<PyTrustRegionRadiusMethod>()?;
    m.setattr("TrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;
    m.setattr("PyTrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;

    m.add_class::<PyCOBYLA>()?;
    m.add_function(wrap_pyfunction!(cobyla, m)?)?;
    m.setattr("COBYLA", m.getattr("PyCOBYLA")?)?;

    Ok(())
}
