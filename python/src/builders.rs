use globalsearch::local_solver::builders::{
    COBYLABuilder, HagerZhangBuilder, LBFGSBuilder, LineSearchParams, MoreThuenteBuilder,
    NelderMeadBuilder, NewtonCGBuilder, SteepestDescentBuilder, TrustRegionBuilder,
    TrustRegionRadiusMethod,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::f64::EPSILON;
use std::f64::INFINITY;

#[pyclass]
#[derive(Debug, Clone)]
/// Hager-Zhang line search method configuration.
///
/// The Hager-Zhang line search is a sophisticated line search algorithm that
/// satisfies both Wolfe conditions and provides strong theoretical guarantees.
/// It's particularly effective for L-BFGS and other quasi-Newton methods.
///
/// :param delta: Armijo parameter for sufficient decrease condition
/// :type delta: float
/// :param sigma: Wolfe parameter for curvature condition
/// :type sigma: float
/// :param epsilon: Tolerance for approximate Wolfe conditions
/// :type epsilon: float
/// :param theta: Parameter for bracketing phase
/// :type theta: float
/// :param gamma: Parameter for update rules
/// :type gamma: float
/// :param eta: Parameter for switching conditions
/// :type eta: float
/// :param bounds: Step length bounds [min, max]
/// :type bounds: list[float]
///
/// Examples
/// --------
/// Default parameters (recommended for most problems):
///
/// >>> hz_config = gs.builders.hagerzhang()
///
/// Conservative line search (more function evaluations, more reliable):
///
/// >>> conservative = gs.builders.hagerzhang(delta=0.01, sigma=0.99)
///
/// Aggressive line search (fewer evaluations, less reliable):
///
/// >>> aggressive = gs.builders.hagerzhang(delta=0.3, sigma=0.7)
///
/// Use with L-BFGS:
///
/// >>> lbfgs_config = gs.builders.lbfgs(line_search_params=hz_config)
pub struct PyHagerZhang {
    #[pyo3(get, set)]
    /// Constant C1 of the strong Wolfe conditions
    pub delta: f64,
    #[pyo3(get, set)]
    /// Constant C2 of the strong Wolfe conditions
    pub sigma: f64,
    #[pyo3(get, set)]
    /// Parameter for approximate Wolfe conditions
    pub epsilon: f64,
    #[pyo3(get, set)]
    /// Parameter used in the update rules when the potential intervals [a, c] or [c, b] violate the opposite slope condition.
    pub theta: f64,
    #[pyo3(get, set)]
    /// Parameter that determines when a bisection step is performed.
    pub gamma: f64,
    #[pyo3(get, set)]
    /// Used in the lower bound for beta_k^N.
    pub eta: f64,
    #[pyo3(get, set)]
    /// Set lower and upper bound of step
    pub bounds: Vec<f64>,
}

#[pymethods]
impl PyHagerZhang {
    #[new]
    #[pyo3(signature = (
        delta = 0.1,
        sigma = 0.9,
        epsilon = 1e-6,
        theta = 0.5,
        gamma = 0.66,
        eta = 0.01,
        bounds = vec![EPSILON.sqrt(), INFINITY],
    ))]
    fn new(
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Vec<f64>,
    ) -> Self {
        PyHagerZhang { delta, sigma, epsilon, theta, gamma, eta, bounds }
    }
}

impl PyHagerZhang {
    pub fn to_builder(&self) -> HagerZhangBuilder {
        HagerZhangBuilder::new(
            self.delta,
            self.sigma,
            self.epsilon,
            self.theta,
            self.gamma,
            self.eta,
            self.bounds.clone().into(),
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(delta: f64, sigma: f64, epsilon: f64, theta: f64, gamma: f64, eta: f64, bounds: List[float])"
)]
fn hagerzhang(
    delta: f64,
    sigma: f64,
    epsilon: f64,
    theta: f64,
    gamma: f64,
    eta: f64,
    bounds: Vec<f64>,
) -> PyHagerZhang {
    PyHagerZhang { delta, sigma, epsilon, theta, gamma, eta, bounds }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyMoreThuente {
    #[pyo3(get, set)]
    /// Constant C1 of the strong Wolfe conditions
    pub c1: f64,
    #[pyo3(get, set)]
    /// Constant C2 of the strong Wolfe conditions
    pub c2: f64,
    #[pyo3(get, set)]
    /// Parameter for approximate Wolfe conditions
    pub width_tolerance: f64,
    #[pyo3(get, set)]
    /// Set lower and upper bound of step
    pub bounds: Vec<f64>,
}

#[pymethods]
impl PyMoreThuente {
    #[new]
    #[pyo3(signature = (
        c1 = 1e-4,
        c2 = 0.9,
        width_tolerance = 1e-10,
        bounds = vec![EPSILON.sqrt(), INFINITY],
    ))]
    fn new(c1: f64, c2: f64, width_tolerance: f64, bounds: Vec<f64>) -> Self {
        PyMoreThuente { c1, c2, width_tolerance, bounds }
    }
}

impl PyMoreThuente {
    pub fn to_builder(&self) -> MoreThuenteBuilder {
        MoreThuenteBuilder::new(self.c1, self.c2, self.width_tolerance, self.bounds.clone().into())
    }
}

#[pyfunction]
#[pyo3(text_signature = "(c1: f64, c2: f64, width_tolerance: f64, bounds: List[float])")]
fn morethuente(c1: f64, c2: f64, width_tolerance: f64, bounds: Vec<f64>) -> PyMoreThuente {
    PyMoreThuente { c1, c2, width_tolerance, bounds }
}

#[derive(Debug, Clone)]
pub enum PyLineSearchMethod {
    MoreThuente(PyMoreThuente),
    HagerZhang(PyHagerZhang),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyLineSearchParams {
    pub method: PyLineSearchMethod,
}

#[pymethods]
impl PyLineSearchParams {
    #[new]
    #[pyo3(signature = (method))]
    fn new(method: Py<pyo3::PyAny>, py: Python) -> PyResult<Self> {
        if let Ok(more_thuente) = method.extract::<PyMoreThuente>(py) {
            return Ok(PyLineSearchParams {
                method: PyLineSearchMethod::MoreThuente(more_thuente),
            });
        }

        if let Ok(hager_zhang) = method.extract::<PyHagerZhang>(py) {
            return Ok(PyLineSearchParams { method: PyLineSearchMethod::HagerZhang(hager_zhang) });
        }

        Err(PyTypeError::new_err("Expected PyMoreThuente or PyHagerZhang"))
    }

    #[staticmethod]
    /// More-Thuente line search configuration
    ///
    /// :param params: More-Thuente line search parameters
    fn morethuente(params: PyMoreThuente) -> Self {
        PyLineSearchParams { method: PyLineSearchMethod::MoreThuente(params) }
    }

    #[staticmethod]
    /// Hager-Zhang line search configuration
    ///
    /// :param params: Hager-Zhang line search parameters
    fn hagerzhang(params: PyHagerZhang) -> Self {
        PyLineSearchParams { method: PyLineSearchMethod::HagerZhang(params) }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
/// L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) solver configuration.
///
/// L-BFGS is a quasi-Newton optimization algorithm that approximates the inverse
/// Hessian using only gradient information and a limited history of previous steps.
/// It's one of the most effective algorithms for smooth, unconstrained optimization.
///
/// :param max_iter: Maximum number of iterations
/// :type max_iter: int
/// :param tolerance_grad: Gradient norm tolerance for convergence
/// :type tolerance_grad: float
/// :param tolerance_cost: Relative function change tolerance
/// :type tolerance_cost: float
/// :param history_size: Number of previous steps to store
/// :type history_size: int
/// :param l1_coefficient: L1 regularization coefficient for sparsity
/// :type l1_coefficient: float, optional
/// :param line_search_params: Line search method configuration
/// :type line_search_params: PyLineSearchParams
///
/// .. rubric:: Key Features
///
/// - Requires only gradient information (no Hessian)
/// - Superlinear convergence near the optimum
/// - Memory-efficient (stores only `m` previous steps)
/// - Excellent for large-scale optimization
///
/// .. rubric:: Convergence Criteria
///
/// L-BFGS stops when:
///
/// - ||∇f(x)|| < tolerance_grad (gradient norm is small)
/// - \|f_new - f_old\| / max(\|f_new\|, \|f_old\|, 1) < tolerance_cost
/// - Maximum iterations reached
///
/// Examples
/// --------
/// Default configuration (good for most problems):
///
/// >>> lbfgs_config = gs.builders.lbfgs()
///
/// High precision optimization:
///
/// >>> precise = gs.builders.lbfgs(
/// ...     tolerance_grad=1e-12,
/// ...     max_iter=1000
/// ... )
///
/// Large-scale problems (more history for better approximation):
///
/// >>> large_scale = gs.builders.lbfgs(
/// ...     history_size=20,
/// ...     line_search_params=gs.builders.hagerzhang()
/// ... )
///
/// Sparse optimization with L1 regularization:
///
/// >>> sparse = gs.builders.lbfgs(
/// ...     l1_coefficient=0.01,  # Promotes sparsity
/// ...     tolerance_grad=1e-6
/// ... )
///
/// Conservative line search for difficult problems:
///
/// >>> robust = gs.builders.lbfgs(
/// ...     line_search_params=gs.builders.morethuente(c1=1e-6, c2=0.99)
/// ... )
pub struct PyLBFGS {
    #[pyo3(get, set)]
    /// Maximum number of iterations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Gradient norm tolerance for convergence
    ///
    /// :type: float
    pub tolerance_grad: f64,

    #[pyo3(get, set)]
    /// Relative function change tolerance
    ///
    /// :type: float
    pub tolerance_cost: f64,

    #[pyo3(get, set)]
    /// Number of previous steps to store
    ///
    /// :type: int
    pub history_size: usize,
    #[pyo3(get, set)]
    /// L1 regularization coefficient for sparsity
    ///
    /// :type: float, optional
    pub l1_coefficient: Option<f64>,

    #[pyo3(get, set)]
    /// Line search method configuration
    ///
    /// :type: PyLineSearchParams
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        tolerance_grad = EPSILON.sqrt(),
        tolerance_cost = EPSILON,
        history_size = 10,
        l1_coefficient = None,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(
        max_iter: u64,
        tolerance_grad: f64,
        tolerance_cost: f64,
        history_size: usize,
        l1_coefficient: Option<f64>,
        line_search_params: PyLineSearchParams,
    ) -> Self {
        PyLBFGS {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            l1_coefficient,
            line_search_params,
        }
    }
}

impl PyLBFGS {
    pub fn to_builder(&self) -> LBFGSBuilder {
        let line_search_params = match &self.line_search_params.method {
            PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                .c1(params.c1)
                .c2(params.c2)
                .width_tolerance(params.width_tolerance)
                .bounds(params.bounds.clone().into())
                .build(),
            PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                .delta(params.delta)
                .sigma(params.sigma)
                .epsilon(params.epsilon)
                .theta(params.theta)
                .gamma(params.gamma)
                .eta(params.eta)
                .bounds(params.bounds.clone().into())
                .build(),
        };

        LBFGSBuilder::new(
            self.max_iter,
            self.tolerance_grad,
            self.tolerance_cost,
            self.history_size,
            self.l1_coefficient,
            line_search_params,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    max_iter,
    tolerance_grad,
    tolerance_cost,
    history_size,
    line_search_params,
    l1_coefficient = None,
))]
#[pyo3(
    text_signature = "(max_iter: u64, tolerance_grad: f64, tolerance_cost: f64, history_size: usize, line_search_params: Union[PyMoreThuente, PyHagerZhang], l1_coefficient: Optional[float] = None)"
)]
fn lbfgs(
    max_iter: u64,
    tolerance_grad: f64,
    tolerance_cost: f64,
    history_size: usize,
    line_search_params: Py<pyo3::PyAny>,
    l1_coefficient: Option<f64>,
    py: Python,
) -> PyResult<PyLBFGS> {
    let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::MoreThuente(more_thuente) }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::HagerZhang(hager_zhang) }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PyLBFGS {
        max_iter,
        tolerance_grad,
        tolerance_cost,
        history_size,
        l1_coefficient,
        line_search_params,
    })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNelderMead {
    #[pyo3(get, set)]
    /// Simplex delta configuration
    ///
    /// :type: float
    pub simplex_delta: f64,

    #[pyo3(get, set)]
    /// Standard deviation tolerance for convergence
    ///
    /// :type: float
    pub sd_tolerance: f64,
    #[pyo3(get, set)]
    /// Maximum number of iterations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Step size for the simplex algorithm
    ///
    /// :type: float
    pub alpha: f64,
    #[pyo3(get, set)]
    /// Reflection coefficient
    ///
    /// :type: float
    pub gamma: f64,
    #[pyo3(get, set)]
    /// Contraction coefficient
    ///
    /// :type: float
    pub rho: f64,
    #[pyo3(get, set)]
    /// Expansion coefficient
    ///
    /// :type: float
    pub sigma: f64,
}

#[pymethods]
impl PyNelderMead {
    #[new]
    #[pyo3(signature = (
        simplex_delta = 0.1,
        sd_tolerance = EPSILON,
        max_iter = 300,
        alpha = 1.0,
        gamma = 2.0,
        rho = 0.5,
        sigma = 0.5,
    ))]
    fn new(
        simplex_delta: f64,
        sd_tolerance: f64,
        max_iter: u64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Self {
        PyNelderMead { simplex_delta, sd_tolerance, max_iter, alpha, gamma, rho, sigma }
    }
}

impl PyNelderMead {
    pub fn to_builder(&self) -> NelderMeadBuilder {
        NelderMeadBuilder::new(
            self.simplex_delta,
            self.sd_tolerance,
            self.max_iter,
            self.alpha,
            self.gamma,
            self.rho,
            self.sigma,
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(simplex_delta: f64, sd_tolerance: f64, max_iter: u64, alpha: f64, gamma: f64, rho: f64, sigma: f64)"
)]
fn neldermead(
    simplex_delta: f64,
    sd_tolerance: f64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
) -> PyNelderMead {
    PyNelderMead { simplex_delta, sd_tolerance, max_iter, alpha, gamma, rho, sigma }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PySteepestDescent {
    #[pyo3(get, set)]
    /// Maximum number of iterations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Line search method configuration
    ///
    /// :type: PyLineSearchParams
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PySteepestDescent {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(max_iter: u64, line_search_params: PyLineSearchParams) -> Self {
        PySteepestDescent { max_iter, line_search_params }
    }
}

impl PySteepestDescent {
    pub fn to_builder(&self) -> SteepestDescentBuilder {
        let line_search_params = match &self.line_search_params.method {
            PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                .c1(params.c1)
                .c2(params.c2)
                .width_tolerance(params.width_tolerance)
                .bounds(params.bounds.clone().into())
                .build(),
            PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                .delta(params.delta)
                .sigma(params.sigma)
                .epsilon(params.epsilon)
                .theta(params.theta)
                .gamma(params.gamma)
                .eta(params.eta)
                .bounds(params.bounds.clone().into())
                .build(),
        };

        SteepestDescentBuilder::new(self.max_iter, line_search_params)
    }
}

#[pyfunction]
#[pyo3(text_signature = "(max_iter: u64, line_search_params: Union[PyMoreThuente, PyHagerZhang])")]
fn steepestdescent(
    max_iter: u64,
    line_search_params: Py<pyo3::PyAny>,
    py: Python,
) -> PyResult<PySteepestDescent> {
    let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::MoreThuente(more_thuente) }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::HagerZhang(hager_zhang) }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PySteepestDescent { max_iter, line_search_params })
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyNewtonCG {
    #[pyo3(get, set)]
    /// Maximum number of iterations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Curvature threshold for accepting Newton step
    ///
    /// :type: float
    pub curvature_threshold: f64,

    #[pyo3(get, set)]
    /// Tolerance for stopping criteria
    ///
    /// :type: float
    pub tolerance: f64,

    #[pyo3(get, set)]
    /// Line search method configuration
    ///
    /// :type: PyLineSearchParams
    pub line_search_params: PyLineSearchParams,
}

#[pymethods]
impl PyNewtonCG {
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        curvature_threshold = 0.0,
        tolerance = EPSILON,
        line_search_params = PyLineSearchParams {
            method: PyLineSearchMethod::MoreThuente(PyMoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: vec![EPSILON.sqrt(), INFINITY],
            }),
        },
    ))]
    fn new(
        max_iter: u64,
        curvature_threshold: f64,
        tolerance: f64,
        line_search_params: PyLineSearchParams,
    ) -> Self {
        PyNewtonCG { max_iter, curvature_threshold, tolerance, line_search_params }
    }
}

impl PyNewtonCG {
    pub fn to_builder(&self) -> NewtonCGBuilder {
        NewtonCGBuilder::new(
            self.max_iter,
            self.curvature_threshold,
            self.tolerance,
            match &self.line_search_params.method {
                PyLineSearchMethod::MoreThuente(params) => LineSearchParams::morethuente()
                    .c1(params.c1)
                    .c2(params.c2)
                    .width_tolerance(params.width_tolerance)
                    .bounds(params.bounds.clone().into())
                    .build(),
                PyLineSearchMethod::HagerZhang(params) => LineSearchParams::hagerzhang()
                    .delta(params.delta)
                    .sigma(params.sigma)
                    .epsilon(params.epsilon)
                    .theta(params.theta)
                    .gamma(params.gamma)
                    .eta(params.eta)
                    .bounds(params.bounds.clone().into())
                    .build(),
            },
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(max_iter: u64, curvature_threshold: f64, tolerance: f64, line_search_params: Union[PyMoreThuente, PyHagerZhang])"
)]
fn newtoncg(
    max_iter: u64,
    curvature_threshold: f64,
    tolerance: f64,
    line_search_params: Py<pyo3::PyAny>,
    py: Python,
) -> PyResult<PyNewtonCG> {
    let line_search_params =
        if let Ok(params) = line_search_params.extract::<PyLineSearchParams>(py) {
            params
        } else if let Ok(more_thuente) = line_search_params.extract::<PyMoreThuente>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::MoreThuente(more_thuente) }
        } else if let Ok(hager_zhang) = line_search_params.extract::<PyHagerZhang>(py) {
            PyLineSearchParams { method: PyLineSearchMethod::HagerZhang(hager_zhang) }
        } else {
            return Err(PyTypeError::new_err(
                "Expected PyLineSearchParams, PyMoreThuente, or PyHagerZhang",
            ));
        };

    Ok(PyNewtonCG { max_iter, curvature_threshold, tolerance, line_search_params })
}

#[pyclass(eq, eq_int)]
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTrustRegion {
    #[pyo3(get, set)]
    /// Trust region radius method
    ///
    /// :type: PyTrustRegionRadiusMethod
    pub trust_region_radius_method: PyTrustRegionRadiusMethod,

    #[pyo3(get, set)]
    /// Maximum number of iterations
    ///
    /// :type: int
    pub max_iter: u64,

    #[pyo3(get, set)]
    /// Trust region radius
    ///
    /// :type: float
    pub radius: f64,

    #[pyo3(get, set)]
    /// Maximum trust region radius
    ///
    /// :type: float
    pub max_radius: f64,

    #[pyo3(get, set)]
    /// Trust region expansion factor
    ///
    /// :type: float
    pub eta: f64,
}

#[pymethods]
impl PyTrustRegion {
    #[new]
    #[pyo3(signature = (
        trust_region_radius_method = PyTrustRegionRadiusMethod::Cauchy,
        max_iter = 300,
        radius = 1.0,
        max_radius = 100.0,
        eta = 0.125,
    ))]
    fn new(
        trust_region_radius_method: PyTrustRegionRadiusMethod,
        max_iter: u64,
        radius: f64,
        max_radius: f64,
        eta: f64,
    ) -> Self {
        PyTrustRegion { trust_region_radius_method, max_iter, radius, max_radius, eta }
    }
}

impl PyTrustRegion {
    pub fn to_builder(&self) -> TrustRegionBuilder {
        TrustRegionBuilder::new(
            self.trust_region_radius_method.clone().into(),
            self.max_iter,
            self.radius,
            self.max_radius,
            self.eta,
        )
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(trust_region_radius_method: PyTrustRegionRadiusMethod, max_iter: u64, radius: f64, max_radius: f64, eta: f64)"
)]
fn trustregion(
    trust_region_radius_method: PyTrustRegionRadiusMethod,
    max_iter: u64,
    radius: f64,
    max_radius: f64,
    eta: f64,
) -> PyTrustRegion {
    PyTrustRegion { trust_region_radius_method, max_iter, radius, max_radius, eta }
}

#[pyclass]
#[derive(Debug, Clone)]
/// COBYLA (Constrained Optimization BY Linear Approximations) solver configuration.
///
/// COBYLA is a derivative-free optimization algorithm that can handle inequality
/// constraints. It works by building linear approximations to the objective function
/// and constraints, making it suitable for problems where gradients are unavailable
/// or unreliable.
///
/// :param max_iter: Maximum number of iterations
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
/// - Maximum iterations reached
/// - Function tolerance satisfied: \|f_new - f_old\| < ftol_abs + ftol_rel * \|f_old\|
/// - Parameter tolerance satisfied: \|x_new - x_old\| < xtol_abs + xtol_rel * \|x_old\|
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
/// ...     xtol_abs=[1e-10] * n_vars  # Very tight parameter tolerance
/// ... )
///
/// For expensive function evaluations:
///
/// >>> efficient = gs.builders.cobyla(
/// ...     max_iter=100,
/// ...     ftol_rel=1e-4,  # Looser function tolerance
/// ...     step_size=0.1   # Smaller initial steps
/// ... )
///
/// Different tolerance per variable (for scaled problems):
///
/// >>> scaled = gs.builders.cobyla(
/// ...     xtol_abs=[1e-6, 1e-8, 1e-4]  # x1: 1e-6, x2: 1e-8, x3: 1e-4
/// ... )

pub struct PyCOBYLA {
    #[pyo3(get, set)]
    /// Maximum number of iterations
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
    /// Absolute tolerance for parameter convergence
    ///
    /// :type: float, optional
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
/// COBYLA (Constrained Optimization BY Linear Approximations) is the only
/// solver in this library that can handle inequality constraints. It's also
/// an excellent choice for derivative-free optimization.
///
/// :param max_iter: Maximum number of optimization iterations
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
///    - If `xtol_abs` is provided, its length must match the problem dimension
///    - For constrained problems, COBYLA is currently the only supported solver
///    - Larger `step_size` values encourage more exploration but may slow convergence
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
/// ...     xtol_abs=[1e-8, 1e-8]  # Same tolerance for both variables
/// ... )
///
/// Different tolerance per variable (useful for scaled problems):
///
/// >>> config = gs.builders.cobyla(
/// ...     xtol_abs=[1e-6, 1e-8, 1e-4]  # x1: loose, x2: tight, x3: very loose
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
    m.add_class::<PyLineSearchParams>()?;

    m.add_class::<PyHagerZhang>()?;
    m.add_function(wrap_pyfunction!(hagerzhang, m)?)?;
    m.setattr("HagerZhang", m.getattr("PyHagerZhang")?)?;

    m.add_class::<PyMoreThuente>()?;
    m.add_function(wrap_pyfunction!(morethuente, m)?)?;
    m.setattr("MoreThuente", m.getattr("PyMoreThuente")?)?;

    m.add_class::<PyLBFGS>()?;
    m.add_function(wrap_pyfunction!(lbfgs, m)?)?;
    m.setattr("LBFGS", m.getattr("PyLBFGS")?)?;

    m.add_class::<PyNelderMead>()?;
    m.add_function(wrap_pyfunction!(neldermead, m)?)?;
    m.setattr("nelder_mead", m.getattr("PyNelderMead")?)?;
    m.setattr("NelderMead", m.getattr("PyNelderMead")?)?;

    m.add_class::<PySteepestDescent>()?;
    m.add_function(wrap_pyfunction!(steepestdescent, m)?)?;
    m.setattr("SteepestDescent", m.getattr("PySteepestDescent")?)?;

    m.add_class::<PyNewtonCG>()?;
    m.add_function(wrap_pyfunction!(newtoncg, m)?)?;
    m.setattr("newton_cg", m.getattr("PyNewtonCG")?)?;
    m.setattr("NewtonCG", m.getattr("PyNewtonCG")?)?;

    m.add_class::<PyTrustRegionRadiusMethod>()?;
    m.setattr("TrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;
    m.setattr("PyTrustRegionRadiusMethod", m.getattr("PyTrustRegionRadiusMethod")?)?;

    m.add_class::<PyTrustRegion>()?;
    m.add_function(wrap_pyfunction!(trustregion, m)?)?;
    m.setattr("TrustRegion", m.getattr("PyTrustRegion")?)?;

    m.add_class::<PyCOBYLA>()?;
    m.add_function(wrap_pyfunction!(cobyla, m)?)?;
    m.setattr("COBYLA", m.getattr("PyCOBYLA")?)?;

    Ok(())
}
