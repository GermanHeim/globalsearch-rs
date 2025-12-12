//! # Local Solver Builders Module
//!
//! This module provides builder patterns for configuring local optimization algorithms
//! used within the OQNLP framework. Each builder allows fine-tuned control over
//! algorithm parameters and behavior.
//!
//! ## Builder Pattern Benefits
//!
//! - **Type Safety**: Compile-time validation of configuration parameters
//! - **Default Values**: Sensible defaults for all parameters
//! - **Fluent Interface**: Chain method calls for readable configuration
//! - **Flexibility**: Easy parameter customization without breaking changes
//!
//! ## Supported Algorithms
//!
//! ### Quasi-Newton Methods
//! - [`LBFGSBuilder`] - Limited-memory BFGS for unconstrained optimization
//!
//! ### Direct Search Methods  
//! - [`NelderMeadBuilder`] - Simplex-based derivative-free optimization
//!
//! ### Gradient Methods
//! - [`SteepestDescentBuilder`] - Basic gradient descent with line search
//!
//! ### Trust Region Methods
//! - [`TrustRegionBuilder`] - Advanced second-order optimization
//!
//! ### Newton Methods
//! - [`NewtonCGBuilder`] - Newton method with conjugate gradient solver
//!
//! ### Constrained Methods
//! - [`COBYLABuilder`] - Constrained optimization without derivatives
//!
//! ## Line Search Algorithms
//! - [`HagerZhangBuilder`] - Hager-Zhang line search (recommended)
//! - [`MoreThuenteBuilder`] - Moré-Thuente line search (robust)

#[cfg(feature = "argmin")]
use ndarray::{Array1, array};

#[cfg(feature = "argmin")]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Trust region subproblem solution methods.
///
/// This enum specifies the algorithm used to solve the trust region subproblem,
/// which determines the step direction and length within the trust region.
///
/// ## Methods
///
/// ### Cauchy Point
/// - **Algorithm**: Steepest descent direction scaled to trust region boundary
/// - **Complexity**: O(n) - very fast
/// - **Quality**: Basic approximation, sufficient for many problems
/// - **Best for**: Simple problems, when speed is critical
///
/// ### Steihaug
/// - **Algorithm**: Truncated conjugate gradient method
/// - **Complexity**: O(k (Thv + n)), for k iterations - moderate computational cost
/// - **Quality**: High-quality approximate solution to subproblem
/// - **Best for**: Problems where Hessian information is valuable
///
/// ## Selection Guidelines
///
/// - Use **Cauchy** for rapid prototyping or when function evaluations dominate
/// - Use **Steihaug** for production optimization requiring high solution quality
pub enum TrustRegionRadiusMethod {
    Cauchy,
    Steihaug,
}

#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Local solver configuration for the OQNLP algorithm
///
/// This enum defines the configuration options for the local solver used in the optimizer, depending on the method used.
pub enum LocalSolverConfig {
    #[cfg(feature = "argmin")]
    LBFGS {
        /// Maximum number of iterations for the L-BFGS local solver
        max_iter: u64,
        /// Tolerance for the gradient
        tolerance_grad: f64,
        /// Tolerance for the cost function
        tolerance_cost: f64,
        /// Number of previous iterations to store in the history
        history_size: usize,
        /// L1 regularization coefficient
        l1_coefficient: Option<f64>,
        /// Line search parameters for the L-BFGS local solver
        line_search_params: LineSearchParams,
    },
    #[cfg(feature = "argmin")]
    NelderMead {
        /// Simplex delta
        ///
        /// Sets the step size for generating the simplex from a given point.
        ///
        /// We add the point as the first vertex of the simplex.
        /// Then, for each dimension of the point, we create a new point by cloning the initial point and
        /// then incrementing the value at the given index by the fixed offset, simplex_delta.
        /// This results in a simplex with one vertex for each coordinate direction offset from the initial point.
        ///
        /// The default value is 0.1.
        simplex_delta: f64,
        /// Sample standard deviation tolerance
        sd_tolerance: f64,
        /// Maximum number of iterations for the Nelder-Mead local solver
        max_iter: u64,
        /// Reflection coefficient
        alpha: f64,
        /// Expansion coefficient
        gamma: f64,
        /// Contraction coefficient
        rho: f64,
        /// Shrinkage coefficient
        sigma: f64,
    },
    #[cfg(feature = "argmin")]
    SteepestDescent {
        /// Maximum number of iterations for the Steepest Descent local solver
        max_iter: u64,
        /// Line search parameters for the Steepest Descent local solver
        line_search_params: LineSearchParams,
    },
    #[cfg(feature = "argmin")]
    TrustRegion {
        /// Trust Region radius method to use to compute the step length and direction
        trust_region_radius_method: TrustRegionRadiusMethod,
        /// The maximum number of iterations for the Trust Region local solver
        max_iter: u64,
        /// The radius for the Trust Region local solver
        radius: f64,
        /// The maximum radius for the Trust Region local solver
        max_radius: f64,
        /// The parameter that determines the acceptance threshold for the trust region step
        ///
        /// Must lie in [0, 1/4) and defaults to 0.125
        eta: f64,
        // TODO: Steihaug's method can take with_epsilon, but Cauchy doesn't
        // Should we include it here?
        // TODO: Currently I don't set Dogleg as a method since it would require using linalg from
        // ndarray. If more methods use ArgminInv then it would be a good idea to switch to using linalg
        // and implement it
    },
    #[cfg(feature = "argmin")]
    NewtonCG {
        /// Maximum number of iterations for the Newton local solver
        max_iter: u64,
        /// Curvature threshold
        ///
        /// The curvature threshold for the Newton-CG method. If the curvature is below this threshold,
        /// the step is considered to be a Newton step. The default value is 0.0.
        curvature_threshold: f64,
        /// Tolerance for the Newton-CG method
        tolerance: f64,
        /// Line search parameters for the Newton-CG method
        line_search_params: LineSearchParams,
    },
    COBYLA {
        /// Maximum number of iterations for the COBYLA local solver
        max_iter: u64,
        /// Initial step size for the algorithm
        ///
        /// This determines the initial step size for the algorithm.
        /// Default is 0.5.
        initial_step_size: f64,
        /// Relative function tolerance
        ///
        /// Convergence criterion based on relative change in function value.
        /// Default is 1e-6.
        ftol_rel: f64,
        /// Absolute function tolerance
        ///
        /// Convergence criterion based on absolute change in function value.
        /// Default is 1e-8.
        ftol_abs: f64,
        /// Relative parameter tolerance
        ///
        /// Convergence criterion based on relative change in parameters.
        /// Default is 0 (disabled).
        xtol_rel: f64,
        /// Absolute parameter tolerance
        ///
        /// Convergence criterion based on absolute change in parameters.
        /// Each element corresponds to the absolute tolerance for that variable.
        /// The algorithm stops when all x\[i\] change by less than xtol_abs\[i\].
        /// Default is empty vector (disabled).
        xtol_abs: Vec<f64>,
    },
}

impl std::fmt::Debug for LocalSolverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "argmin")]
            LocalSolverConfig::LBFGS { .. } => f.debug_struct("LBFGS").finish_non_exhaustive(),
            #[cfg(feature = "argmin")]
            LocalSolverConfig::NelderMead { .. } => {
                f.debug_struct("NelderMead").finish_non_exhaustive()
            }
            #[cfg(feature = "argmin")]
            LocalSolverConfig::SteepestDescent { .. } => {
                f.debug_struct("SteepestDescent").finish_non_exhaustive()
            }
            #[cfg(feature = "argmin")]
            LocalSolverConfig::TrustRegion { .. } => {
                f.debug_struct("TrustRegion").finish_non_exhaustive()
            }
            #[cfg(feature = "argmin")]
            LocalSolverConfig::NewtonCG { .. } => {
                f.debug_struct("NewtonCG").finish_non_exhaustive()
            }
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => f
                .debug_struct("COBYLA")
                .field("max_iter", max_iter)
                .field("initial_step_size", initial_step_size)
                .field("ftol_rel", ftol_rel)
                .field("ftol_abs", ftol_abs)
                .field("xtol_rel", xtol_rel)
                .field("xtol_abs", xtol_abs)
                .finish(),
        }
    }
}

impl Clone for LocalSolverConfig {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "argmin")]
            LocalSolverConfig::LBFGS {
                max_iter,
                tolerance_grad,
                tolerance_cost,
                history_size,
                l1_coefficient,
                line_search_params,
            } => LocalSolverConfig::LBFGS {
                max_iter: *max_iter,
                tolerance_grad: *tolerance_grad,
                tolerance_cost: *tolerance_cost,
                history_size: *history_size,
                l1_coefficient: *l1_coefficient,
                line_search_params: line_search_params.clone(),
            },
            #[cfg(feature = "argmin")]
            LocalSolverConfig::NelderMead {
                simplex_delta,
                sd_tolerance,
                max_iter,
                alpha,
                gamma,
                rho,
                sigma,
            } => LocalSolverConfig::NelderMead {
                simplex_delta: *simplex_delta,
                sd_tolerance: *sd_tolerance,
                max_iter: *max_iter,
                alpha: *alpha,
                gamma: *gamma,
                rho: *rho,
                sigma: *sigma,
            },
            #[cfg(feature = "argmin")]
            LocalSolverConfig::SteepestDescent { max_iter, line_search_params } => {
                LocalSolverConfig::SteepestDescent {
                    max_iter: *max_iter,
                    line_search_params: line_search_params.clone(),
                }
            }
            #[cfg(feature = "argmin")]
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method,
                max_iter,
                radius,
                max_radius,
                eta,
            } => LocalSolverConfig::TrustRegion {
                trust_region_radius_method: trust_region_radius_method.clone(),
                max_iter: *max_iter,
                radius: *radius,
                max_radius: *max_radius,
                eta: *eta,
            },
            #[cfg(feature = "argmin")]
            LocalSolverConfig::NewtonCG {
                max_iter,
                curvature_threshold,
                tolerance,
                line_search_params,
            } => LocalSolverConfig::NewtonCG {
                max_iter: *max_iter,
                curvature_threshold: *curvature_threshold,
                tolerance: *tolerance,
                line_search_params: line_search_params.clone(),
            },
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => LocalSolverConfig::COBYLA {
                max_iter: *max_iter,
                initial_step_size: *initial_step_size,
                ftol_rel: *ftol_rel,
                ftol_abs: *ftol_abs,
                xtol_rel: *xtol_rel,
                xtol_abs: xtol_abs.clone(),
            },
        }
    }
}

impl LocalSolverConfig {
    #[cfg(feature = "argmin")]
    pub fn lbfgs() -> LBFGSBuilder {
        LBFGSBuilder::default()
    }

    #[cfg(feature = "argmin")]
    pub fn neldermead() -> NelderMeadBuilder {
        NelderMeadBuilder::default()
    }

    #[cfg(feature = "argmin")]
    pub fn steepestdescent() -> SteepestDescentBuilder {
        SteepestDescentBuilder::default()
    }

    #[cfg(feature = "argmin")]
    pub fn trustregion() -> TrustRegionBuilder {
        TrustRegionBuilder::default()
    }

    #[cfg(feature = "argmin")]
    pub fn newton_cg() -> NewtonCGBuilder {
        NewtonCGBuilder::default()
    }

    pub fn cobyla() -> COBYLABuilder {
        COBYLABuilder::default()
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimization.
///
/// L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
/// a limited amount of memory. It's particularly effective for smooth, unconstrained
/// optimization problems with many variables.
///
/// ## Algorithm Characteristics
/// - **Memory efficient**: Stores only the last `m` updates (typically 5-20)
/// - **Superlinear convergence**: Near second-order convergence rate
/// - **Gradient-based**: Requires first-order derivatives
/// - **Unconstrained**: Best for problems without constraints
///
/// ## When to Use
/// - Large-scale smooth optimization problems
/// - When gradient information is available and reliable
/// - Problems where Hessian computation is expensive
/// - When memory usage needs to be controlled
///
/// ## Configuration Parameters
/// - **Convergence**: `tolerance_grad`, `tolerance_cost`, `max_iter`
/// - **Memory**: `history_size` controls approximation quality vs. memory usage
/// - **Regularization**: `l1_coefficient` for sparse solutions
/// - **Line Search**: Configurable algorithm for step size determination
pub struct LBFGSBuilder {
    max_iter: u64,
    tolerance_grad: f64,
    tolerance_cost: f64,
    history_size: usize,
    l1_coefficient: Option<f64>,
    line_search_params: LineSearchParams,
}

/// L-BFGS Configuration Builder
///
/// Provides a fluent interface for configuring the L-BFGS optimization algorithm.
/// All parameters have sensible defaults but can be customized for specific problems.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::{LBFGSBuilder, HagerZhangBuilder};
///
/// // High-precision configuration for smooth problems
/// let config = LBFGSBuilder::default()
///     .max_iter(1000)
///     .tolerance_grad(1e-12)
///     .history_size(20)  // More memory for better approximation
///     .line_search_params(
///         HagerZhangBuilder::default()
///             .delta(0.01)   // Stricter sufficient decrease
///             .build()
///     )
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl LBFGSBuilder {
    /// Create a new L-BFGS builder
    pub fn new(
        max_iter: u64,
        tolerance_grad: f64,
        tolerance_cost: f64,
        history_size: usize,
        l1_coefficient: Option<f64>,
        line_search_params: LineSearchParams,
    ) -> Self {
        LBFGSBuilder {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            l1_coefficient,
            line_search_params,
        }
    }

    /// Build the L-BFGS local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::LBFGS {
            max_iter: self.max_iter,
            tolerance_grad: self.tolerance_grad,
            tolerance_cost: self.tolerance_cost,
            history_size: self.history_size,
            l1_coefficient: self.l1_coefficient,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the L-BFGS algorithm.
    ///
    /// Controls the maximum number of optimization steps before termination.
    /// Higher values allow more thorough optimization but increase computation time.
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the gradient tolerance for convergence.
    ///
    /// The algorithm terminates when the gradient norm falls below this threshold.
    /// Smaller values provide more precise solutions but require more iterations.
    ///
    /// # Recommended Values
    /// - **Standard precision**: 1e-6 to 1e-8
    /// - **High precision**: 1e-10 to 1e-12
    /// - **Fast approximation**: 1e-4 to 1e-6
    pub fn tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    /// Set the cost function tolerance for convergence.
    ///
    /// The algorithm terminates when the relative change in function value
    /// falls below this threshold between consecutive iterations.
    ///
    /// # Recommended Values
    /// - **Standard**: 1e-8 to 1e-12
    /// - **Loose**: 1e-6 (for noisy functions)
    /// - **Tight**: 1e-15 (for smooth, well-conditioned problems)
    pub fn tolerance_cost(mut self, tolerance_cost: f64) -> Self {
        self.tolerance_cost = tolerance_cost;
        self
    }

    /// Set the L-BFGS history size (memory parameter).
    ///
    /// Controls how many previous iterations are used to approximate the inverse Hessian.
    /// Larger values provide better approximation but use more memory.
    ///
    /// # Recommended Values
    /// - **Memory constrained**: 3-7
    /// - **Standard**: 10-15
    /// - **High quality**: 20-50
    /// - **Diminishing returns**: > 50
    pub fn history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Set the line search parameters for step size determination.
    ///
    /// Line search quality significantly affects L-BFGS performance.
    /// Use HagerZhang for efficiency or MoreThuente for robustness.
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }

    /// Set the L1 regularization coefficient for sparse solutions.
    ///
    /// When set, promotes sparsity in the solution by adding L1 penalty.
    /// Useful for feature selection and compressed sensing problems.
    pub fn l1_coefficient(mut self, l1_coefficient: Option<f64>) -> Self {
        self.l1_coefficient = l1_coefficient;
        self
    }
}

/// Default implementation for the L-BFGS builder
///
/// This implementation sets the default values for the L-BFGS builder.
/// Default values:
/// - `max_iter`: 300
/// - `tolerance_grad`: sqrt(EPSILON)
/// - `tolerance_cost`: EPSILON
/// - `history_size`: 10
/// - `l1_coefficient`: None
/// - `line_search_params`: Default LineSearchParams
#[cfg(feature = "argmin")]
impl Default for LBFGSBuilder {
    fn default() -> Self {
        LBFGSBuilder {
            max_iter: 300,
            tolerance_grad: f64::EPSILON.sqrt(),
            tolerance_cost: f64::EPSILON,
            history_size: 10,
            l1_coefficient: None,
            line_search_params: LineSearchParams::default(),
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Nelder-Mead simplex optimization algorithm.
///
/// The Nelder-Mead method is a direct search algorithm that doesn't require
/// gradient information. It maintains a simplex (geometric shape with n+1 vertices
/// in n-dimensional space) and iteratively improves it through geometric operations.
///
/// ## Algorithm Characteristics
/// - **Derivative-free**: Works with discontinuous or noisy functions
/// - **Robust**: Handles non-smooth optimization landscapes
/// - **Simple**: Few parameters to tune
/// - **Slower convergence**: Linear convergence rate
///
/// ## When to Use
/// - Functions without available gradients
/// - Noisy or discontinuous objective functions
/// - Black-box optimization problems
/// - Small to medium-dimensional problems (typically < 20 variables)
/// - When robustness is more important than speed
///
/// ## Simplex Operations
/// - **Reflection** (`alpha`): Standard operation to move away from worst point
/// - **Expansion** (`gamma`): Aggressive step when reflection succeeds
/// - **Contraction** (`rho`): Conservative step when reflection fails
/// - **Shrinkage** (`sigma`): Global reduction when all else fails
pub struct NelderMeadBuilder {
    simplex_delta: f64,
    sd_tolerance: f64,
    max_iter: u64,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
}

/// Nelder-Mead Configuration Builder
///
/// Provides a fluent interface for configuring the Nelder-Mead simplex algorithm.
/// The default parameters work well for most problems, but can be tuned for
/// specific characteristics like function roughness or convergence requirements.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::NelderMeadBuilder;
///
/// // Configuration for noisy objective functions
/// let config = NelderMeadBuilder::default()
///     .simplex_delta(0.1)     // Delta for simplex creation from point
///     .sd_tolerance(1e-6)     // Looser tolerance for noisy functions
///     .max_iter(2000)         // More iterations for difficult convergence
///     .alpha(1.2)             // Slightly more aggressive reflection
///     .gamma(2.5)             // Enhanced expansion for exploration
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl NelderMeadBuilder {
    /// Create a new Nelder-Mead builder
    pub fn new(
        simplex_delta: f64,
        sd_tolerance: f64,
        max_iter: u64,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> Self {
        NelderMeadBuilder { simplex_delta, sd_tolerance, max_iter, alpha, gamma, rho, sigma }
    }

    /// Build the Nelder-Mead local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::NelderMead {
            simplex_delta: self.simplex_delta,
            sd_tolerance: self.sd_tolerance,
            max_iter: self.max_iter,
            alpha: self.alpha,
            gamma: self.gamma,
            rho: self.rho,
            sigma: self.sigma,
        }
    }

    /// Set the simplex delta parameter
    pub fn simplex_delta(mut self, simplex_delta: f64) -> Self {
        self.simplex_delta = simplex_delta;
        self
    }

    /// Set the sample standard deviation tolerance
    pub fn sd_tolerance(mut self, sd_tolerance: f64) -> Self {
        self.sd_tolerance = sd_tolerance;
        self
    }

    /// Set the maximum number of iterations for the Nelder-Mead local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the reflection coefficient
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the expansion coefficient
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the contraction coefficient
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Set the shrinkage coefficient
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }
}

/// Default implementation for the Nelder-Mead builder
///
/// This implementation sets the default values for the Nelder-Mead builder.
/// Default values:
/// - `simplex_delta`: 0.1
/// - `sd_tolerance`: EPSILON
/// - `max_iter`: 300
/// - `alpha`: 1.0
/// - `gamma`: 2.0
/// - `rho`: 0.5
/// - `sigma`: 0.5
#[cfg(feature = "argmin")]
impl Default for NelderMeadBuilder {
    fn default() -> Self {
        NelderMeadBuilder {
            simplex_delta: 0.1,
            sd_tolerance: f64::EPSILON,
            max_iter: 300,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Steepest Descent (Gradient Descent) optimization.
///
/// Steepest Descent is the most basic gradient-based optimization algorithm.
/// It moves in the direction of the negative gradient at each iteration,
/// with step size determined by line search.
///
/// ## Performance Notes
/// - Slow convergence, especially near optimum
/// - Can zigzag on ill-conditioned problems
/// - Line search quality significantly affects performance
pub struct SteepestDescentBuilder {
    max_iter: u64,
    line_search_params: LineSearchParams,
}

/// Steepest Descent Configuration Builder
///
/// Provides a fluent interface for configuring the steepest descent algorithm.
/// While simple, proper line search configuration is crucial for performance.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::{SteepestDescentBuilder, HagerZhangBuilder};
///
/// // Enhanced configuration with better line search
/// let config = SteepestDescentBuilder::default()
///     .max_iter(5000)  // More iterations due to slow convergence
///     .line_search_params(
///         HagerZhangBuilder::default()
///             .delta(0.01)     // Stricter decrease requirement
///             .sigma(0.99)     // More thorough search
///             .build()
///     )
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl SteepestDescentBuilder {
    /// Create a new Steepest Descent builder
    pub fn new(max_iter: u64, line_search_params: LineSearchParams) -> Self {
        SteepestDescentBuilder { max_iter, line_search_params }
    }

    /// Build the Steepest Descent local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::SteepestDescent {
            max_iter: self.max_iter,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the Steepest Descent local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the line search parameters for the Steepest Descent local solver
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }
}

/// Default implementation for the Steepest Descent builder
///
/// This implementation sets the default values for the Steepest Descent builder.
/// Default values:
/// - `max_iter`: 300
/// - `line_search_params`: Default LineSearchParams
#[cfg(feature = "argmin")]
impl Default for SteepestDescentBuilder {
    fn default() -> Self {
        SteepestDescentBuilder { max_iter: 300, line_search_params: LineSearchParams::default() }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Trust Region optimization methods.
///
/// Trust Region methods solve optimization problems by repeatedly minimizing
/// a quadratic model within a "trust region" - a neighborhood where the model
/// is believed to be accurate. The trust region radius adapts based on the
/// quality of the model predictions.
///
/// ## Algorithm Characteristics
/// - **Second-order**: Uses Hessian information for faster convergence
/// - **Robust**: Adaptive radius provides stability
/// - **Superlinear convergence**: Near Newton-like performance
///
/// ## When to Use
/// - Smooth optimization problems with available Hessian
/// - When robustness is important (compared to line search Newton)
/// - Medium-scale problems where Hessian computation is feasible
/// - Problems with potential numerical difficulties
///
/// ## Trust Region Management
/// - **Radius adaptation**: Automatically adjusts based on model quality
/// - **Subproblem solver**: Cauchy point (fast) or Steihaug (accurate)
/// - **Acceptance criteria**: `eta` parameter controls step acceptance
pub struct TrustRegionBuilder {
    trust_region_radius_method: TrustRegionRadiusMethod,
    max_iter: u64,
    radius: f64,
    max_radius: f64,
    eta: f64,
}

/// Trust Region Configuration Builder
///
/// Provides a fluent interface for configuring trust region optimization methods.
/// The choice of subproblem solver and radius management parameters significantly
/// affects performance and robustness.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::{TrustRegionBuilder, TrustRegionRadiusMethod};
///
/// // High-quality configuration for smooth problems
/// let config = TrustRegionBuilder::default()
///     .method(TrustRegionRadiusMethod::Steihaug)  // More accurate subproblem solver
///     .radius(0.5)                                // Conservative initial radius
///     .max_radius(10.0)                           // Allow large steps when beneficial
///     .eta(0.1)                                   // Accept steps with modest improvement
///     .max_iter(1000)
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl TrustRegionBuilder {
    /// Create a new Trust Region builder
    pub fn new(
        trust_region_radius_method: TrustRegionRadiusMethod,
        max_iter: u64,
        radius: f64,
        max_radius: f64,
        eta: f64,
    ) -> Self {
        TrustRegionBuilder { trust_region_radius_method, max_iter, radius, max_radius, eta }
    }

    /// Build the Trust Region local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::TrustRegion {
            trust_region_radius_method: self.trust_region_radius_method,
            max_iter: self.max_iter,
            radius: self.radius,
            max_radius: self.max_radius,
            eta: self.eta,
        }
    }

    /// Set the Trust Region Method for the Trust Region local solver
    pub fn method(mut self, method: TrustRegionRadiusMethod) -> Self {
        self.trust_region_radius_method = method;
        self
    }

    /// Set the maximum number of iterations for the Trust Region local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the Trust Region radius for the Trust Region local solver
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set the maximum Trust Region radius for the Trust Region local solver
    pub fn max_radius(mut self, max_radius: f64) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta for the Trust Region local solver
    ///
    /// The parameter that determines the acceptance threshold for the trust region step.
    /// Must lie in [0, 1/4) and defaults to 0.125
    pub fn eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }
}

/// Default implementation for the Trust Region builder
///
/// This implementation sets the default values for the Trust Region builder.
/// Default values:
/// - `trust_region_radius_method`: Cauchy
/// - `radius`: 1.0
/// - `max_radius`: 100.0
/// - `eta`: 0.125
#[cfg(feature = "argmin")]
impl Default for TrustRegionBuilder {
    fn default() -> Self {
        TrustRegionBuilder {
            trust_region_radius_method: TrustRegionRadiusMethod::Cauchy,
            max_iter: 300,
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Newton-CG (Newton-Conjugate Gradient) optimization.
///
/// Newton-CG combines Newton's method with conjugate gradient for solving
/// the Newton linear system. This avoids explicit Hessian inversion while
/// maintaining fast convergence properties of Newton's method.
///
/// ## Algorithm Characteristics
/// - **Second-order**: Uses Hessian information
/// - **Superlinear convergence**: Fast convergence near optimum
/// - **Scalable**: Suitable for large-scale problems
///
/// ## When to Use
/// - Large-scale smooth optimization problems
/// - When Hessian information is available
/// - Problems where direct Hessian methods are too expensive
/// - When faster convergence than L-BFGS is needed
///
/// ## Key Features
/// - **Curvature detection**: Handles negative curvature gracefully
/// - **Inexact Newton**: CG iterations can be terminated early
/// - **Line search**: Ensures global convergence
pub struct NewtonCGBuilder {
    max_iter: u64,
    curvature_threshold: f64,
    tolerance: f64,
    line_search_params: LineSearchParams,
}

/// Newton-CG Configuration Builder
///
/// Provides a fluent interface for configuring the Newton-CG algorithm.
/// The combination of curvature threshold and CG tolerance controls the
/// trade-off between accuracy and computational cost.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::{NewtonCGBuilder, MoreThuenteBuilder};
///
/// // High-performance configuration for large-scale problems
/// let config = NewtonCGBuilder::default()
///     .max_iter(500)
///     .curvature_threshold(1e-3)      // Handle negative curvature
///     .tolerance(1e-6)                // CG stopping tolerance
///     .line_search_params(
///         MoreThuenteBuilder::default()
///             .c1(1e-4)
///             .c2(0.9)                // Suitable for Newton methods
///             .build()
///     )
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl NewtonCGBuilder {
    /// Create a new Newton-CG builder
    pub fn new(
        max_iter: u64,
        curvature_threshold: f64,
        tolerance: f64,
        line_search_params: LineSearchParams,
    ) -> Self {
        NewtonCGBuilder { max_iter, curvature_threshold, tolerance, line_search_params }
    }

    /// Build the Newton-CG method local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::NewtonCG {
            max_iter: self.max_iter,
            curvature_threshold: self.curvature_threshold,
            tolerance: self.tolerance,
            line_search_params: self.line_search_params,
        }
    }

    /// Set the maximum number of iterations for the L-BFGS local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the curvature threshold
    pub fn curvature_threshold(mut self, curvature_threshold: f64) -> Self {
        self.curvature_threshold = curvature_threshold;
        self
    }

    /// Set the tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the line search parameters for the Newton-CG method local solver
    pub fn line_search_params(mut self, line_search_params: LineSearchParams) -> Self {
        self.line_search_params = line_search_params;
        self
    }
}

/// Default implementation for Newton-CG builder
///
/// This implementation sets the default values for Newton-CG builder.
/// Default values:
/// - `max_iter`: 300
/// - `curvature_threshold`: 0.0
/// - `tolerance`: EPSILON
/// - `line_search_params`: Default LineSearchParams
#[cfg(feature = "argmin")]
impl Default for NewtonCGBuilder {
    fn default() -> Self {
        NewtonCGBuilder {
            max_iter: 300,
            curvature_threshold: 0.0,
            tolerance: f64::EPSILON,
            line_search_params: LineSearchParams::default(),
        }
    }
}

/// Configuration builder for COBYLA (Constrained Optimization BY Linear Approximations).
///
/// COBYLA is a derivative-free optimization algorithm specifically designed for
/// constrained optimization problems. It uses linear approximations of the
/// objective function and constraints to guide the search.
///
/// ## Algorithm Characteristics
/// - **Derivative-free**: No gradient or Hessian information required
/// - **Constraint handling**: Native support for inequality constraints
/// - **Robust**: Handles noisy and discontinuous functions
/// - **Linear approximations**: Uses simplex-based linear interpolation
///
/// ## When to Use
/// - Constrained optimization problems without derivatives
/// - Black-box functions with constraints
/// - Engineering optimization with simulation-based objectives
/// - When constraint gradients are unavailable or unreliable
/// - Problems with mixed discrete-continuous variables (after relaxation)
///
/// ## Convergence Control
/// - **Function tolerances**: `ftol_rel`, `ftol_abs` for objective convergence
/// - **Parameter tolerances**: `xtol_rel`, `xtol_abs` for variable convergence
/// - **Step size**: `initial_step_size` controls exploration scale
///
/// ## Performance Notes
/// - Slower than gradient-based methods but more robust
/// - Performance depends heavily on initial step size
/// - Best for small to medium-scale problems (< 50 variables)
pub struct COBYLABuilder {
    max_iter: u64,
    initial_step_size: f64,
    ftol_rel: Option<f64>,
    ftol_abs: Option<f64>,
    xtol_rel: Option<f64>,
    xtol_abs: Option<Vec<f64>>,
}

/// COBYLA Configuration Builder
///
/// Provides a fluent interface for configuring the COBYLA constrained optimization
/// algorithm. Tolerance settings are crucial for balancing convergence speed
/// and solution accuracy.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::COBYLABuilder;
///
/// // High-precision configuration for engineering optimization
/// let config = COBYLABuilder::default()
///     .max_iter(1000)
///     .initial_step_size(0.1)     // Match problem scaling
///     .ftol_rel(1e-8)             // Tight relative tolerance
///     .ftol_abs(1e-10)            // Tight absolute tolerance
///     .xtol_rel(1e-6)             // Parameter convergence
///     .xtol_abs(vec![1e-6, 1e-8]) // Per-variable absolute tolerances
///     .build();
/// ```
impl COBYLABuilder {
    /// Create a new COBYLA builder
    pub fn new(max_iter: u64, initial_step_size: f64) -> Self {
        COBYLABuilder {
            max_iter,
            initial_step_size,
            ftol_rel: None,
            ftol_abs: None,
            xtol_rel: None,
            xtol_abs: None,
        }
    }

    /// Build the COBYLA local solver configuration
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::COBYLA {
            max_iter: self.max_iter,
            initial_step_size: self.initial_step_size,
            ftol_rel: self.ftol_rel.unwrap_or(1e-6),
            ftol_abs: self.ftol_abs.unwrap_or(1e-8),
            xtol_rel: self.xtol_rel.unwrap_or(0.0), // No default for x tolerances
            xtol_abs: self.xtol_abs.unwrap_or_default(), // Empty vector means no tolerance
        }
    }

    /// Set the maximum number of iterations for the COBYLA local solver
    pub fn max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the initial step size for the COBYLA local solver
    pub fn initial_step_size(mut self, initial_step_size: f64) -> Self {
        self.initial_step_size = initial_step_size;
        self
    }

    /// Set the relative function tolerance for the COBYLA local solver
    ///
    /// The local solver stops when the objective function changes by less than `ftol_rel * |f(x)|`
    pub fn ftol_rel(mut self, ftol_rel: f64) -> Self {
        self.ftol_rel = Some(ftol_rel);
        self
    }

    /// Set the absolute function tolerance for the COBYLA local solver
    ///
    /// The local solver stops when the objective function changes by less than `ftol_abs`
    pub fn ftol_abs(mut self, ftol_abs: f64) -> Self {
        self.ftol_abs = Some(ftol_abs);
        self
    }

    /// Set the relative parameter tolerance for the COBYLA local solver
    ///
    /// The local solver stops when all `x[i]` changes by less than `xtol_rel * x[i]`
    pub fn xtol_rel(mut self, xtol_rel: f64) -> Self {
        self.xtol_rel = Some(xtol_rel);
        self
    }

    /// Set the absolute parameter tolerance for the COBYLA local solver
    ///
    /// The local solver stops when all `x\[i\]` changes by less than `xtol_abs\[i\]`.
    /// Each element in the vector corresponds to the tolerance for that variable.
    /// If the vector is shorter than the number of variables, the last value is used
    /// for remaining variables. An empty vector disables this convergence criterion.
    pub fn xtol_abs(mut self, xtol_abs: Vec<f64>) -> Self {
        self.xtol_abs = Some(xtol_abs);
        self
    }
}

/// Default implementation for the COBYLA builder
///
/// This implementation sets the default values for the COBYLA builder.
/// Default values:
/// - `max_iter`: 300
/// - `initial_step_size`: 0.5
/// - Function tolerances: `ftol_abs = 1e-8`, `ftol_rel = 1e-6`
/// - Parameter tolerances: disabled (empty vectors)
impl Default for COBYLABuilder {
    fn default() -> Self {
        COBYLABuilder {
            max_iter: 300,
            initial_step_size: 0.5,
            ftol_rel: Some(1e-6),
            ftol_abs: Some(1e-8),
            xtol_rel: None,
            xtol_abs: None,
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Line search methods for the local solver
///
/// This enum defines the types of line search methods that can be used in some of the local solver, including MoreThuente, HagerZhang, and Backtracking.
pub enum LineSearchMethod {
    MoreThuente {
        c1: f64,
        c2: f64,
        width_tolerance: f64,
        bounds: Array1<f64>,
    },
    HagerZhang {
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Array1<f64>,
    },
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Line search parameters for the local solver
///
/// This struct defines the parameters for the line search algorithm used in the local solver. It is only needed for the optimizers that use line search methods.
pub struct LineSearchParams {
    pub method: LineSearchMethod,
}

#[cfg(feature = "argmin")]
impl LineSearchParams {
    pub fn morethuente() -> MoreThuenteBuilder {
        MoreThuenteBuilder::default()
    }

    pub fn hagerzhang() -> HagerZhangBuilder {
        HagerZhangBuilder::default()
    }
}

/// Default implementation for the line search parameters
///
/// This implementation sets the default values for the line search parameters.
/// Default values:
/// - `c1`: 1e-4
/// - `c2`: 0.9
/// - `width_tolerance`: 1e-10
/// - `bounds`: [sqrt(EPSILON), INFINITY]
/// - `method`: MoreThuente
#[cfg(feature = "argmin")]
impl Default for LineSearchParams {
    fn default() -> Self {
        LineSearchParams {
            method: LineSearchMethod::MoreThuente {
                c1: 1e-4,
                c2: 0.9,
                width_tolerance: 1e-10,
                bounds: array![f64::EPSILON.sqrt(), f64::INFINITY],
            },
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Moré-Thuente line search algorithm.
///
/// The Moré-Thuente line search is a robust algorithm that satisfies the strong
/// Wolfe conditions. It's particularly reliable for optimization algorithms that
/// require high-quality line search, such as L-BFGS and Newton methods.
///
/// ## Algorithm Characteristics
/// - **Strong Wolfe conditions**: Ensures both sufficient decrease and curvature
/// - **Robust bracketing**: Reliable identification of acceptable step lengths
/// - **Interpolation-based**: Uses cubic interpolation for efficiency
/// - **Safeguarded**: Includes fallback mechanisms for numerical stability
///
/// ## Wolfe Conditions
/// 1. **Sufficient decrease** (`c1`): f(x + αp) ≤ f(x) + c₁α∇f(x)ᵀp
/// 2. **Curvature condition** (`c2`): |∇f(x + αp)ᵀp| ≤ c₂|∇f(x)ᵀp|
///
/// ## Parameter Guidelines
/// - **c1**: Typically 1e-4, controls sufficient decrease requirement
/// - **c2**: Usually 0.9 for Newton/quasi-Newton, 0.1 for steepest descent
/// - **width_tolerance**: Controls termination of bracketing phase
/// - **bounds**: [min_step, max_step] to prevent extreme step sizes
///
/// ## When to Use
/// - When robustness is critical
/// - With L-BFGS, Newton, or quasi-Newton methods
/// - Problems where line search quality affects convergence
pub struct MoreThuenteBuilder {
    c1: f64,
    c2: f64,
    width_tolerance: f64,
    bounds: Array1<f64>,
}

/// Moré-Thuente Line Search Configuration Builder
///
/// Provides a fluent interface for configuring the Moré-Thuente line search algorithm.
/// This implementation is particularly robust and reliable for optimization methods
/// requiring high-quality line search.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::MoreThuenteBuilder;
/// use ndarray::array;
///
/// // Conservative configuration for challenging problems
/// let line_search = MoreThuenteBuilder::default()
///     .c1(1e-4)                    // Standard sufficient decrease
///     .c2(0.9)                     // Suitable for quasi-Newton methods
///     .width_tolerance(1e-12)      // High precision bracketing
///     .bounds(array![1e-8, 1e3])   // Reasonable step size range
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl MoreThuenteBuilder {
    /// Create a new More-Thuente builder
    pub fn new(c1: f64, c2: f64, width_tolerance: f64, bounds: Array1<f64>) -> Self {
        MoreThuenteBuilder { c1, c2, width_tolerance, bounds }
    }

    /// Build the More-Thuente line search parameters
    pub fn build(self) -> LineSearchParams {
        LineSearchParams {
            method: LineSearchMethod::MoreThuente {
                c1: self.c1,
                c2: self.c2,
                width_tolerance: self.width_tolerance,
                bounds: self.bounds,
            },
        }
    }

    /// Set the strong Wolfe conditions parameter c1
    pub fn c1(mut self, c1: f64) -> Self {
        self.c1 = c1;
        self
    }

    /// Set the strong Wolfe conditions parameter c2
    pub fn c2(mut self, c2: f64) -> Self {
        self.c2 = c2;
        self
    }

    /// Set the width tolerance
    pub fn width_tolerance(mut self, width_tolerance: f64) -> Self {
        self.width_tolerance = width_tolerance;
        self
    }

    /// Set the bounds
    pub fn bounds(mut self, bounds: Array1<f64>) -> Self {
        self.bounds = bounds;
        self
    }
}

/// Default implementation for the More-Thuente builder
///
/// This implementation sets the default values for the More-Thuente builder.
/// Default values:
/// - `c1`: 1e-4
/// - `c2`: 0.9
/// - `width_tolerance`: 1e-10
/// - `bounds`: [sqrt(EPSILON), INFINITY]
#[cfg(feature = "argmin")]
impl Default for MoreThuenteBuilder {
    fn default() -> Self {
        MoreThuenteBuilder {
            c1: 1e-4,
            c2: 0.9,
            width_tolerance: 1e-10,
            bounds: array![f64::EPSILON.sqrt(), f64::INFINITY],
        }
    }
}

#[cfg(feature = "argmin")]
#[derive(Debug, Clone)]
/// Configuration builder for Hager-Zhang line search algorithm.
///
/// The Hager-Zhang line search is an efficient algorithm that satisfies the strong
/// Wolfe conditions with additional safeguards. It often outperforms other line
/// search methods in terms of function evaluations required.
///
/// ## Algorithm Characteristics
/// - **Efficient**: Typically requires fewer function evaluations
/// - **Strong Wolfe conditions**: Ensures convergence guarantees
/// - **Adaptive**: Self-adjusting parameters based on problem characteristics
/// - **Robust**: Handles difficult cases with automatic safeguards
///
/// ## Key Parameters
/// - **delta** (δ): Sufficient decrease parameter, typically 0.1
/// - **sigma** (σ): Controls the curvature condition, usually 0.9
/// - **epsilon** (ε): Relative tolerance for approximate Wolfe conditions
/// - **theta** (θ): Controls the updating of the interval, typically 0.5
/// - **gamma** (γ): Parameter for the approximate Wolfe conditions
/// - **eta** (η): Lower bound for the relative width of the interval
///
/// ## Performance Notes
/// - Often faster than Moré-Thuente in practice
/// - Excellent for L-BFGS and conjugate gradient methods
/// - Self-tuning reduces need for parameter adjustment
///
/// ## When to Use
/// - When efficiency is important
/// - With L-BFGS, CG, or Newton methods
/// - When default Moré-Thuente is too conservative
pub struct HagerZhangBuilder {
    delta: f64,
    sigma: f64,
    epsilon: f64,
    theta: f64,
    gamma: f64,
    eta: f64,
    bounds: Array1<f64>,
}

/// Hager-Zhang Line Search Configuration Builder
///
/// Provides a fluent interface for configuring the Hager-Zhang line search algorithm.
/// This implementation is often more efficient than Moré-Thuente while maintaining
/// strong theoretical guarantees.
///
/// ## Example Usage
/// ```rust
/// use globalsearch::local_solver::builders::HagerZhangBuilder;
/// use ndarray::array;
///
/// // Efficient configuration for L-BFGS
/// let line_search = HagerZhangBuilder::default()
///     .delta(0.01)                 // Stricter sufficient decrease
///     .sigma(0.99)                 // Thorough curvature search
///     .epsilon(1e-6)               // Standard tolerance
///     .bounds(array![1e-10, 1e5])  // Wide step size range
///     .build();
/// ```
#[cfg(feature = "argmin")]
impl HagerZhangBuilder {
    /// Create a new Hager-Zhang builder
    pub fn new(
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        eta: f64,
        bounds: Array1<f64>,
    ) -> Self {
        HagerZhangBuilder { delta, sigma, epsilon, theta, gamma, eta, bounds }
    }

    /// Build the Hager-Zhang line search parameters
    pub fn build(self) -> LineSearchParams {
        LineSearchParams {
            method: LineSearchMethod::HagerZhang {
                delta: self.delta,
                sigma: self.sigma,
                epsilon: self.epsilon,
                theta: self.theta,
                gamma: self.gamma,
                eta: self.eta,
                bounds: self.bounds,
            },
        }
    }

    /// Set the delta parameter
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the sigma parameter
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the theta parameter
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Set the gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the eta parameter
    pub fn eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Set the bounds
    pub fn bounds(mut self, bounds: Array1<f64>) -> Self {
        self.bounds = bounds;
        self
    }
}

/// Default implementation for the Hager-Zhang builder
///
/// This implementation sets the default values for the Hager-Zhang builder.
/// Default values:
/// - `delta`: 0.1
/// - `sigma`: 0.9
/// - `epsilon`: 1e-6
/// - `theta`: 0.5
/// - `gamma`: 0.66
/// - `eta`: 0.01
/// - `bounds`: [sqrt(EPSILON), INFINITY]
#[cfg(feature = "argmin")]
impl Default for HagerZhangBuilder {
    fn default() -> Self {
        HagerZhangBuilder {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            gamma: 0.66,
            eta: 0.01,
            bounds: array![f64::EPSILON, 1e5],
        }
    }
}

#[cfg(test)]
mod tests_builders {
    use super::*;

    // ==== Argmin-specific tests ====
    #[cfg(feature = "argmin")]
    mod argmin_tests {
        use super::*;

        #[test]
        /// Test the default values for the L-BFGS builder
        ///
        /// The default values are:
        /// - `max_iter`: 300
        /// - `tolerance_grad`: sqrt(EPSILON)
        /// - `tolerance_cost`: EPSILON
        /// - `history_size`: 10
        /// - `line_search_params`: Default LineSearchParams
        fn test_default_lbfgs() {
            let lbfgs: LocalSolverConfig = LBFGSBuilder::default().build();
            match lbfgs {
                LocalSolverConfig::LBFGS {
                    max_iter,
                    tolerance_grad,
                    tolerance_cost,
                    history_size,
                    l1_coefficient,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 300);
                    assert_eq!(tolerance_grad, f64::EPSILON.sqrt());
                    assert_eq!(tolerance_cost, f64::EPSILON);
                    assert_eq!(history_size, 10);
                    assert_eq!(l1_coefficient, None);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-4);
                            assert_eq!(c2, 0.9);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected L-BFGS local solver"),
            }
        }

        #[test]
        /// Test the default values for the Nelder-Mead builder
        ///
        /// The default values are:
        /// - `simplex_delta`: 0.1
        /// - `sd_tolerance`: EPSILON
        /// - `max_iter`: 300
        /// - `alpha`: 1.0
        /// - `gamma`: 2.0
        /// - `rho`: 0.5
        /// - `sigma`: 0.5
        fn test_default_neldermead() {
            let neldermead: LocalSolverConfig = NelderMeadBuilder::default().build();
            match neldermead {
                LocalSolverConfig::NelderMead {
                    simplex_delta,
                    sd_tolerance,
                    max_iter,
                    alpha,
                    gamma,
                    rho,
                    sigma,
                } => {
                    assert_eq!(simplex_delta, 0.1);
                    assert_eq!(sd_tolerance, f64::EPSILON);
                    assert_eq!(max_iter, 300);
                    assert_eq!(alpha, 1.0);
                    assert_eq!(gamma, 2.0);
                    assert_eq!(rho, 0.5);
                    assert_eq!(sigma, 0.5);
                }
                _ => panic!("Expected Nelder-Mead local solver"),
            }
        }

        #[test]
        /// Test the default values for the Steepest Descent builder
        ///
        /// The default values are:
        /// - `max_iter`: 300
        /// - `line_search_params`: Default LineSearchParams
        fn test_default_steepestdescent() {
            let steepestdescent: LocalSolverConfig = SteepestDescentBuilder::default().build();
            match steepestdescent {
                LocalSolverConfig::SteepestDescent { max_iter, line_search_params } => {
                    assert_eq!(max_iter, 300);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-4);
                            assert_eq!(c2, 0.9);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Steepest Descent local solver"),
            }
        }

        #[test]
        /// Test the default values for the Trust Region builder
        ///
        /// The default values are:
        /// - `trust_region_radius_method`: Cauchy
        /// - `radius`: 1.0
        /// - `max_radius`: 100.0
        /// - `eta`: 0.125
        fn test_default_trustregion() {
            let trustregion: LocalSolverConfig = TrustRegionBuilder::default().build();
            match trustregion {
                LocalSolverConfig::TrustRegion {
                    trust_region_radius_method,
                    max_iter,
                    radius,
                    max_radius,
                    eta,
                } => {
                    assert_eq!(trust_region_radius_method, TrustRegionRadiusMethod::Cauchy);
                    assert_eq!(max_iter, 300);
                    assert_eq!(radius, 1.0);
                    assert_eq!(max_radius, 100.0);
                    assert_eq!(eta, 0.125);
                }
                _ => panic!("Expected Trust Region local solver"),
            }
        }

        #[test]
        /// Test the default values for the Newton-CG builder
        ///
        /// The default values are:
        /// - `max_iter`: 300
        /// - `curvature_threshold`: 0.0
        /// - `tolerance`: EPSILON
        /// - `line_search_params`: Default LineSearchParams
        fn test_default_newton_cg() {
            let newtoncg: LocalSolverConfig = NewtonCGBuilder::default().build();
            match newtoncg {
                LocalSolverConfig::NewtonCG {
                    max_iter,
                    curvature_threshold,
                    tolerance,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 300);
                    assert_eq!(curvature_threshold, 0.0);
                    assert_eq!(tolerance, f64::EPSILON);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-4);
                            assert_eq!(c2, 0.9);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Newton-CG local solver"),
            }
        }

        /// Test the default values for the More-Thuente builder
        ///
        /// The default values are:
        /// - `c1`: 1e-4
        /// - `c2`: 0.9
        /// - `width_tolerance`: 1e-10
        /// - `bounds`: [sqrt(EPSILON), INFINITY]
        #[test]
        fn test_default_morethuente() {
            let morethuente: LineSearchParams = MoreThuenteBuilder::default().build();
            match morethuente.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    assert_eq!(c1, 1e-4);
                    assert_eq!(c2, 0.9);
                    assert_eq!(width_tolerance, 1e-10);
                    assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                }
                _ => panic!("Expected MoreThuente line search method"),
            }
        }

        #[test]
        /// Test the default values for the Hager-Zhang builder
        ///
        /// The default values are:
        /// - `delta`: 0.1
        /// - `sigma`: 0.9
        /// - `epsilon`: 1e-6
        /// - `theta`: 0.5
        /// - `gamma`: 0.66
        /// - `eta`: 0.01
        /// - `bounds`: [sqrt(EPSILON), 1e5]
        fn test_default_hagerzhang() {
            let hagerzhang: LineSearchParams = HagerZhangBuilder::default().build();
            match hagerzhang.method {
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    assert_eq!(delta, 0.1);
                    assert_eq!(sigma, 0.9);
                    assert_eq!(epsilon, 1e-6);
                    assert_eq!(theta, 0.5);
                    assert_eq!(gamma, 0.66);
                    assert_eq!(eta, 0.01);
                    assert_eq!(bounds, array![f64::EPSILON, 1e5]);
                }
                _ => panic!("Expected HagerZhang line search method"),
            }
        }

        #[test]
        /// Test changing the parameters of L-BFGS builder
        fn change_params_lbfgs() {
            let linesearch: LineSearchParams =
                MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
            let lbfgs: LocalSolverConfig = LBFGSBuilder::default()
                .max_iter(500)
                .tolerance_grad(1e-8)
                .tolerance_cost(1e-8)
                .history_size(5)
                .line_search_params(linesearch)
                .build();
            match lbfgs {
                LocalSolverConfig::LBFGS {
                    max_iter,
                    tolerance_grad,
                    tolerance_cost,
                    history_size,
                    l1_coefficient,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 500);
                    assert_eq!(tolerance_grad, 1e-8);
                    assert_eq!(tolerance_cost, 1e-8);
                    assert_eq!(history_size, 5);
                    assert_eq!(l1_coefficient, None);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected L-BFGS local solver"),
            }
        }

        #[test]
        /// Test changing the parameters of Nelder-Mead builder
        fn change_params_neldermead() {
            let neldermead: LocalSolverConfig = NelderMeadBuilder::default()
                .simplex_delta(0.5)
                .sd_tolerance(1e-5)
                .max_iter(1000)
                .alpha(1.5)
                .gamma(3.0)
                .rho(0.6)
                .sigma(0.6)
                .build();
            match neldermead {
                LocalSolverConfig::NelderMead {
                    simplex_delta,
                    sd_tolerance,
                    max_iter,
                    alpha,
                    gamma,
                    rho,
                    sigma,
                } => {
                    assert_eq!(simplex_delta, 0.5);
                    assert_eq!(sd_tolerance, 1e-5);
                    assert_eq!(max_iter, 1000);
                    assert_eq!(alpha, 1.5);
                    assert_eq!(gamma, 3.0);
                    assert_eq!(rho, 0.6);
                    assert_eq!(sigma, 0.6);
                }
                _ => panic!("Expected Nelder-Mead local solver"),
            }
        }

        #[test]
        /// Test changing the parameters of Steepest Descent builder
        fn change_params_steepestdescent() {
            let linesearch: LineSearchParams =
                MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
            let steepestdescent: LocalSolverConfig = SteepestDescentBuilder::default()
                .max_iter(500)
                .line_search_params(linesearch)
                .build();
            match steepestdescent {
                LocalSolverConfig::SteepestDescent { max_iter, line_search_params } => {
                    assert_eq!(max_iter, 500);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Steepest Descent local solver"),
            }
        }

        #[test]
        /// Test changing the parameters of Trust Region builder
        fn change_params_trustregion() {
            let trustregion: LocalSolverConfig = TrustRegionBuilder::default()
                .method(TrustRegionRadiusMethod::Steihaug)
                .max_iter(500)
                .radius(2.0)
                .max_radius(200.0)
                .eta(0.1)
                .build();
            match trustregion {
                LocalSolverConfig::TrustRegion {
                    trust_region_radius_method,
                    max_iter,
                    radius,
                    max_radius,
                    eta,
                } => {
                    assert_eq!(trust_region_radius_method, TrustRegionRadiusMethod::Steihaug);
                    assert_eq!(max_iter, 500);
                    assert_eq!(radius, 2.0);
                    assert_eq!(max_radius, 200.0);
                    assert_eq!(eta, 0.1);
                }
                _ => panic!("Expected Trust Region local solver"),
            }
        }

        #[test]
        /// Test changing the parameters of Newton-CG builder
        fn change_params_newton_cg() {
            let linesearch: LineSearchParams =
                MoreThuenteBuilder::default().c1(1e-5).c2(0.8).build();
            let newtoncg: LocalSolverConfig = NewtonCGBuilder::default()
                .max_iter(500)
                .curvature_threshold(0.1)
                .tolerance(1e-7)
                .line_search_params(linesearch)
                .build();
            match newtoncg {
                LocalSolverConfig::NewtonCG {
                    max_iter,
                    curvature_threshold,
                    tolerance,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 500);
                    assert_eq!(curvature_threshold, 0.1);
                    assert_eq!(tolerance, 1e-7);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Newton-CG local solver"),
            }
        }

        #[test]
        /// Test changing the parameters of More-Thuente builder
        fn change_params_morethuente() {
            let morethuente: LineSearchParams = MoreThuenteBuilder::default()
                .c1(1e-5)
                .c2(0.8)
                .width_tolerance(1e-8)
                .bounds(array![1e-5, 1e5])
                .build();
            match morethuente.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    assert_eq!(c1, 1e-5);
                    assert_eq!(c2, 0.8);
                    assert_eq!(width_tolerance, 1e-8);
                    assert_eq!(bounds, array![1e-5, 1e5]);
                }
                _ => panic!("Expected MoreThuente line search method"),
            }
        }

        #[test]
        /// Test changing the parameters of Hager-Zhang builder
        fn change_params_hagerzhang() {
            let hagerzhang = HagerZhangBuilder::default()
                .delta(0.2)
                .sigma(0.8)
                .epsilon(1e-7)
                .theta(0.6)
                .gamma(0.7)
                .eta(0.05)
                .bounds(array![1e-6, 1e6])
                .build();

            match hagerzhang.method {
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    assert_eq!(delta, 0.2);
                    assert_eq!(sigma, 0.8);
                    assert_eq!(epsilon, 1e-7);
                    assert_eq!(theta, 0.6);
                    assert_eq!(gamma, 0.7);
                    assert_eq!(eta, 0.05);
                    assert_eq!(bounds, array![1e-6, 1e6]);
                }
                _ => panic!("Expected HagerZhang line search method"),
            }
        }

        #[test]
        /// Test creating a LBFGSdBuilder using new()
        fn test_lbfgs_new() {
            let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
            let lbfgs = LBFGSBuilder::new(500, 1e-8, 1e-8, 5, None, ls).build();
            match lbfgs {
                LocalSolverConfig::LBFGS {
                    max_iter,
                    tolerance_grad,
                    tolerance_cost,
                    history_size,
                    l1_coefficient,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 500);
                    assert_eq!(tolerance_grad, 1e-8);
                    assert_eq!(tolerance_cost, 1e-8);
                    assert_eq!(history_size, 5);
                    assert_eq!(l1_coefficient, None);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected L-BFGS local solver"),
            }
        }

        #[test]
        /// Test creating a NelderMeadBuilder using new()
        fn test_neldermead_new() {
            let nm = NelderMeadBuilder::new(0.5, 1e-5, 1000, 1.5, 3.0, 0.6, 0.6).build();
            match nm {
                LocalSolverConfig::NelderMead {
                    simplex_delta,
                    sd_tolerance,
                    max_iter,
                    alpha,
                    gamma,
                    rho,
                    sigma,
                } => {
                    assert_eq!(simplex_delta, 0.5);
                    assert_eq!(sd_tolerance, 1e-5);
                    assert_eq!(max_iter, 1000);
                    assert_eq!(alpha, 1.5);
                    assert_eq!(gamma, 3.0);
                    assert_eq!(rho, 0.6);
                    assert_eq!(sigma, 0.6);
                }
                _ => panic!("Expected Nelder-Mead local solver"),
            }
        }

        #[test]
        /// Test creating a SteepestDescentBuilder using new()
        fn test_steepestdescent_new() {
            let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
            let sd = SteepestDescentBuilder::new(500, ls).build();
            match sd {
                LocalSolverConfig::SteepestDescent { max_iter, line_search_params } => {
                    assert_eq!(max_iter, 500);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Steepest Descent local solver"),
            }
        }

        #[test]
        /// Test creating a TrustRegionBuilder using new()
        fn test_trustregion_new() {
            let tr =
                TrustRegionBuilder::new(TrustRegionRadiusMethod::Steihaug, 500, 2.0, 200.0, 0.1)
                    .build();
            match tr {
                LocalSolverConfig::TrustRegion {
                    trust_region_radius_method,
                    max_iter,
                    radius,
                    max_radius,
                    eta,
                } => {
                    assert_eq!(trust_region_radius_method, TrustRegionRadiusMethod::Steihaug);
                    assert_eq!(max_iter, 500);
                    assert_eq!(radius, 2.0);
                    assert_eq!(max_radius, 200.0);
                    assert_eq!(eta, 0.1);
                }
                _ => panic!("Expected Trust Region local solver"),
            }
        }

        #[test]
        /// Test creating a NewtonCGBuilder using new()
        fn test_newtoncg_new() {
            let ls = LineSearchParams::morethuente().c1(1e-5).c2(0.8).build();
            let ncg = NewtonCGBuilder::new(500, 0.1, 1e-7, ls).build();
            match ncg {
                LocalSolverConfig::NewtonCG {
                    max_iter,
                    curvature_threshold,
                    tolerance,
                    line_search_params,
                } => {
                    assert_eq!(max_iter, 500);
                    assert_eq!(curvature_threshold, 0.1);
                    assert_eq!(tolerance, 1e-7);
                    match line_search_params.method {
                        LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                            assert_eq!(c1, 1e-5);
                            assert_eq!(c2, 0.8);
                            assert_eq!(width_tolerance, 1e-10);
                            assert_eq!(bounds, array![f64::EPSILON.sqrt(), f64::INFINITY]);
                        }
                        _ => panic!("Expected MoreThuente line search method"),
                    }
                }
                _ => panic!("Expected Newton-CG local solver"),
            }
        }

        #[test]
        /// Test creating a MoreThuenteBuilder using new()
        fn test_morethuente_new() {
            let mt = MoreThuenteBuilder::new(1e-5, 0.8, 1e-8, array![1e-5, 1e5]).build();
            match mt.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    assert_eq!(c1, 1e-5);
                    assert_eq!(c2, 0.8);
                    assert_eq!(width_tolerance, 1e-8);
                    assert_eq!(bounds, array![1e-5, 1e5]);
                }
                _ => panic!("Expected MoreThuente line search method"),
            }
        }

        #[test]
        /// Test creating a HagerZhangBuilder using new()
        fn test_hagerzhang_new() {
            let hz =
                HagerZhangBuilder::new(0.2, 0.8, 1e-7, 0.6, 0.7, 0.05, array![1e-6, 1e6]).build();
            match hz.method {
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    assert_eq!(delta, 0.2);
                    assert_eq!(sigma, 0.8);
                    assert_eq!(epsilon, 1e-7);
                    assert_eq!(theta, 0.6);
                    assert_eq!(gamma, 0.7);
                    assert_eq!(eta, 0.05);
                    assert_eq!(bounds, array![1e-6, 1e6]);
                }
                _ => panic!("Expected HagerZhang line search method"),
            }
        }
    } // end argmin_tests

    // ==== COBYLA tests (always available) ====
    #[test]
    /// Test the default values for the COBYLA builder
    ///
    /// The default values are:
    /// - `max_iter`: 300
    /// - `initial_step_size`: 0.5
    /// - `ftol_rel`: 1e-6
    /// - `ftol_abs`: 1e-8
    fn test_default_cobyla() {
        let cobyla: LocalSolverConfig = COBYLABuilder::default().build();
        match cobyla {
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => {
                assert_eq!(max_iter, 300);
                assert_eq!(initial_step_size, 0.5);
                assert_eq!(ftol_rel, 1e-6); // REL_TOL default
                assert_eq!(ftol_abs, 1e-8); // ABS_TOL default
                assert_eq!(xtol_rel, 0.0); // No default
                assert_eq!(xtol_abs, Vec::<f64>::new()); // No default (empty vector)
            }
            #[cfg_attr(not(feature = "argmin"), allow(unreachable_patterns))]
            _ => panic!("Expected COBYLA local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of COBYLA builder
    fn change_params_cobyla() {
        let cobyla: LocalSolverConfig =
            COBYLABuilder::default().max_iter(500).initial_step_size(0.1).ftol_rel(1e-10).build();
        match cobyla {
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(initial_step_size, 0.1);
                assert_eq!(ftol_rel, 1e-10);
                assert_eq!(ftol_abs, 1e-8);
                assert_eq!(xtol_rel, 0.0);
                assert_eq!(xtol_abs, Vec::<f64>::new());
            }
            #[cfg_attr(not(feature = "argmin"), allow(unreachable_patterns))]
            _ => panic!("Expected COBYLA local solver"),
        }
    }

    #[test]
    /// Test creating a COBYLABuilder using new()
    fn test_cobyla_new() {
        let cobyla = COBYLABuilder::new(500, 0.5).build();
        match cobyla {
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(initial_step_size, 0.5);
                assert_eq!(ftol_rel, 1e-6); // default
                assert_eq!(ftol_abs, 1e-8); // default
                assert_eq!(xtol_rel, 0.0); // default (no x tolerance)
                assert_eq!(xtol_abs, Vec::<f64>::new()); // default (no x tolerance)
            }
            #[cfg_attr(not(feature = "argmin"), allow(unreachable_patterns))]
            _ => panic!("Expected COBYLA local solver"),
        }
    }

    #[test]
    /// Test COBYLA builder with vector-based xtol_abs
    fn test_cobyla_vector_xtol_abs() {
        let xtol_vec = vec![1e-6, 1e-8, 1e-10];
        let cobyla: LocalSolverConfig = COBYLABuilder::default()
            .max_iter(1000)
            .initial_step_size(0.1)
            .ftol_rel(1e-8)
            .xtol_abs(xtol_vec.clone())
            .build();

        match cobyla {
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs,
            } => {
                assert_eq!(max_iter, 1000);
                assert_eq!(initial_step_size, 0.1);
                assert_eq!(ftol_rel, 1e-8);
                assert_eq!(ftol_abs, 1e-8); // default
                assert_eq!(xtol_rel, 0.0); // default
                assert_eq!(xtol_abs, xtol_vec); // per-variable tolerances
            }
            #[cfg_attr(not(feature = "argmin"), allow(unreachable_patterns))]
            _ => panic!("Expected COBYLA local solver"),
        }
    }
}
