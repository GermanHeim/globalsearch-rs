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
//! All solvers are backed by Basin:
//!
//! - [`LBFGSBuilder`] - Unconstrained L-BFGS
//! - [`GradientDescentBuilder`] - Unconstrained gradient descent
//! - [`TrustRegionBuilder`] - Unconstrained trust region (Cauchy or Steihaug)
//! - [`NelderMeadBuilder`] - Unconstrained Nelder-Mead
//! - [`LBFGSBBuilder`] - Box-constrained L-BFGS-B
//! - [`BoundedNelderMeadBuilder`] - Box-constrained Nelder-Mead
//! - [`BOBYQABuilder`] - Box-constrained BOBYQA
//! - [`COBYLABuilder`] - Constrained optimization without derivatives
//! - [`SLSQPBuilder`] - Gradient-based SLSQP with nonlinear constraints
//! - [`BarrierBuilder`] - Log-barrier method for linear inequalities
//! - [`AugmentedLagrangianBuilder`] - Augmented Lagrangian for linear equalities

mod basin;
pub use basin::*;

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
#[derive(Clone)]
pub enum LocalSolverConfig {
    COBYLA {
        /// Maximum number of objective evaluations for the COBYLA local solver
        max_iter: u64,
        /// Initial step size for the algorithm
        ///
        /// This determines the initial trust-region radius for the algorithm.
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
        /// Sets the final trust-region radius relative to `initial_step_size`.
        /// Default is 0 (disabled).
        xtol_rel: f64,
        /// Per-variable absolute parameter tolerances
        ///
        /// Each element corresponds to the absolute tolerance for that variable.
        /// The largest positive tolerance contributes to the final trust-region radius.
        /// Default is an empty vector (disabled).
        xtol_abs: Vec<f64>,
    },
    /// Unconstrained L-BFGS with the default More-Thuente line search.
    LBFGS {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
        tolerance_grad: Option<f64>,
        /// Absolute change in cost between iterates. Disabled by default.
        tolerance_cost: Option<f64>,
        /// Number of correction pairs retained by L-BFGS. Must be positive.
        history_size: usize,
    },

    /// Unconstrained gradient descent with the default More-Thuente line search and no momentum.
    GradientDescent {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
        tolerance_grad: Option<f64>,
        /// Absolute change in cost between iterates. Disabled by default.
        tolerance_cost: Option<f64>,
    },

    /// Unconstrained trust-region optimization using the supplied gradient and Hessian.
    TrustRegion {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
        tolerance_grad: Option<f64>,
        /// Trust-region subproblem method, defaulting to Steihaug.
        trust_region_radius_method: TrustRegionRadiusMethod,
        /// Positive initial trust-region radius.
        radius: f64,
        /// Maximum trust-region radius, at least the initial radius.
        max_radius: f64,
        /// Step acceptance threshold, in [0, 0.25).
        eta: f64,
    },

    /// Unconstrained Nelder-Mead with standard coefficients.
    NelderMead {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Positive absolute coordinate step for the initial simplex.
        simplex_delta: f64,
        /// Maximum simplex distance from its best vertex in the infinity norm.
        tolerance_simplex: Option<f64>,
        /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
        tolerance_cost: Option<f64>,
    },

    /// Box-constrained L-BFGS-B with the default More-Thuente line search.
    LBFGSB {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Absolute projected-gradient infinity-norm tolerance. None disables.
        tolerance_projected_grad: Option<f64>,
        /// Absolute change in cost between iterates. Disabled by default.
        tolerance_cost: Option<f64>,
        /// Number of correction pairs retained by L-BFGS. Must be positive.
        history_size: usize,
    },

    /// Box-constrained Nelder-Mead with projected trial vertices and standard coefficients.
    BoundedNelderMead {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Positive absolute coordinate step for the initial simplex.
        simplex_delta: f64,
        /// Maximum simplex distance from its best vertex in the infinity norm.
        tolerance_simplex: Option<f64>,
        /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
        tolerance_cost: Option<f64>,
    },

    /// Box-constrained BOBYQA. Basin reduces the radii automatically for narrow boxes.
    BOBYQA {
        /// Maximum number of solver iterations, excluding initialization.
        max_iter: u64,
        /// Positive initial trust-region radius, larger than the final radius.
        initial_radius: f64,
        /// Positive final trust-region radius.
        final_radius: f64,
        /// Interpolation-set size, between 2n+1 and (n+1)(n+2)/2. None selects 2n+1.
        interpolation_points: Option<usize>,
    },

    /// Gradient-based SLSQP with box, linear, and nonlinear constraints.
    ///
    /// New variants are appended at the end so existing `bincode` encodings
    /// (COBYLA is index 0) remain stable.
    SLSQP {
        /// Maximum number of executor iterations (outer SLSQP steps).
        max_iter: u64,
        /// Kraft composite accuracy tolerance. None disables convergence
        /// tests; zero requests exact zero.
        accuracy: Option<f64>,
        /// NNLS active-set iteration limit per QP subproblem. None selects
        /// the reference default (three times the column count).
        max_subproblem_iter: Option<usize>,
    },

    /// Log-barrier method over an L-BFGS-B inner solver for linear
    /// inequalities `A x <= b` (plus box bounds).
    Barrier {
        /// Maximum number of outer barrier iterations (safety budget).
        max_iter: u64,
        /// Initial barrier parameter `mu0`. Must be positive.
        mu0: f64,
        /// Per-iteration shrink factor `mu <- mu / reduction`. Must exceed 1.
        reduction: f64,
        /// Outer duality-gap tolerance: stop once `m * mu <= tol`.
        duality_gap_tol: f64,
        /// Iteration budget for each inner barrier-subproblem solve.
        inner_max_iter: u64,
    },

    /// Augmented-Lagrangian method over an L-BFGS-B inner solver for linear
    /// equalities `A x = b` (plus box bounds).
    AugmentedLagrangian {
        /// Maximum number of outer iterations (safety budget).
        max_iter: u64,
        /// Initial penalty parameter. Must be positive.
        rho0: f64,
        /// Penalty growth factor applied when feasibility stalls. Must exceed 1.
        rho_increase: f64,
        /// Required feasibility-decrease ratio in (0, 1) for multiplier updates.
        feasibility_decrease: f64,
        /// Outer feasibility tolerance: stop once `||A x - b|| <= tol`.
        feasibility_tol: f64,
        /// Iteration budget for each inner subproblem solve.
        inner_max_iter: u64,
    },
}

impl std::fmt::Debug for LocalSolverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LBFGS { max_iter, tolerance_grad, tolerance_cost, history_size } => f
                .debug_struct("LBFGS")
                .field("max_iter", max_iter)
                .field("tolerance_grad", tolerance_grad)
                .field("tolerance_cost", tolerance_cost)
                .field("history_size", history_size)
                .finish(),

            Self::GradientDescent { max_iter, tolerance_grad, tolerance_cost } => f
                .debug_struct("GradientDescent")
                .field("max_iter", max_iter)
                .field("tolerance_grad", tolerance_grad)
                .field("tolerance_cost", tolerance_cost)
                .finish(),

            Self::TrustRegion {
                max_iter,
                tolerance_grad,
                trust_region_radius_method,
                radius,
                max_radius,
                eta,
            } => f
                .debug_struct("TrustRegion")
                .field("max_iter", max_iter)
                .field("tolerance_grad", tolerance_grad)
                .field("trust_region_radius_method", trust_region_radius_method)
                .field("radius", radius)
                .field("max_radius", max_radius)
                .field("eta", eta)
                .finish(),

            Self::NelderMead { max_iter, simplex_delta, tolerance_simplex, tolerance_cost } => f
                .debug_struct("NelderMead")
                .field("max_iter", max_iter)
                .field("simplex_delta", simplex_delta)
                .field("tolerance_simplex", tolerance_simplex)
                .field("tolerance_cost", tolerance_cost)
                .finish(),

            Self::LBFGSB { max_iter, tolerance_projected_grad, tolerance_cost, history_size } => f
                .debug_struct("LBFGSB")
                .field("max_iter", max_iter)
                .field("tolerance_projected_grad", tolerance_projected_grad)
                .field("tolerance_cost", tolerance_cost)
                .field("history_size", history_size)
                .finish(),

            Self::BoundedNelderMead {
                max_iter,
                simplex_delta,
                tolerance_simplex,
                tolerance_cost,
            } => f
                .debug_struct("BoundedNelderMead")
                .field("max_iter", max_iter)
                .field("simplex_delta", simplex_delta)
                .field("tolerance_simplex", tolerance_simplex)
                .field("tolerance_cost", tolerance_cost)
                .finish(),

            Self::BOBYQA { max_iter, initial_radius, final_radius, interpolation_points } => f
                .debug_struct("BOBYQA")
                .field("max_iter", max_iter)
                .field("initial_radius", initial_radius)
                .field("final_radius", final_radius)
                .field("interpolation_points", interpolation_points)
                .finish(),

            Self::SLSQP { max_iter, accuracy, max_subproblem_iter } => f
                .debug_struct("SLSQP")
                .field("max_iter", max_iter)
                .field("accuracy", accuracy)
                .field("max_subproblem_iter", max_subproblem_iter)
                .finish(),

            Self::Barrier { max_iter, mu0, reduction, duality_gap_tol, inner_max_iter } => f
                .debug_struct("Barrier")
                .field("max_iter", max_iter)
                .field("mu0", mu0)
                .field("reduction", reduction)
                .field("duality_gap_tol", duality_gap_tol)
                .field("inner_max_iter", inner_max_iter)
                .finish(),

            Self::AugmentedLagrangian {
                max_iter,
                rho0,
                rho_increase,
                feasibility_decrease,
                feasibility_tol,
                inner_max_iter,
            } => f
                .debug_struct("AugmentedLagrangian")
                .field("max_iter", max_iter)
                .field("rho0", rho0)
                .field("rho_increase", rho_increase)
                .field("feasibility_decrease", feasibility_decrease)
                .field("feasibility_tol", feasibility_tol)
                .field("inner_max_iter", inner_max_iter)
                .finish(),

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

impl LocalSolverConfig {
    /// Unconstrained L-BFGS with the default More-Thuente line search.
    pub fn lbfgs() -> LBFGSBuilder {
        LBFGSBuilder::default()
    }

    /// Unconstrained gradient descent with the default More-Thuente line search and no momentum.
    pub fn gradient_descent() -> GradientDescentBuilder {
        GradientDescentBuilder::default()
    }

    /// Unconstrained trust-region optimization using the supplied gradient and Hessian.
    pub fn trust_region() -> TrustRegionBuilder {
        TrustRegionBuilder::default()
    }

    /// Unconstrained Nelder-Mead with standard coefficients.
    pub fn nelder_mead() -> NelderMeadBuilder {
        NelderMeadBuilder::default()
    }

    /// Box-constrained L-BFGS-B with the default More-Thuente line search.
    pub fn lbfgsb() -> LBFGSBBuilder {
        LBFGSBBuilder::default()
    }

    /// Box-constrained Nelder-Mead with projected trial vertices and standard coefficients.
    pub fn bounded_nelder_mead() -> BoundedNelderMeadBuilder {
        BoundedNelderMeadBuilder::default()
    }

    /// Box-constrained BOBYQA. Basin reduces the radii automatically for narrow boxes.
    pub fn bobyqa() -> BOBYQABuilder {
        BOBYQABuilder::default()
    }

    /// Gradient-based SLSQP with box, linear, and nonlinear constraints.
    pub fn slsqp() -> SLSQPBuilder {
        SLSQPBuilder::default()
    }

    /// Log-barrier method over an L-BFGS-B inner solver for linear inequalities.
    pub fn barrier() -> BarrierBuilder {
        BarrierBuilder::default()
    }

    /// Augmented-Lagrangian method over an L-BFGS-B inner solver for linear equalities.
    pub fn augmented_lagrangian() -> AugmentedLagrangianBuilder {
        AugmentedLagrangianBuilder::default()
    }

    pub fn cobyla() -> COBYLABuilder {
        COBYLABuilder::default()
    }

    /// Returns the corresponding LocalSolverType for this configuration
    pub fn solver_type(&self) -> crate::types::LocalSolverType {
        match self {
            Self::LBFGS { .. } => crate::types::LocalSolverType::LBFGS,
            Self::GradientDescent { .. } => crate::types::LocalSolverType::GradientDescent,
            Self::TrustRegion { .. } => crate::types::LocalSolverType::TrustRegion,
            Self::NelderMead { .. } => crate::types::LocalSolverType::NelderMead,
            Self::LBFGSB { .. } => crate::types::LocalSolverType::LBFGSB,
            Self::BoundedNelderMead { .. } => crate::types::LocalSolverType::BoundedNelderMead,
            Self::BOBYQA { .. } => crate::types::LocalSolverType::BOBYQA,
            LocalSolverConfig::COBYLA { .. } => crate::types::LocalSolverType::COBYLA,
            Self::SLSQP { .. } => crate::types::LocalSolverType::SLSQP,
            Self::Barrier { .. } => crate::types::LocalSolverType::Barrier,
            Self::AugmentedLagrangian { .. } => crate::types::LocalSolverType::AugmentedLagrangian,
        }
    }
}

/// Configuration builder for COBYLA (Constrained Optimization BY Linear Approximations).
///
/// This builder configures Basin's derivative-free COBYLA implementation for
/// constrained optimization problems. COBYLA uses linear approximations of the
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
/// - **Function tolerances**: `ftol_rel` and `ftol_abs` control objective convergence
/// - **Parameter tolerances**: `xtol_rel` and `xtol_abs` control the final resolution
/// - **Step size**: `initial_step_size` controls the initial exploration scale
///
/// The final trust-region radius is the maximum of `xtol_rel * initial_step_size`,
/// the positive entries in `xtol_abs`, and a numerical floor of
/// `max(sqrt(f64::EPSILON) * initial_step_size, f64::MIN_POSITIVE)`.
/// This floor also applies when parameter tolerances are disabled.
///
/// ## Performance Notes
/// - Slower than gradient-based methods but more robust
/// - Performance depends heavily on the initial trust-region radius
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
///     .initial_step_size(0.1)     // Match problem scaling.
///     .ftol_rel(1e-8)             // Tight relative tolerance.
///     .ftol_abs(1e-10)            // Tight absolute tolerance.
///     .xtol_rel(1e-6)             // Request fine parameter resolution.
///     .xtol_abs(vec![1e-6, 1e-8]) // Per-variable absolute tolerances.
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
            xtol_rel: self.xtol_rel.unwrap_or(0.0), // Zero disables relative parameter tolerance.
            xtol_abs: self.xtol_abs.unwrap_or_default(), // An empty vector disables absolute tolerances.
        }
    }

    /// Set the maximum number of objective evaluations for the COBYLA local solver
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
    /// At changes in the trust-region radius, the local solver stops when the objective
    /// decreases by less than `ftol_rel * (|f_new| + |f_old|) / 2`.
    /// Nonpositive values disable this convergence criterion.
    pub fn ftol_rel(mut self, ftol_rel: f64) -> Self {
        self.ftol_rel = Some(ftol_rel);
        self
    }

    /// Set the absolute function tolerance for the COBYLA local solver
    ///
    /// At changes in the trust-region radius, the local solver stops when the objective
    /// decreases by less than `ftol_abs`.
    /// Nonpositive values disable this convergence criterion.
    pub fn ftol_abs(mut self, ftol_abs: f64) -> Self {
        self.ftol_abs = Some(ftol_abs);
        self
    }

    /// Set the relative parameter tolerance for the COBYLA local solver
    ///
    /// Contributes `xtol_rel * initial_step_size` to the final trust-region radius.
    /// The radius also accounts for absolute tolerances and a numerical floor.
    /// Nonpositive values disable this relative tolerance.
    pub fn xtol_rel(mut self, xtol_rel: f64) -> Self {
        self.xtol_rel = Some(xtol_rel);
        self
    }

    /// Set the per-variable absolute parameter tolerances for the COBYLA local solver
    ///
    /// Each element corresponds to the absolute tolerance for that variable.
    /// A nonempty vector must contain one entry per variable. Its largest positive
    /// entry contributes to the final trust-region radius, together with the relative
    /// tolerance and a numerical floor. An empty vector disables absolute tolerances.
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
/// - `ftol_rel`: 1e-6
/// - `ftol_abs`: 1e-8
/// - Parameter tolerances: disabled
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

/// Configuration builder for SLSQP (Sequential Least-Squares Programming).
///
/// Gradient-based solver for smooth problems with box bounds, linear
/// constraints, and nonlinear equalities/inequalities. Requires the problem
/// to implement `gradient` and `constraint_jacobian`. Nonlinear feasibility
/// is not required at the start.
#[derive(Debug, Clone)]
pub struct SLSQPBuilder {
    max_iter: u64,
    accuracy: Option<f64>,
    max_subproblem_iter: Option<usize>,
}

impl Default for SLSQPBuilder {
    fn default() -> Self {
        Self { max_iter: 1000, accuracy: Some(1e-6), max_subproblem_iter: None }
    }
}

impl SLSQPBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of executor iterations (outer SLSQP steps).
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Kraft composite accuracy tolerance. None disables convergence tests;
    /// zero requests exact-zero tests. Must be finite and nonnegative.
    pub fn accuracy(mut self, value: impl Into<Option<f64>>) -> Self {
        self.accuracy = value.into();
        self
    }

    /// NNLS active-set iteration limit per QP subproblem. None selects the
    /// reference default (three times the column count).
    pub fn max_subproblem_iter(mut self, value: impl Into<Option<usize>>) -> Self {
        self.max_subproblem_iter = value.into();
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::SLSQP {
            max_iter: self.max_iter,
            accuracy: self.accuracy,
            max_subproblem_iter: self.max_subproblem_iter,
        }
    }
}

/// Configuration builder for the log-barrier method.
///
/// Wraps an L-BFGS-B inner solver to handle linear inequalities `A x <= b`
/// (plus box bounds). Requires the problem to implement `gradient` and
/// `linear_inequalities`. Phase I automatically finds a strictly feasible
/// point from infeasible starts.
#[derive(Debug, Clone)]
pub struct BarrierBuilder {
    max_iter: u64,
    mu0: f64,
    reduction: f64,
    duality_gap_tol: f64,
    inner_max_iter: u64,
}

impl Default for BarrierBuilder {
    fn default() -> Self {
        Self { max_iter: 100, mu0: 1.0, reduction: 10.0, duality_gap_tol: 1e-8, inner_max_iter: 50 }
    }
}

impl BarrierBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of outer barrier iterations (safety budget).
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Initial barrier parameter. Must be positive.
    pub fn mu0(mut self, value: f64) -> Self {
        self.mu0 = value;
        self
    }

    /// Per-iteration shrink factor. Must exceed 1.
    pub fn reduction(mut self, value: f64) -> Self {
        self.reduction = value;
        self
    }

    /// Outer duality-gap tolerance: stop once `m * mu <= tol`.
    pub fn duality_gap_tol(mut self, value: f64) -> Self {
        self.duality_gap_tol = value;
        self
    }

    /// Iteration budget for each inner barrier-subproblem solve.
    pub fn inner_max_iter(mut self, value: u64) -> Self {
        self.inner_max_iter = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::Barrier {
            max_iter: self.max_iter,
            mu0: self.mu0,
            reduction: self.reduction,
            duality_gap_tol: self.duality_gap_tol,
            inner_max_iter: self.inner_max_iter,
        }
    }
}

/// Configuration builder for the augmented-Lagrangian method.
///
/// Wraps an L-BFGS-B inner solver to handle linear equalities `A x = b`
/// (plus box bounds). Requires the problem to implement `gradient` and
/// `linear_equalities`. Infeasible starts are fine.
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianBuilder {
    max_iter: u64,
    rho0: f64,
    rho_increase: f64,
    feasibility_decrease: f64,
    feasibility_tol: f64,
    inner_max_iter: u64,
}

impl Default for AugmentedLagrangianBuilder {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rho0: 10.0,
            rho_increase: 10.0,
            feasibility_decrease: 0.25,
            feasibility_tol: 1e-8,
            inner_max_iter: 50,
        }
    }
}

impl AugmentedLagrangianBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of outer iterations (safety budget).
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Initial penalty parameter. Must be positive.
    pub fn rho0(mut self, value: f64) -> Self {
        self.rho0 = value;
        self
    }

    /// Penalty growth factor when feasibility stalls. Must exceed 1.
    pub fn rho_increase(mut self, value: f64) -> Self {
        self.rho_increase = value;
        self
    }

    /// Required feasibility-decrease ratio in (0, 1) for multiplier updates.
    pub fn feasibility_decrease(mut self, value: f64) -> Self {
        self.feasibility_decrease = value;
        self
    }

    /// Outer feasibility tolerance: stop once `||A x - b|| <= tol`.
    pub fn feasibility_tol(mut self, value: f64) -> Self {
        self.feasibility_tol = value;
        self
    }

    /// Iteration budget for each inner subproblem solve.
    pub fn inner_max_iter(mut self, value: u64) -> Self {
        self.inner_max_iter = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::AugmentedLagrangian {
            max_iter: self.max_iter,
            rho0: self.rho0,
            rho_increase: self.rho_increase,
            feasibility_decrease: self.feasibility_decrease,
            feasibility_tol: self.feasibility_tol,
            inner_max_iter: self.inner_max_iter,
        }
    }
}

#[cfg(test)]
mod tests_builders {
    use super::*;

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
                assert_eq!(ftol_rel, 1e-6);
                assert_eq!(ftol_abs, 1e-8);
                assert_eq!(xtol_rel, 0.0);
                assert!(xtol_abs.is_empty());
            }
            _ => panic!("Expected COBYLA local solver"),
        }
    }

    #[test]
    /// Test changing the parameters of COBYLA builder
    fn change_params_cobyla() {
        let xtol_abs = vec![1e-6, 1e-8];
        let cobyla: LocalSolverConfig = COBYLABuilder::default()
            .max_iter(500)
            .initial_step_size(0.1)
            .ftol_rel(1e-10)
            .ftol_abs(1e-12)
            .xtol_rel(1e-9)
            .xtol_abs(xtol_abs.clone())
            .build();
        match cobyla {
            LocalSolverConfig::COBYLA {
                max_iter,
                initial_step_size,
                ftol_rel,
                ftol_abs,
                xtol_rel,
                xtol_abs: actual_xtol_abs,
            } => {
                assert_eq!(max_iter, 500);
                assert_eq!(initial_step_size, 0.1);
                assert_eq!(ftol_rel, 1e-10);
                assert_eq!(ftol_abs, 1e-12);
                assert_eq!(xtol_rel, 1e-9);
                assert_eq!(actual_xtol_abs, xtol_abs);
            }
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
                assert_eq!(ftol_rel, 1e-6);
                assert_eq!(ftol_abs, 1e-8);
                assert_eq!(xtol_rel, 0.0);
                assert!(xtol_abs.is_empty());
            }
            _ => panic!("Expected COBYLA local solver"),
        }
    }

    #[test]
    /// Test that `LocalSolverConfig::solver_type()` returns the correct `LocalSolverType`
    /// for the COBYLA variant
    fn test_solver_type_cobyla() {
        use crate::types::LocalSolverType;
        let config = COBYLABuilder::default().build();
        assert_eq!(config.solver_type(), LocalSolverType::COBYLA);
    }

    #[test]
    /// Test that `LocalSolverConfig::solver_type()` returns the correct `LocalSolverType`
    /// for all Basin-backed variants
    fn test_solver_type_variants() {
        use crate::types::LocalSolverType;

        assert_eq!(LBFGSBuilder::default().build().solver_type(), LocalSolverType::LBFGS);
        assert_eq!(NelderMeadBuilder::default().build().solver_type(), LocalSolverType::NelderMead);
        assert_eq!(
            GradientDescentBuilder::default().build().solver_type(),
            LocalSolverType::GradientDescent
        );
        assert_eq!(
            TrustRegionBuilder::default().build().solver_type(),
            LocalSolverType::TrustRegion
        );
        assert_eq!(LBFGSBBuilder::default().build().solver_type(), LocalSolverType::LBFGSB);
        assert_eq!(
            BoundedNelderMeadBuilder::default().build().solver_type(),
            LocalSolverType::BoundedNelderMead
        );
        assert_eq!(BOBYQABuilder::default().build().solver_type(), LocalSolverType::BOBYQA);
        assert_eq!(SLSQPBuilder::default().build().solver_type(), LocalSolverType::SLSQP);
        assert_eq!(BarrierBuilder::default().build().solver_type(), LocalSolverType::Barrier);
        assert_eq!(
            AugmentedLagrangianBuilder::default().build().solver_type(),
            LocalSolverType::AugmentedLagrangian
        );
    }

    #[test]
    /// Test SLSQP builder defaults and parameter overrides
    fn test_slsqp_builder() {
        let config = SLSQPBuilder::default().build();
        match config {
            LocalSolverConfig::SLSQP { max_iter, accuracy, max_subproblem_iter } => {
                assert_eq!(max_iter, 1000);
                assert_eq!(accuracy, Some(1e-6));
                assert_eq!(max_subproblem_iter, None);
            }
            _ => panic!("Expected SLSQP local solver"),
        }
        let config =
            SLSQPBuilder::default().max_iter(50).accuracy(None).max_subproblem_iter(10).build();
        match config {
            LocalSolverConfig::SLSQP { max_iter, accuracy, max_subproblem_iter } => {
                assert_eq!(max_iter, 50);
                assert_eq!(accuracy, None);
                assert_eq!(max_subproblem_iter, Some(10));
            }
            _ => panic!("Expected SLSQP local solver"),
        }
    }

    #[test]
    /// Test Barrier and AugmentedLagrangian builder defaults
    fn test_barrier_and_alm_builders() {
        let config = BarrierBuilder::default().build();
        match config {
            LocalSolverConfig::Barrier {
                max_iter,
                mu0,
                reduction,
                duality_gap_tol,
                inner_max_iter,
            } => {
                assert_eq!(max_iter, 100);
                assert_eq!(mu0, 1.0);
                assert_eq!(reduction, 10.0);
                assert_eq!(duality_gap_tol, 1e-8);
                assert_eq!(inner_max_iter, 50);
            }
            _ => panic!("Expected Barrier local solver"),
        }
        let config = AugmentedLagrangianBuilder::default().build();
        match config {
            LocalSolverConfig::AugmentedLagrangian {
                max_iter,
                rho0,
                rho_increase,
                feasibility_decrease,
                feasibility_tol,
                inner_max_iter,
            } => {
                assert_eq!(max_iter, 100);
                assert_eq!(rho0, 10.0);
                assert_eq!(rho_increase, 10.0);
                assert_eq!(feasibility_decrease, 0.25);
                assert_eq!(feasibility_tol, 1e-8);
                assert_eq!(inner_max_iter, 50);
            }
            _ => panic!("Expected AugmentedLagrangian local solver"),
        }
    }
}
