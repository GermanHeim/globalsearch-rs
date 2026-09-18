//! Solver builders backed by Basin. Settings are validated when the local solve starts.
//! COBYLA and SLSQP accept nonlinear constraints; Barrier accepts linear
//! inequalities and AugmentedLagrangian accepts linear equalities. All other
//! solvers reject constrained problems.
//!
//! Defaults are 1,000 executor iterations and a gradient tolerance of 1e-6.
//! L-BFGS methods retain 10 correction pairs. Cost-change stopping is disabled
//! for gradient methods. Nelder-Mead uses an absolute simplex step of 0.1,
//! a simplex-size tolerance of 1e-6, and a simplex-cost tolerance of 1e-8.
//! Trust region uses Steihaug, radii 1 and 100, and eta 0.125. BOBYQA uses
//! radii 1 and 1e-6 and 2n+1 interpolation points, adjusting for narrow boxes.
//! Optional tolerances accept None to disable and zero for an exact threshold.

use super::{LocalSolverConfig, TrustRegionRadiusMethod};

/// Unconstrained L-BFGS with the default More-Thuente line search.
#[derive(Debug, Clone)]
pub struct LBFGSBuilder {
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
}

impl Default for LBFGSBuilder {
    fn default() -> Self {
        Self { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None, history_size: 10 }
    }
}

impl LBFGSBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    pub fn tolerance_grad(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_grad = value.into();
        self
    }

    /// Absolute change in cost between iterates. Disabled by default.
    pub fn tolerance_cost(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_cost = value.into();
        self
    }

    /// Number of correction pairs retained by L-BFGS. Must be positive.
    pub fn history_size(mut self, value: usize) -> Self {
        self.history_size = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::LBFGS {
            max_iter: self.max_iter,
            tolerance_grad: self.tolerance_grad,
            tolerance_cost: self.tolerance_cost,
            history_size: self.history_size,
        }
    }
}

/// Unconstrained basin gradient descent with the default More-Thuente line search and no momentum.
#[derive(Debug, Clone)]
pub struct GradientDescentBuilder {
    max_iter: u64,
    tolerance_grad: Option<f64>,
    tolerance_cost: Option<f64>,
}

impl Default for GradientDescentBuilder {
    fn default() -> Self {
        Self { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None }
    }
}

impl GradientDescentBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    pub fn tolerance_grad(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_grad = value.into();
        self
    }

    /// Absolute change in cost between iterates. Disabled by default.
    pub fn tolerance_cost(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_cost = value.into();
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::GradientDescent {
            max_iter: self.max_iter,
            tolerance_grad: self.tolerance_grad,
            tolerance_cost: self.tolerance_cost,
        }
    }
}

/// Unconstrained basin trust-region optimization using the supplied gradient and Hessian.
#[derive(Debug, Clone)]
pub struct TrustRegionBuilder {
    max_iter: u64,
    tolerance_grad: Option<f64>,
    trust_region_radius_method: TrustRegionRadiusMethod,
    radius: f64,
    max_radius: f64,
    eta: f64,
}

impl Default for TrustRegionBuilder {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance_grad: Some(1e-6),
            trust_region_radius_method: TrustRegionRadiusMethod::Steihaug,
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
        }
    }
}

impl TrustRegionBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Absolute Euclidean gradient tolerance. None disables; zero requires an exact zero.
    pub fn tolerance_grad(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_grad = value.into();
        self
    }

    /// Trust-region subproblem method, defaulting to Steihaug.
    pub fn method(mut self, value: TrustRegionRadiusMethod) -> Self {
        self.trust_region_radius_method = value;
        self
    }

    /// Positive initial trust-region radius.
    pub fn radius(mut self, value: f64) -> Self {
        self.radius = value;
        self
    }

    /// Maximum trust-region radius, at least the initial radius.
    pub fn max_radius(mut self, value: f64) -> Self {
        self.max_radius = value;
        self
    }

    /// Step acceptance threshold, in [0, 0.25).
    pub fn eta(mut self, value: f64) -> Self {
        self.eta = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::TrustRegion {
            max_iter: self.max_iter,
            tolerance_grad: self.tolerance_grad,
            trust_region_radius_method: self.trust_region_radius_method,
            radius: self.radius,
            max_radius: self.max_radius,
            eta: self.eta,
        }
    }
}

/// Unconstrained basin Nelder-Mead with standard coefficients.
#[derive(Debug, Clone)]
pub struct NelderMeadBuilder {
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
}

impl Default for NelderMeadBuilder {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            simplex_delta: 0.1,
            tolerance_simplex: Some(1e-6),
            tolerance_cost: Some(1e-8),
        }
    }
}

impl NelderMeadBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Positive absolute coordinate step for the initial simplex.
    pub fn simplex_delta(mut self, value: f64) -> Self {
        self.simplex_delta = value;
        self
    }

    /// Maximum simplex distance from its best vertex in the infinity norm.
    pub fn tolerance_simplex(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_simplex = value.into();
        self
    }

    /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
    pub fn tolerance_cost(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_cost = value.into();
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::NelderMead {
            max_iter: self.max_iter,
            simplex_delta: self.simplex_delta,
            tolerance_simplex: self.tolerance_simplex,
            tolerance_cost: self.tolerance_cost,
        }
    }
}

/// Box-constrained basin L-BFGS-B with the default More-Thuente line search.
#[derive(Debug, Clone)]
pub struct LBFGSBBuilder {
    max_iter: u64,
    tolerance_projected_grad: Option<f64>,
    tolerance_cost: Option<f64>,
    history_size: usize,
}

impl Default for LBFGSBBuilder {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance_projected_grad: Some(1e-6),
            tolerance_cost: None,
            history_size: 10,
        }
    }
}

impl LBFGSBBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Absolute projected-gradient infinity-norm tolerance. None disables.
    pub fn tolerance_projected_grad(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_projected_grad = value.into();
        self
    }

    /// Absolute change in cost between iterates. Disabled by default.
    pub fn tolerance_cost(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_cost = value.into();
        self
    }

    /// Number of correction pairs retained by L-BFGS. Must be positive.
    pub fn history_size(mut self, value: usize) -> Self {
        self.history_size = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::LBFGSB {
            max_iter: self.max_iter,
            tolerance_projected_grad: self.tolerance_projected_grad,
            tolerance_cost: self.tolerance_cost,
            history_size: self.history_size,
        }
    }
}

/// Box-constrained basin Nelder-Mead with projected trial vertices and standard coefficients.
#[derive(Debug, Clone)]
pub struct BoundedNelderMeadBuilder {
    max_iter: u64,
    simplex_delta: f64,
    tolerance_simplex: Option<f64>,
    tolerance_cost: Option<f64>,
}

impl Default for BoundedNelderMeadBuilder {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            simplex_delta: 0.1,
            tolerance_simplex: Some(1e-6),
            tolerance_cost: Some(1e-8),
        }
    }
}

impl BoundedNelderMeadBuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Positive absolute coordinate step for the initial simplex.
    pub fn simplex_delta(mut self, value: f64) -> Self {
        self.simplex_delta = value;
        self
    }

    /// Maximum simplex distance from its best vertex in the infinity norm.
    pub fn tolerance_simplex(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_simplex = value.into();
        self
    }

    /// Maximum absolute cost difference from the best simplex vertex. Enabled simplex tests combine with AND.
    pub fn tolerance_cost(mut self, value: impl Into<Option<f64>>) -> Self {
        self.tolerance_cost = value.into();
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::BoundedNelderMead {
            max_iter: self.max_iter,
            simplex_delta: self.simplex_delta,
            tolerance_simplex: self.tolerance_simplex,
            tolerance_cost: self.tolerance_cost,
        }
    }
}

/// Box-constrained basin BOBYQA. Basin reduces the radii automatically for narrow boxes.
#[derive(Debug, Clone)]
pub struct BOBYQABuilder {
    max_iter: u64,
    initial_radius: f64,
    final_radius: f64,
    interpolation_points: Option<usize>,
}

impl Default for BOBYQABuilder {
    fn default() -> Self {
        Self { max_iter: 1000, initial_radius: 1.0, final_radius: 1e-6, interpolation_points: None }
    }
}

impl BOBYQABuilder {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of basin executor iterations, excluding initialization.
    pub fn max_iter(mut self, value: u64) -> Self {
        self.max_iter = value;
        self
    }

    /// Positive initial trust-region radius, larger than the final radius.
    pub fn initial_radius(mut self, value: f64) -> Self {
        self.initial_radius = value;
        self
    }

    /// Positive final trust-region radius.
    pub fn final_radius(mut self, value: f64) -> Self {
        self.final_radius = value;
        self
    }

    /// Interpolation-set size, between 2n+1 and (n+1)(n+2)/2. None selects 2n+1.
    pub fn interpolation_points(mut self, value: Option<usize>) -> Self {
        self.interpolation_points = value;
        self
    }

    /// Builds the configuration. Validation occurs when solving.
    pub fn build(self) -> LocalSolverConfig {
        LocalSolverConfig::BOBYQA {
            max_iter: self.max_iter,
            initial_radius: self.initial_radius,
            final_radius: self.final_radius,
            interpolation_points: self.interpolation_points,
        }
    }
}
