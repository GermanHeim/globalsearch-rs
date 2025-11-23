//! # Local Solver Module
//!
//! This module provides a comprehensive interface to classical optimization algorithms
//! from the `cobyla` and `argmin` crates, adapted specifically for use within the OQNLP global
//! optimization framework.
//!
//! ## Module Structure
//!
//! - [`builders`] - Configuration builders for all supported local solvers
//! - [`runner`] - Execution engine that runs configured solvers on problems
//!
//! ## Supported Local Solvers
//!
//! ### Gradient-Based Methods
//! - **L-BFGS**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno
//!   - Best for: Smooth, unconstrained problems
//!   - Requires: Gradient information and Line Search method
//!
//! - **Steepest Descent**: Basic gradient descent with line search
//!   - Best for: Simple problems, debugging
//!   - Requires: Gradient information and Line Search method
//!
//! - **Trust Region**: Advanced second-order method
//!   - Best for: Smooth problems with available Hessian
//!   - Requires: Gradient and Hessian
//!
//! - **Newton-CG**: Newton method with conjugate gradient
//!   - Best for: Large-scale smooth problems
//!   - Requires: Gradient, Hessian and Line Search method
//!
//! ### Derivative-Free Methods
//! - **Nelder-Mead**: Simplex-based direct search
//!   - Best for: Non-smooth, noisy problems
//!   - Requires: Only objective function
//!
//! - **COBYLA**: Constrained Optimization BY Linear Approximation
//!   - Best for: Constrained problems without derivatives
//!   - Requires: Objective (optional constraints support)
//!
//! ## Usage in OQNLP
//!
//! Local solvers are automatically applied during the OQNLP optimization process:
//! 1. **Stage 1**: Refine initial scattered solutions
//! 2. **Stage 2**: Polish newly generated candidate solutions
//!
//! ## Configuration Example
//!
//! ```ignore
//! use globalsearch::local_solver::builders::{
//!     LBFGSBuilder, HagerZhangBuilder, TrustRegionBuilder, TrustRegionRadiusMethod
//! };
//! use globalsearch::types::{LocalSolverType, OQNLPParams};
//!
//! // L-BFGS with custom line search
//! let lbfgs_config = LBFGSBuilder::default()
//!     .max_iter(1000)
//!     .tolerance_grad(1e-8)
//!     .line_search_params(HagerZhangBuilder::default()
//!         .delta(0.1)
//!         .sigma(0.9)
//!         .build())
//!     .build();
//!
//! // Trust region with Steihaug solver
//! let trust_region_config = TrustRegionBuilder::default()
//!     .method(TrustRegionRadiusMethod::Steihaug)
//!     .max_iter(500)
//!     .radius(1.0)
//!     .build();
//!
//! // Use in OQNLP parameters
//! let params = OQNLPParams {
//!     local_solver_type: LocalSolverType::LBFGS,
//!     local_solver_config: lbfgs_config,
//!     ..OQNLPParams::default()
//! };
//! ```

pub mod builders;
pub mod runner;
