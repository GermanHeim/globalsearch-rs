//! # Local Solver Module
//!
//! This module provides a comprehensive interface to classical optimization algorithms
//! from the `basin` crate, adapted specifically for use within the OQNLP global
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
//!   - Requires: Gradient information
//!
//! - **Gradient Descent**: Basic gradient descent with line search
//!   - Best for: Simple problems, debugging
//!   - Requires: Gradient information
//!
//! - **Trust Region**: Advanced second-order method
//!   - Best for: Smooth problems with available Hessian
//!   - Requires: Gradient and Hessian
//!
//! - **L-BFGS-B**: Box-constrained limited-memory BFGS
//!   - Best for: Smooth problems with box bounds
//!   - Requires: Gradient information
//!
//! ### Derivative-Free Methods
//! - **Nelder-Mead**: Simplex-based direct search
//!   - Best for: Non-smooth, noisy problems
//!   - Requires: Only objective function
//!
//! - **Bounded Nelder-Mead**: Projected simplex for box bounds
//!   - Best for: Non-smooth problems with box bounds
//!   - Requires: Only objective function
//!
//! - **BOBYQA**: Model-based trust region for box bounds
//!   - Best for: Expensive smooth problems with box bounds
//!   - Requires: Only objective function
//!
//! - **COBYLA**: Basin's Constrained Optimization BY Linear Approximation
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
//! ```rust
//! use globalsearch::local_solver::builders::LBFGSBuilder;
//! use globalsearch::types::OQNLPParams;
//!
//! // L-BFGS with custom tolerances
//! let lbfgs_config = LBFGSBuilder::default()
//!     .max_iter(1000)
//!     .tolerance_grad(1e-8)
//!     .history_size(10)
//!     .build();
//!
//! // Trust region with Steihaug solver
//! let trust_region_config = globalsearch::local_solver::builders::TrustRegionBuilder::default()
//!     .method(globalsearch::local_solver::builders::TrustRegionRadiusMethod::Steihaug)
//!     .max_iter(500)
//!     .radius(1.0)
//!     .build();
//!
//! // Use in OQNLP parameters
//! let params = OQNLPParams {
//!     local_solver_config: lbfgs_config,
//!     ..OQNLPParams::default()
//! };
//! ```

pub mod builders;
pub mod runner;

mod basin;
mod constrained;
