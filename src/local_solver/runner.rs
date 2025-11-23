//! # Local Solver Runner Module
//!
//! This module implements the execution engine for local optimization algorithms,
//! providing a unified interface between the OQNLP framework and the `cobyla` and `argmin`
//! crates.
//!
//! ## Architecture
//!
//! The runner acts as an adapter layer that:
//! - Converts problem definitions to `cobyla` or`argmin`-compatible formats
//! - Manages solver configuration and initialization
//! - Handles execution and result extraction
//! - Provides error handling and recovery mechanisms
//!
//! ## Supported Local Solvers
//!
//! ### Gradient-Based Algorithms
//! These methods require gradient information and are typically more efficient
//! for smooth problems:
//!
//! #### L-BFGS (Limited-memory BFGS)
//! - **Requirements**: Objective function + Gradient + Line Search
//! - **Memory**: Limited-memory quasi-Newton approximation
//! - **Line Search**: Requires compatible line search method
//!
//! #### Steepest Descent
//! - **Requirements**: Objective function + Gradient + Line Search
//! - **Method**: Simple gradient descent with line search
//! - **Line Search**: Requires compatible line search method
//!
//! #### Trust Region
//! - **Requirements**: Objective function + Gradient + Hessian
//! - **Method**: Second-order optimization with adaptive step sizing
//!
//! #### Newton-CG
//! - **Requirements**: Objective function + Gradient + Hessian + Line Search
//! - **Method**: Newton direction via conjugate gradient
//! - **Line Search**: Requires compatible line search method
//!
//! ### Derivative-Free Algorithms
//! These methods only require function evaluations and are suitable for
//! non-smooth, discontinuous, or noisy problems:
//!
//! #### Nelder-Mead
//! - **Requirements**: Objective function only
//! - **Method**: Simplex-based direct search
//!
//! #### COBYLA (Constrained Optimization BY Linear Approximation)
//! - **Requirements**: Objective function (optional constraints support)
//! - **Method**: Trust region with linear constraint approximation
//!
//! ## Error Handling
//!
//! The runner provides comprehensive error handling for common failure modes:
//! - Invalid solver configurations
//! - Numerical instabilities during optimization
//! - Function evaluation failures
//! - Convergence failures
//!
//! ## Integration with OQNLP
//!
//! Local solvers are automatically invoked by OQNLP at strategic points:
//! 1. **Reference set refinement** in Stage 1
//! 2. **Candidate solution polishing** in Stage 2

use crate::local_solver::builders::LocalSolverConfig;
#[cfg(feature = "argmin")]
use crate::local_solver::builders::{LineSearchMethod, TrustRegionRadiusMethod};
use crate::problem::Problem;
use crate::types::{LocalSolution, LocalSolverType};
#[cfg(feature = "argmin")]
use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian};
#[cfg(feature = "argmin")]
use argmin::solver::{
    gradientdescent::SteepestDescent,
    linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
    neldermead::NelderMead,
    newton::NewtonCG,
    quasinewton::LBFGS,
    trustregion::{CauchyPoint, Steihaug, TrustRegion},
};
use ndarray::Array1;
#[cfg(feature = "argmin")]
use ndarray::Array2;
use thiserror::Error;

// TODO: Do not repeat code in the linesearch branch, use helper function?

#[derive(Error, Debug, PartialEq)]
/// Local solver error enum
pub enum LocalSolverError {
    #[cfg(feature = "argmin")]
    #[error("Local Solver Error: Invalid LocalSolverConfig for L-BFGS solver. {0}")]
    InvalidLBFGSConfig(String),

    #[cfg(feature = "argmin")]
    #[error("Local Solver Error: Invalid LocalSolverConfig for Nelder-Mead solver. {0}")]
    InvalidNelderMeadConfig(String),

    #[cfg(feature = "argmin")]
    #[error("Local Solver Error: Invalid LocalSolverConfig for Steepest Descent solver. {0}")]
    InvalidSteepestDescentConfig(String),

    #[cfg(feature = "argmin")]
    #[error("Local Solver Error: Invalid LocalSolverConfig for Trust Region solver. {0}")]
    InvalidTrustRegionConfig(String),

    #[cfg(feature = "argmin")]
    #[error("Local Solver Error: Invalid LocalSolverConfig for Newton-CG method solver. {0}")]
    InvalidNewtonCG(String),

    #[error("Local Solver Error: Invalid LocalSolverConfig for COBYLA solver. {0}")]
    InvalidCOBYLAConfig(String),

    #[error("Local Solver Error: Failed to run local solver. {0}")]
    RunFailed(String),

    #[error("Local Solver Error: No solution found")]
    NoSolution,
}

/// # Local solver struct
///
/// This struct contains the problem to solve and the local solver type and configuration.
///
/// It has a `solve` method that uses a match to select the local solver function to use based on the `LocalSolverType` enum.
/// The `solve` method returns a `LocalSolution` struct.
///
/// The `LocalSolver` struct is generic over the `Problem` trait.
/// It has a problem field of type `P` and a local solver type field of type `LocalSolverType`.
/// It also has a local solver configuration field of type `LocalSolverConfig` to configure the local solver.
pub struct LocalSolver<P: Problem> {
    problem: P,
    local_solver_type: LocalSolverType,
    local_solver_config: LocalSolverConfig,
}

impl<P: Problem> LocalSolver<P> {
    pub fn new(
        problem: P,
        local_solver_type: LocalSolverType,
        local_solver_config: LocalSolverConfig,
    ) -> Self {
        Self { problem, local_solver_type, local_solver_config }
    }

    /// Solve the optimization problem using the local solver
    ///
    /// This function uses a match to select the local solver function to use based on the `LocalSolverType` enum.
    /// If `track_evaluations` is true, function evaluations will be counted (incurs small overhead).
    pub fn solve(&self, initial_point: Array1<f64>) -> Result<LocalSolution, LocalSolverError> {
        let (solution, _) = self.solve_with_tracking(initial_point, false)?;
        Ok(solution)
    }

    /// Solve with optional function evaluation tracking
    pub fn solve_with_tracking(
        &self,
        initial_point: Array1<f64>,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        match self.local_solver_type {
            #[cfg(feature = "argmin")]
            LocalSolverType::LBFGS => {
                self.solve_lbfgs(initial_point, &self.local_solver_config, track_evaluations)
            }
            #[cfg(feature = "argmin")]
            LocalSolverType::NelderMead => {
                self.solve_nelder_mead(initial_point, &self.local_solver_config, track_evaluations)
            }
            #[cfg(feature = "argmin")]
            LocalSolverType::SteepestDescent => self.solve_steepestdescent(
                initial_point,
                &self.local_solver_config,
                track_evaluations,
            ),
            #[cfg(feature = "argmin")]
            LocalSolverType::TrustRegion => {
                self.solve_trust_region(initial_point, &self.local_solver_config, track_evaluations)
            }
            #[cfg(feature = "argmin")]
            LocalSolverType::NewtonCG => {
                self.solve_newton_cg(initial_point, &self.local_solver_config, track_evaluations)
            }
            LocalSolverType::COBYLA => {
                self.solve_cobyla(initial_point, &self.local_solver_config, track_evaluations)
            }
        }
    }

    /// Solve the optimization problem using the L-BFGS local solver
    #[cfg(feature = "argmin")]
    fn solve_lbfgs(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
            eval_count: Option<Arc<AtomicU64>>,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                if let Some(counter) = &self.eval_count {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.problem.objective(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem.gradient(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        let eval_count = if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let cost = ProblemCost { problem: &self.problem, eval_count: eval_count.clone() };

        if let LocalSolverConfig::LBFGS {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
            l1_coefficient,
            line_search_params,
        } = solver_config
        {
            // Match line search method
            match &line_search_params.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    let linesearch = MoreThuenteLineSearch::new()
                        .with_c(*c1, *c2)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_width_tolerance(*width_tolerance)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let mut solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_tolerance_grad(*tolerance_grad)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    if let Some(l1_coeff) = l1_coefficient {
                        solver = solver.with_l1_regularization(*l1_coeff).map_err(|e: Error| {
                            LocalSolverError::InvalidLBFGSConfig(e.to_string())
                        })?;
                    }

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    let linesearch = HagerZhangLineSearch::new()
                        .with_delta_sigma(*delta, *sigma)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_epsilon(*epsilon)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_theta(*theta)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_gamma(*gamma)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_eta(*eta)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    let mut solver = LBFGS::new(linesearch, *history_size)
                        .with_tolerance_cost(*tolerance_cost)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?
                        .with_tolerance_grad(*tolerance_grad)
                        .map_err(|e: Error| LocalSolverError::InvalidLBFGSConfig(e.to_string()))?;

                    if let Some(l1_coeff) = l1_coefficient {
                        solver = solver.with_l1_regularization(*l1_coeff).map_err(|e: Error| {
                            LocalSolverError::InvalidLBFGSConfig(e.to_string())
                        })?;
                    }

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
            }
        } else {
            Err(LocalSolverError::InvalidLBFGSConfig("Error parsing solver config".to_string()))
        }
    }

    /// Solve the optimization problem using the Nelder-Mead local solver
    #[cfg(feature = "argmin")]
    fn solve_nelder_mead(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
            eval_count: Option<Arc<AtomicU64>>,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                if let Some(counter) = &self.eval_count {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.problem.objective(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        let eval_count = if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let cost = ProblemCost { problem: &self.problem, eval_count: eval_count.clone() };

        if let LocalSolverConfig::NelderMead {
            simplex_delta,
            sd_tolerance,
            max_iter,
            alpha,
            gamma,
            rho,
            sigma,
        } = solver_config
        {
            // Generate initial simplex
            let mut simplex = vec![initial_point.clone()];
            for i in 0..initial_point.len() {
                let mut point = initial_point.clone();
                point[i] += simplex_delta;
                simplex.push(point);
            }

            let solver = NelderMead::new(simplex)
                .with_sd_tolerance(*sd_tolerance)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_alpha(*alpha)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_gamma(*gamma)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_rho(*rho)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?
                .with_sigma(*sigma)
                .map_err(|e: Error| LocalSolverError::InvalidNelderMeadConfig(e.to_string()))?;

            let res = Executor::new(cost, solver)
                .configure(|state| state.max_iters(*max_iter))
                .run()
                .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

            let solution = LocalSolution {
                point: res.state().best_param.as_ref().ok_or(LocalSolverError::NoSolution)?.clone(),
                objective: res.state().best_cost,
            };
            let evaluations = eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
            Ok((solution, evaluations))
        } else {
            Err(LocalSolverError::InvalidNelderMeadConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }

    /// Solve the optimization problem using the Steepest Descent local solver
    #[cfg(feature = "argmin")]
    fn solve_steepestdescent(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
            eval_count: Option<Arc<AtomicU64>>,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                if let Some(counter) = &self.eval_count {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.problem.objective(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem.gradient(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        let eval_count = if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let cost = ProblemCost { problem: &self.problem, eval_count: eval_count.clone() };

        if let LocalSolverConfig::SteepestDescent { max_iter, line_search_params } = solver_config {
            // Match line search method
            match &line_search_params.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    let linesearch = MoreThuenteLineSearch::new()
                        .with_c(*c1, *c2)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_width_tolerance(*width_tolerance)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    let linesearch = HagerZhangLineSearch::new()
                        .with_delta_sigma(*delta, *sigma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_epsilon(*epsilon)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_theta(*theta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_gamma(*gamma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = SteepestDescent::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
            }
        } else {
            Err(LocalSolverError::InvalidSteepestDescentConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }

    /// Solve the optimization problem using the Trust Region local solver
    #[cfg(feature = "argmin")]
    fn solve_trust_region(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
            eval_count: Option<Arc<AtomicU64>>,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                if let Some(counter) = &self.eval_count {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.problem.objective(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem.gradient(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Hessian for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, param: &Self::Param) -> std::result::Result<Self::Hessian, Error> {
                self.problem.hessian(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        let eval_count = if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let cost = ProblemCost { problem: &self.problem, eval_count: eval_count.clone() };

        if let LocalSolverConfig::TrustRegion {
            trust_region_radius_method,
            max_iter,
            radius,
            max_radius,
            eta,
        } = solver_config
        {
            match trust_region_radius_method {
                TrustRegionRadiusMethod::Cauchy => {
                    let subproblem = CauchyPoint::new();
                    let solver = TrustRegion::new(subproblem)
                        .with_radius(*radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_max_radius(*max_radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?;
                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
                TrustRegionRadiusMethod::Steihaug => {
                    let subproblem = Steihaug::new();
                    let solver = TrustRegion::new(subproblem)
                        .with_radius(*radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_max_radius(*max_radius)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidTrustRegionConfig(e.to_string())
                        })?;
                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
            }
        } else {
            Err(LocalSolverError::InvalidTrustRegionConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }

    #[cfg(feature = "argmin")]
    fn solve_newton_cg(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        struct ProblemCost<'a, P: Problem> {
            problem: &'a P,
            eval_count: Option<Arc<AtomicU64>>,
        }

        impl<P: Problem> CostFunction for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> std::result::Result<Self::Output, Error> {
                if let Some(counter) = &self.eval_count {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                self.problem.objective(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Gradient for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, param: &Self::Param) -> std::result::Result<Self::Gradient, Error> {
                self.problem.gradient(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        impl<P: Problem> Hessian for ProblemCost<'_, P> {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, param: &Self::Param) -> std::result::Result<Self::Hessian, Error> {
                self.problem.hessian(param).map_err(|e| Error::msg(e.to_string()))
            }
        }

        let eval_count = if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
        let cost = ProblemCost { problem: &self.problem, eval_count: eval_count.clone() };

        if let LocalSolverConfig::NewtonCG {
            max_iter,
            curvature_threshold,
            tolerance,
            line_search_params,
        } = solver_config
        {
            match &line_search_params.method {
                LineSearchMethod::MoreThuente { c1, c2, width_tolerance, bounds } => {
                    let linesearch = MoreThuenteLineSearch::new()
                        .with_c(*c1, *c2)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_width_tolerance(*width_tolerance)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = NewtonCG::new(linesearch)
                        .with_curvature_threshold(*curvature_threshold)
                        .with_tolerance(*tolerance)
                        .map_err(|e: Error| LocalSolverError::InvalidNewtonCG(e.to_string()))?;

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
                LineSearchMethod::HagerZhang {
                    delta,
                    sigma,
                    epsilon,
                    theta,
                    gamma,
                    eta,
                    bounds,
                } => {
                    let linesearch = HagerZhangLineSearch::new()
                        .with_delta_sigma(*delta, *sigma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_epsilon(*epsilon)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_theta(*theta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_gamma(*gamma)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_eta(*eta)
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?
                        .with_bounds(bounds[0], bounds[1])
                        .map_err(|e: Error| {
                            LocalSolverError::InvalidSteepestDescentConfig(e.to_string())
                        })?;

                    let solver = NewtonCG::new(linesearch);

                    let res = Executor::new(cost, solver)
                        .configure(|state| state.param(initial_point).max_iters(*max_iter))
                        .run()
                        .map_err(|e: Error| LocalSolverError::RunFailed(e.to_string()))?;

                    let solution = LocalSolution {
                        point: res
                            .state()
                            .best_param
                            .as_ref()
                            .ok_or(LocalSolverError::NoSolution)?
                            .clone(),
                        objective: res.state().best_cost,
                    };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
            }
        } else {
            Err(LocalSolverError::InvalidNewtonCG("Error parsing solver configuration".to_string()))
        }
    }

    /// Solve the optimization problem using the COBYLA local solver
    fn solve_cobyla(
        &self,
        initial_point: Array1<f64>,
        solver_config: &LocalSolverConfig,
        track_evaluations: bool,
    ) -> Result<(LocalSolution, u64), LocalSolverError> {
        use std::sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        };

        #[cfg_attr(not(feature = "argmin"), allow(irrefutable_let_patterns))]
        if let LocalSolverConfig::COBYLA {
            max_iter,
            initial_step_size,
            ftol_rel,
            ftol_abs,
            xtol_rel,
            xtol_abs,
        } = solver_config
        {
            // Convert initial point to Vec<f64> as required by COBYLA
            let x0: Vec<f64> = initial_point.to_vec();

            // Conditionally track function evaluations
            let eval_count =
                if track_evaluations { Some(Arc::new(AtomicU64::new(0))) } else { None };
            let eval_count_for_closure = eval_count.clone();

            // Create the objective function for COBYLA (needs 2 arguments: x and user_data)
            let objective = move |x: &[f64], _user_data: &mut ()| -> f64 {
                if let Some(ref counter) = eval_count_for_closure {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                let point = Array1::from_vec(x.to_vec());

                match self.problem.objective(&point) {
                    Ok(value) => value,
                    Err(_) => f64::INFINITY,
                }
            };

            let constraint_funcs = self.problem.constraints();
            let problem_bounds = self.problem.variable_bounds();

            if problem_bounds.nrows() != x0.len() {
                return Err(LocalSolverError::InvalidCOBYLAConfig(
                    format!("Problem bounds dimension mismatch: expected {} bounds for {} variables, got {} bounds", 
                           x0.len(), x0.len(), problem_bounds.nrows())
                ));
            }

            let bounds: Vec<(f64, f64)> =
                (0..x0.len()).map(|i| (problem_bounds[[i, 0]], problem_bounds[[i, 1]])).collect();

            match cobyla::minimize(
                objective,
                &x0,
                &bounds,
                &constraint_funcs,
                (),
                *max_iter as usize,
                cobyla::RhoBeg::All(*initial_step_size),
                Some(cobyla::StopTols {
                    ftol_rel: *ftol_rel,
                    ftol_abs: *ftol_abs,
                    xtol_rel: *xtol_rel,
                    xtol_abs: if xtol_abs.is_empty() { vec![] } else { xtol_abs.clone() },
                }),
            ) {
                Ok((_status, solution_x, objective_value)) => {
                    let solution_point = Array1::from_vec(solution_x);
                    let solution =
                        LocalSolution { point: solution_point, objective: objective_value };
                    let evaluations =
                        eval_count.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(0);
                    Ok((solution, evaluations))
                }
                Err(e) => {
                    Err(LocalSolverError::RunFailed(format!("COBYLA solver failed: {:?}", e)))
                }
            }
        } else {
            Err(LocalSolverError::InvalidCOBYLAConfig(
                "Error parsing solver configuration".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests_local_solvers {
    use super::*;
    #[cfg(feature = "argmin")]
    use crate::local_solver::builders::{
        HagerZhangBuilder, LBFGSBuilder, MoreThuenteBuilder, SteepestDescentBuilder,
    };
    use crate::types::{EvaluationError, LocalSolverType};
    use ndarray::{array, Array2};

    #[derive(Debug, Clone)]
    pub struct NoGradientSixHumpCamel;

    impl Problem for NoGradientSixHumpCamel {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok((4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                + x[0] * x[1]
                + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2))
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-3.0, 3.0], [-2.0, 2.0]]
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConstrainedQuadratic;

    impl Problem for ConstrainedQuadratic {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            // Simple quadratic: (x-1)² + (y-1)²
            Ok((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2))
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[0.0, 2.0], [0.0, 2.0]]
        }

        fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
            vec![
                |x: &[f64], _: &mut ()| 1.5 - x[0] - x[1], // x + y <= 1.5 -> 1.5 - x - y >= 0
            ]
        }
    }

    #[derive(Debug, Clone)]
    pub struct NoHessianSixHumpCamel;

    impl Problem for NoHessianSixHumpCamel {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok((4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                + x[0] * x[1]
                + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2))
        }

        // Calculated analytically, reference didn't provide gradient
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![
                (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
                x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
            ])
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-3.0, 3.0], [-2.0, 2.0]]
        }
    }

    // Tests for argmin-based solvers

    #[cfg(feature = "argmin")]
    #[test]
    /// Test the Nelder-Mead local solver with a problem that doesn't
    /// have a gradient. Since Nelder-Mead doesn't require a gradient,
    /// the local solver should run without an error.
    fn test_nelder_mead_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::NelderMead,
            LocalSolverConfig::NelderMead {
                simplex_delta: 0.1,
                sd_tolerance: 1e-6,
                max_iter: 1000,
                alpha: 1.0,
                gamma: 2.0,
                rho: 0.5,
                sigma: 0.5,
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let res: LocalSolution = local_solver.solve(initial_point).unwrap();
        assert_eq!(res.objective, -1.0316278623977673);
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test the Steepest Descent local solver with a problem that doesn't
    /// have a gradient. Since Steepest Descent requires a gradient,
    /// the local solver should return an error.
    fn test_steepest_descent_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::SteepestDescent,
            SteepestDescentBuilder::default().build(),
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Gradient not implemented and needed for local solver.".to_string()
            )
        );
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test the L-BFGS local solver with a problem that doesn't
    /// have a gradient. Since L-BFGS requires a gradient,
    /// the local solver should return an error.
    fn test_lbfgs_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> =
            LocalSolver::new(problem, LocalSolverType::LBFGS, LBFGSBuilder::default().build());

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Gradient not implemented and needed for local solver.".to_string()
            )
        );
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test the Newton CG local solver with a problem that doesn't
    /// have a gradient. Since Newton CG requires a gradient and a hessian,
    /// the local solver should return an error.
    fn test_newton_cg_no_gradient_hessian() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::NewtonCG,
            LocalSolverConfig::NewtonCG {
                max_iter: 1000,
                curvature_threshold: 1e-6,
                tolerance: 1e-6,
                line_search_params: HagerZhangBuilder::default().build(),
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Gradient not implemented and needed for local solver.".to_string()
            )
        );

        let problem: NoHessianSixHumpCamel = NoHessianSixHumpCamel;

        let local_solver: LocalSolver<NoHessianSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::NewtonCG,
            LocalSolverConfig::NewtonCG {
                max_iter: 1000,
                curvature_threshold: 1e-6,
                tolerance: 1e-6,
                line_search_params: HagerZhangBuilder::default().build(),
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::RunFailed(
                "Hessian not implemented and needed for local solver.".to_string()
            )
        );
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test creating a HagerZhangLineSearch instance with an invalid configurations
    fn invalid_hagerzhang() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;
        let initial_point: Array1<f64> = array![0.0, 0.0];

        // Invalid delta value
        // Delta must be in (0, 1) and sigma must be in [delta, 1)
        // Here we set it to 2.0
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().delta(2.0).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();

        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: delta must be in (0, 1) and sigma must be in [delta, 1).\""
                    .to_string()
            )
        );

        // Invalid sigma value
        // Delta must be in (0, 1) and sigma must be in [delta, 1)
        // Here we set delta to 0.7 and sigma to 0.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().delta(0.7).sigma(0.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: delta must be in (0, 1) and sigma must be in [delta, 1).\""
                    .to_string()
            )
        );

        // Invalid epsilon value
        // Epsilon must be non-negative
        // Here we set epsilon to -0.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().epsilon(-0.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: epsilon must be >= 0.\"".to_string()
            )
        );

        // Invalid theta value
        // Theta must be in (0, 1)
        // Here we set theta to 1.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().theta(1.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: theta must be in (0, 1).\""
                    .to_string()
            )
        );

        // Invalid gamma value
        // Gamma must be in (0, 1)
        // Here we set gamma to 1.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().gamma(1.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: gamma must be in (0, 1).\""
                    .to_string()
            )
        );

        // Invalid eta value
        // Eta must be larger than zero
        // Here we set eta to -0.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().eta(-0.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: eta must be > 0.\"".to_string()
            )
        );

        // Invalid bounds value
        // Bounds must be a tuple with two values, where the first value
        // (step_min) is smaller than the second value (step_max)
        // both values should be higher or equal to zero
        // Here we set bounds to [1.0, 0.0]
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: HagerZhangBuilder::default().bounds(array![1.0, 0.0]).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`HagerZhangLineSearch`: minimum and maximum step length must be chosen such that 0 <= step_min < step_max.\""
                    .to_string()
            )
        );
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test creating a MoreThuenteLineSearch instance with an invalid configurations
    fn invalid_morethuente() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;
        let initial_point: Array1<f64> = array![0.0, 0.0];

        // Invalid c1 and c2 values
        // c1 and c2 must be in (0, 1) and c1 < c2
        // Here we set c1 to 1.0 and c2 to 0.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: MoreThuenteBuilder::default().c1(1.0).c2(0.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`MoreThuenteLineSearch`: Parameter c1 must be in (0, c2).\""
                    .to_string()
            )
        );

        // Invalid bounds value
        // Bounds must be a tuple with two values, where the first value
        // (step_min) is smaller than the second value (step_max)
        // Here we set bounds to [1.0, 0.0]
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: MoreThuenteBuilder::default().bounds(array![1.0, 0.0]).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point.clone()).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`MoreThuenteLineSearch`: step_min must be smaller than step_max.\""
                    .to_string()
            )
        );

        // Invalid width_tolerance value
        // Width tolerance must be larger than zero
        // Here we set width_tolerance to -0.5
        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::LBFGS,
            LocalSolverConfig::LBFGS {
                max_iter: 1000,
                tolerance_grad: 1e-6,
                tolerance_cost: 1e-6,
                history_size: 5,
                l1_coefficient: None,
                line_search_params: MoreThuenteBuilder::default().width_tolerance(-0.5).build(),
            },
        );

        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();
        assert_eq!(
            error,
            LocalSolverError::InvalidLBFGSConfig(
                "Invalid parameter: \"`MoreThuenteLineSearch`: relative width tolerance must be >= 0.0.\""
                    .to_string()
            )
        );
    }

    #[cfg(feature = "argmin")]
    #[test]
    /// Test creating a Trust Region solver using an invalid eta value
    /// In this case, eta must be in [0, 1/4) and we set it to 1.0
    fn invalid_trust_region_eta() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem,
            LocalSolverType::TrustRegion,
            LocalSolverConfig::TrustRegion {
                trust_region_radius_method: TrustRegionRadiusMethod::Steihaug,
                max_iter: 1000,
                radius: 0.1,
                max_radius: 1.0,
                eta: 1.0,
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let error: LocalSolverError = local_solver.solve(initial_point).unwrap_err();

        assert_eq!(
            error,
            LocalSolverError::InvalidTrustRegionConfig(
                "Invalid parameter: \"`TrustRegion`: eta must be in [0, 1/4).\"".to_string()
            )
        );
    }

    // COBYLA tests (always available)

    #[test]
    /// Test the COBYLA local solver with a problem that doesn't
    /// have a gradient. Since COBYLA doesn't require a gradient,
    /// the local solver should run without an error.
    fn test_cobyla_no_gradient() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::COBYLA,
            LocalSolverConfig::COBYLA {
                max_iter: 1000,
                initial_step_size: 1.0,
                ftol_rel: 1e-6,
                ftol_abs: 1e-8,
                xtol_rel: 0.0,
                xtol_abs: vec![],
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];
        let res: LocalSolution = local_solver.solve(initial_point).unwrap();
        // COBYLA should find a reasonable solution for the Six Hump Camel function
        // The global minimum is around -1.0316, but COBYLA might not find the exact global minimum
        assert!(res.objective < 0.0); // Should at least find a negative value
    }

    #[test]
    /// Test COBYLA with constraints using a simple quadratic problem
    fn test_cobyla_with_constraints() {
        let problem: ConstrainedQuadratic = ConstrainedQuadratic;

        let local_solver: LocalSolver<ConstrainedQuadratic> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::COBYLA,
            LocalSolverConfig::COBYLA {
                max_iter: 500,
                initial_step_size: 0.5,
                ftol_rel: 1e-6,
                ftol_abs: 1e-8,
                xtol_rel: 0.0,
                xtol_abs: vec![],
            },
        );

        let initial_point: Array1<f64> = array![0.5, 0.5];
        let res: LocalSolution = local_solver.solve(initial_point).unwrap();

        // Check that the solution respects bounds
        assert!(res.point[0] >= 0.0 && res.point[0] <= 2.0);
        assert!(res.point[1] >= 0.0 && res.point[1] <= 2.0);

        // Check that the constraint is approximately satisfied (with tolerance)
        let constraint_value = res.point[0] + res.point[1] - 1.5;
        assert!(constraint_value <= 0.01); // Small tolerance for numerical errors

        // The constrained optimum should be around (0.75, 0.75) with objective ~0.125
        // With penalty method, the result may be slightly different
        let expected_obj = 0.125;
        assert!(
            (res.objective - expected_obj).abs() < 0.2,
            "Expected objective ~{}, got {}",
            expected_obj,
            res.objective
        );
    }

    #[test]
    /// Test that constraint evaluation works correctly
    fn test_constraint_evaluation() {
        let problem = ConstrainedQuadratic;
        let constraints = problem.constraints();

        // Test constraint at a point that satisfies it
        let feasible_point = array![0.5, 0.5];
        let constraint_val = constraints[0](&[feasible_point[0], feasible_point[1]], &mut ());
        assert!(constraint_val > 0.0); // Should be positive (satisfied in COBYLA convention)

        // Test constraint at a point that violates it
        let infeasible_point = array![1.0, 1.0];
        let constraint_val = constraints[0](&[infeasible_point[0], infeasible_point[1]], &mut ());
        assert!(constraint_val < 0.0); // Should be negative (violated in COBYLA convention)
    }

    #[test]
    /// Test that COBYLA tracks function evaluations correctly
    fn test_cobyla_tracks_evaluations() {
        let problem: NoGradientSixHumpCamel = NoGradientSixHumpCamel;

        let local_solver: LocalSolver<NoGradientSixHumpCamel> = LocalSolver::new(
            problem.clone(),
            LocalSolverType::COBYLA,
            LocalSolverConfig::COBYLA {
                max_iter: 100,
                initial_step_size: 1.0,
                ftol_rel: 1e-6,
                ftol_abs: 1e-8,
                xtol_rel: 0.0,
                xtol_abs: vec![],
            },
        );

        let initial_point: Array1<f64> = array![0.0, 0.0];

        // Test with tracking enabled
        let (res, eval_count) =
            local_solver.solve_with_tracking(initial_point.clone(), true).unwrap();
        assert!(
            eval_count > 0,
            "COBYLA should track function evaluations when enabled, got {}",
            eval_count
        );
        assert!(res.objective < 0.0);

        // Test with tracking disabled
        let (res2, eval_count2) = local_solver.solve_with_tracking(initial_point, false).unwrap();
        assert_eq!(
            eval_count2, 0,
            "COBYLA should return 0 evaluations when tracking disabled, got {}",
            eval_count2
        );
        assert!(res2.objective < 0.0);
    }
}
