//! # OQNLP (OptQuest for Nonlinear Programming) Module
//!
//! The OQNLP (OptQuest/NLP) algorithm is a global optimization algorithm that combines scatter search with local optimization methods.
//! This module implements the OQNLP global optimization algorithm, which combines
//! the systematic exploration of scatter search with the local refinement capabilities
//! of classical optimization methods.
//!
//! ## Algorithm Overview
//!
//! OQNLP is a hybrid metaheuristic that operates through a two-stage process:
//!
//! ### Stage 1: Reference Set Construction
//! - **Population Initialization**: Generate diverse candidate solutions within bounds
//! - **Quality Evaluation**: Assess all candidates using the objective function
//! - **Local Optimization**: Apply local solvers to the best candidate
//!
//! ### Stage 2: Iterative Improvement
//! - **Merit Filtering**: Optimize only solutions meeting quality thresholds
//! - **Distance Filtering**: Ensure sufficient diversity between solutions
//! - **Local Refinement**: Use local solvers to polish selected candidates
//! - **Solution Set Update**: Replace inferior solutions with new discoveries
//!
//! ## Key Features
//!
//! ### Global Search Capability
//! - **Systematic Exploration**: Scatter search prevents premature convergence
//! - **Diversity Maintenance**: Distance filters ensure broad search coverage
//! - **Multi-modal Optimization**: Can find multiple local optima simultaneously
//!
//! ### Local Refinement
//! - **Gradient-Based Methods**: L-BFGS, trust region, Newton-CG for smooth problems
//! - **Derivative-Free Methods**: Nelder-Mead, COBYLA for non-smooth problems
//! - **Flexible Configuration**: Easily switch between different local solvers
//!
//! ### Robustness Features
//! - **Checkpointing Support**: Resume long optimizations from saved states
//! - **Constraint Handling**: Support for bound and general nonlinear constraints using COBYLA
//! - **Error Recovery**: Graceful handling of function evaluation failures
//!
//! ## Mathematical Formulation
//!
//! OQNLP solves optimization problems of the form:
//!
//! ```text
//! minimize    f(x)
//! subject to  lₙ ≤ xₙ ≤ uₙ    (bound constraints)
//!             gₙ(x) ≥ 0        (inequality constraints)
//! ```
//!
//! Where:
//! - `f(x)`: Objective function to minimize
//! - `x`: Decision variables vector
//! - `lₙ, uₙ`: Lower and upper bounds for variable i
//! - `gₙ(x)`: Inequality constraint functions
//!
//! ## Example: Six-Hump Camel Function
//!
//! ```rust
//! // Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 27-28. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
//! use globalsearch::local_solver::builders::{TrustRegionBuilder, TrustRegionRadiusMethod};
//! use globalsearch::problem::Problem;
//! use globalsearch::{
//!     oqnlp::OQNLP,
//!     types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
//! };
//! use ndarray::{array, Array1, Array2};
//!
//! #[derive(Debug, Clone)]
//! pub struct SixHumpCamel;
//!
//! impl Problem for SixHumpCamel {
//!    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//!       Ok(
//!          (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
//!            + x[0] * x[1]
//!            + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
//!          )
//!     }
//!
//!     // Calculated analytically, reference didn't provide gradient
//!     fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
//!         Ok(array![
//!             (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
//!             x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
//!         ])
//!     }
//!
//!     // Calculated analytically, reference didn't provide hessian
//!     fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
//!         Ok(array![
//!             [(4.0 * x[0].powi(2) - 4.2) * x[0].powi(2)
//!                 + 4.0 * (4.0 / 3.0 * x[0].powi(3) - 4.2 * x[0]) * x[0]
//!                 + 2.0 * (x[0].powi(4) / 3.0 - 2.1 * x[0].powi(2) + 4.0),
//!             1.0],
//!             [1.0, 40.0 * x[1].powi(2) + 2.0 * (4.0 * x[1].powi(2) - 4.0)]])
//!     }
//!
//!     fn variable_bounds(&self) -> Array2<f64> {
//!         array![[-3.0, 3.0], [-2.0, 2.0]]
//!     }
//! }
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Solving the Six-Hump Camel problem using globalsearch-rs
//!     // using the Trust Region method as a local solver
//!     let problem: SixHumpCamel = SixHumpCamel;
//!
//!     // Set the parameters for the OQNLP algorithm
//!     // It is recommended that you adjust these parameters to your problem
//!     // and the desired behavior of the algorithm, instead of using the default values.
//!     let params: OQNLPParams = OQNLPParams {
//!         local_solver_type: LocalSolverType::TrustRegion,
//!         local_solver_config: TrustRegionBuilder::default()
//!             .method(TrustRegionRadiusMethod::Steihaug)
//!             .build(),
//!             ..OQNLPParams::default()
//!     };
//!
//!     let mut oqnlp: OQNLP<SixHumpCamel> = OQNLP::new(problem, params).unwrap();
//!     let sol_set: SolutionSet = oqnlp.run().unwrap();
//!     println!("Solution set:");
//!     println!("{}", sol_set);
//!     Ok(())
//!
//! }
//! ```

use crate::filters::{DistanceFilter, MeritFilter};
use crate::local_solver::runner::LocalSolver;
use crate::observers::Observer;
use crate::problem::Problem;
use crate::scatter_search::ScatterSearch;
use crate::types::{FilterParams, LocalSolution, OQNLPParams, SolutionSet};
#[cfg(feature = "checkpointing")]
use crate::{
    checkpoint::{CheckpointError, CheckpointManager},
    types::{CheckpointConfig, OQNLPCheckpoint},
};
#[cfg(feature = "checkpointing")]
use chrono;
#[cfg(feature = "progress_bar")]
use kdam::{Bar, BarExt};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use thiserror::Error;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug, Error)]
/// ONQLP errors
pub enum OQNLPError {
    /// Error when the local solver fails to find a solution
    #[error("OQNLP Error: Local solver failed: {0}")]
    LocalSolverError(#[from] crate::local_solver::runner::LocalSolverError),

    /// Error when OQNLP fails to find a feasible solution
    #[error("OQNLP Error: No feasible solution found.")]
    NoFeasibleSolution,

    /// Error when the objective function evaluation fails
    #[error("OQNLP Error: Objective function evaluation failed.")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when creating a new ScatterSearch instance
    #[error("OQNLP Error: Failed to create a new ScatterSearch instance: {0}")]
    ScatterSearchError(#[from] crate::scatter_search::ScatterSearchError),

    /// Error when running the ScatterSearch instance
    #[error("OQNLP Error: Failed to run the ScatterSearch instance: {0}")]
    ScatterSearchRunError(crate::scatter_search::ScatterSearchError),

    /// Error when the population size is invalid
    ///
    /// Population size should be at least 3, since the reference set
    /// pushes the bounds and the midpoint.
    #[error("OQNLP Error: Population size should be at least 3, got {0}. Reference Set size should be at least 3, since it pushes the bounds and the midpoint.")]
    InvalidPopulationSize(usize),

    /// Error when the iterations are invalid
    ///
    /// Iterations should be less than or equal to the population size.
    #[error("OQNLP Error: Iterations should be less than or equal to population size. OQNLP received `iterations`: {0}, `population size`: {1}.")]
    InvalidIterations(usize, usize),

    /// Error when creating the distance filter
    #[error("OQNLP Error: Failed to create distance filter: {0}")]
    DistanceFilterError(#[from] crate::filters::FiltersErrors),

    /// Error when the threshold factor is invalid
    ///
    /// Threshold factor should be positive (greater than 0.0).
    #[error("OQNLP Error: Threshold factor must be positive, got {0}.")]
    InvalidThresholdFactor(f64),

    /// Error when custom points have invalid dimensions
    #[error("OQNLP Error: Custom points must have the same dimension as the problem. Expected {expected} dimensions, got {got}.")]
    InvalidCustomPointsDimension { expected: usize, got: usize },

    /// Error when custom points are outside variable bounds
    #[error("OQNLP Error: Custom point at index {index} is outside variable bounds.")]
    CustomPointOutOfBounds { index: usize },

    /// Error related to checkpointing operations
    #[cfg(feature = "checkpointing")]
    #[error("OQNLP Error: Checkpointing error: {0}")]
    CheckpointError(#[from] CheckpointError),
}

/// Main OQNLP optimization algorithm implementation.
///
/// This struct encapsulates the complete OQNLP (OptQuest for Nonlinear Programming) algorithm,
/// which combines scatter search with local optimization methods to solve global optimization
/// problems efficiently.
///
/// ## Algorithm Components
///
/// The OQNLP struct manages several key algorithmic components:
///
/// ### Core Algorithm Elements
/// - **Problem Definition**: The optimization problem implementing the [`Problem`] trait
/// - **Algorithm Parameters**: Configuration via [`OQNLPParams`] controlling behavior
/// - **Filtering Mechanisms**: Merit and distance filters for solution quality and diversity
/// - **Local Solver**: Refinement of candidate solutions using classical optimization methods
///
/// ### Solution Management
/// - **Solution Set**: Collection of high-quality solutions discovered during optimization
/// - **Reference Set**: Internal population of candidate solutions for scatter search
/// - **Checkpointing**: Optional state persistence for long-running optimizations
///
/// ## Optimization Process
///
/// The algorithm operates through a two-stage process:
///
/// 1. **Stage 1 (Initialization)**: Build initial reference set via scatter search
/// 2. **Stage 2 (Refinement)**: Iteratively improve solutions through local optimization
///
/// ## Configuration Options
///
#[cfg_attr(feature = "rayon", doc = "")]
#[cfg_attr(feature = "rayon", doc = "### Parallel Processing")]
#[cfg_attr(feature = "rayon", doc = "")]
#[cfg_attr(
    feature = "rayon",
    doc = "- [`batch_iterations()`](OQNLP::batch_iterations): Set batch size for parallel processing of stage two iterations"
)]
#[cfg_attr(feature = "rayon", doc = "")]
/// ### Time Control
///
/// - [`max_time()`](OQNLP::max_time): Set maximum optimization time
/// - [`target_objective()`](OQNLP::target_objective): Early stopping criterion
///
/// ### Solution Quality
///
/// - [`exclude_out_of_bounds()`](OQNLP::exclude_out_of_bounds): Enforce variable bounds
/// - [`verbose()`](OQNLP::verbose): Enable detailed progress output
///
#[cfg_attr(feature = "checkpointing", doc = "")]
#[cfg_attr(feature = "checkpointing", doc = "### State Persistence")]
#[cfg_attr(feature = "checkpointing", doc = "")]
#[cfg_attr(
    feature = "checkpointing",
    doc = "- [`with_checkpointing()`](Self::with_checkpointing): Enable automatic state saving"
)]
#[cfg_attr(
    feature = "checkpointing",
    doc = "- [`resume_with_modified_params()`](Self::resume_with_modified_params): Continue with new parameters"
)]
#[cfg_attr(feature = "checkpointing", doc = "")]
///
/// ## Example Usage
///
/// ```rust
/// use globalsearch::{oqnlp::OQNLP, types::OQNLPParams, problem::Problem, types::EvaluationError};
/// use ndarray::{array, Array1, Array2};
///
/// #[derive(Clone)]
/// struct QuadraticProblem;
///
/// impl Problem for QuadraticProblem {
///     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
///         Ok(x[0].powi(2) + x[1].powi(2))  // Minimize x² + y²
///     }
///
///     fn variable_bounds(&self) -> Array2<f64> {
///         array![[-5.0, 5.0], [-5.0, 5.0]]  // x, y ∈ [-5, 5]
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let problem = QuadraticProblem;
/// let params = OQNLPParams::default();
///
/// let mut optimizer = OQNLP::new(problem, params)?
///     .target_objective(0.01)  // Stop when f(x) ≤ 0.01
///     .max_time(60.0)          // Maximum 60 seconds
///     .verbose();              // Show progress
///
/// let solutions = optimizer.run()?;
/// println!("Found {} solutions", solutions.len());
/// # Ok(())
/// # }
/// ```
pub struct OQNLP<P: Problem + Clone> {
    /// Optimization problem definition.
    ///
    /// Defines the objective function, constraints, and variable bounds for the optimization.
    problem: P,

    /// Algorithm configuration parameters.
    ///
    /// Controls population size, iterations, convergence criteria, and local solver selection.
    params: OQNLPParams,
    /// Merit filter for solution quality control.
    ///
    /// Maintains dynamic threshold to accept only improving or competitive solutions.
    merit_filter: MeritFilter,

    /// Distance filter for population diversity.
    ///
    /// Enforces minimum separation between solutions to prevent clustering.
    distance_filter: DistanceFilter,

    /// Local optimization solver interface.
    ///
    /// Refines candidate solutions using classical optimization algorithms.
    local_solver: LocalSolver<P>,

    /// Collection of high-quality solutions discovered.
    ///
    /// Maintains the best solutions found during optimization. Initially `None`.
    solution_set: Option<SolutionSet>,

    /// Maximum optimization time limit (seconds).
    ///
    /// When set, optimization stops after this duration in Stage 2.
    /// Timer starts after the first local search completes.
    max_time: Option<f64>,

    /// Batch size for parallel processing in Stage 2 (only available with rayon feature).
    ///
    /// When the `rayon` feature is enabled and this is set to > 1, Stage 2 iterations
    /// are processed in batches of this size in parallel. When `None`, uses sequential
    /// processing or auto-determined batch size.
    #[cfg(feature = "rayon")]
    batch_iterations: Option<usize>,

    /// Verbose flag to enable additional output during the optimization process.
    ///
    /// When `true`, prints progress information to the console.
    ///
    /// Consider using an [`Observer`] for more detailed output
    verbose: bool,

    /// Enable parallel processing when Rayon is available.
    ///
    /// When `true` (default), uses parallel processing for applicable operations when
    /// the `rayon` feature is enabled. When `false`, forces sequential processing
    /// even if Rayon is available. Useful for benchmarking and Python bindings.
    #[cfg(feature = "rayon")]
    enable_parallel: bool,

    /// Early stopping criterion based on objective value.
    ///
    /// Optimization stops when a solution with objective ≤ this value is found.
    /// Useful when the global optimum or "good enough" threshold is known.
    target_objective: Option<f64>,

    /// Variable bounds enforcement flag.
    ///
    /// When `true`, solutions outside variable bounds are rejected.
    /// Combined with `target_objective`, both conditions must be satisfied.
    exclude_out_of_bounds: bool,

    /// Checkpoint manager for state persistence (optional).
    ///
    /// Handles saving and loading optimization state for long-running problems.
    #[cfg(feature = "checkpointing")]
    checkpoint_manager: Option<CheckpointManager>,

    /// Current iteration counter for checkpointing.
    #[cfg(feature = "checkpointing")]
    current_iteration: usize,

    /// Active reference set for scatter search state.
    #[cfg(feature = "checkpointing")]
    current_reference_set: Option<Vec<Array1<f64>>>,

    /// Counter for cycles without solution improvement.
    #[cfg(feature = "checkpointing")]
    unchanged_cycles: usize,

    /// Optimization start timestamp for elapsed time tracking.
    #[cfg(feature = "checkpointing")]
    start_time: Option<std::time::Instant>,

    /// Current random seed for reproducible continuation.
    #[cfg(feature = "checkpointing")]
    current_seed: u64,

    /// Observer for tracking algorithm state and metrics
    observer: Option<Observer>,

    /// Custom points to seed the reference set
    custom_points: Option<Vec<Array1<f64>>>,
}

impl<P: Problem + Clone + Send + Sync> OQNLP<P> {
    /// # OQNLP Constructor
    ///
    /// This method creates a new OQNLP instance with the given optimization problem and parameters.
    ///
    /// ## Errors
    ///
    /// Returns an error if the population size is less than 3 or if the iterations are greater than the population size.
    ///
    /// Returns an error if the distance filter fails to be created.
    ///
    /// ## Warnings
    ///
    /// If the `wait_cycle` parameter is greater than or equal to the `iterations` parameter, a warning is printed.
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, OQNLPError> {
        if params.population_size <= 3 {
            return Err(OQNLPError::InvalidPopulationSize(params.population_size));
        }

        if params.iterations > params.population_size {
            return Err(OQNLPError::InvalidIterations(params.iterations, params.population_size));
        }

        if params.threshold_factor <= 0.0 {
            return Err(OQNLPError::InvalidThresholdFactor(params.threshold_factor));
        }

        if params.wait_cycle >= params.iterations {
            eprintln!(
                "Warning: `wait_cycle` is greater than or equal to `iterations`. This may lead to suboptimal results."
            );
        }

        let filter_params: FilterParams = FilterParams {
            distance_factor: params.distance_factor,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
        };

        Ok(Self {
            problem: problem.clone(),
            params: params.clone(),
            merit_filter: MeritFilter::new(),
            distance_filter: DistanceFilter::new(filter_params)?,
            local_solver: LocalSolver::new(
                problem,
                params.local_solver_type.clone(),
                params.local_solver_config.clone(),
            ),
            solution_set: None,
            max_time: None,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            verbose: false,
            #[cfg(feature = "rayon")]
            enable_parallel: true, // Default to true for parallel processing
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "checkpointing")]
            checkpoint_manager: None,
            #[cfg(feature = "checkpointing")]
            current_iteration: 0,
            #[cfg(feature = "checkpointing")]
            current_reference_set: None,
            #[cfg(feature = "checkpointing")]
            unchanged_cycles: 0,
            #[cfg(feature = "checkpointing")]
            start_time: None,
            #[cfg(feature = "checkpointing")]
            current_seed: params.seed,
            observer: None,
            custom_points: None,
        })
    }

    /// Add an observer to track algorithm state
    ///
    /// The observer will track metrics during the optimization process,
    /// such as function evaluations, solution set size, and timing information.
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer to add
    ///
    /// ## Example Usage
    ///
    /// ```rust
    /// use globalsearch::observers::{Observer, ObserverMode};
    /// use globalsearch::oqnlp::OQNLP;
    /// use globalsearch::types::OQNLPParams;
    /// # use globalsearch::problem::Problem;
    /// # use globalsearch::types::EvaluationError;
    /// # use ndarray::{Array1, Array2, array};
    /// #
    /// # #[derive(Clone)]
    /// # struct TestProblem;
    /// # impl Problem for TestProblem {
    /// #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
    /// #         Ok(x[0].powi(2) + x[1].powi(2))
    /// #     }
    /// #     fn variable_bounds(&self) -> Array2<f64> {
    /// #         array![[-5.0, 5.0], [-5.0, 5.0]]
    /// #     }
    /// # }
    ///
    /// let problem = TestProblem;
    /// let params = OQNLPParams::default();
    ///
    /// // Create an observer with tracking for both stages
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_timing();
    ///
    /// let mut optimizer = OQNLP::new(problem, params)
    ///     .unwrap()
    ///     .add_observer(observer);
    ///
    /// let solutions = optimizer.run();
    ///
    /// // After optimization, access observer metrics
    /// // You can also use the observer's callback functionality to log metrics during optimization
    /// if let Some(observer) = optimizer.observer() {
    ///        if let Some(stage1) = observer.stage1_final() {
    ///            println!("\nStage 1 (Scatter Search):");
    ///            println!("  Reference set size: {}", stage1.reference_set_size());
    ///            println!("  Best objective: {:.8}", stage1.best_objective());
    ///            println!("  Function evaluations: {}", stage1.function_evaluations());
    ///            println!("  Trial points generated: {}", stage1.trial_points_generated());
    ///            if let Some(time) = stage1.total_time() {
    ///                println!("  Total time: {:.3}s", time);
    ///            }
    ///        }
    ///
    ///        if let Some(stage2) = observer.stage2() {
    ///            println!("\nStage 2 (Local Refinement):");
    ///            println!("  Iterations completed: {}", stage2.current_iteration() + 1);
    ///            println!("  Best objective: {:.8}", stage2.best_objective());
    ///            println!("  Solutions found: {}", stage2.solution_set_size());
    ///            println!(
    ///                "  Local solver calls: {} (improved: {})",
    ///                stage2.local_solver_calls(),
    ///                stage2.improved_local_calls()
    ///            );
    ///            println!("  Function evaluations: {}", stage2.function_evaluations());
    ///            println!("  Unchanged cycles: {}", stage2.unchanged_cycles());
    ///            if let Some(time) = stage2.total_time() {
    ///                println!("  Total time: {:.3}s", time);
    ///          }
    ///        }
    ///    }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_observer(mut self, observer: Observer) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Get a reference to the observer
    pub fn observer(&self) -> Option<&Observer> {
        self.observer.as_ref()
    }

    /// Set custom points to seed the reference set
    ///
    /// This method allows you to provide initial points that will be included in the
    /// reference set during Stage 1 initialization. These points are added after the
    /// three default seed points (lower bound, upper bound, and midpoint) and before
    /// the diversification step.
    ///
    /// # Arguments
    ///
    /// * `points` - A 2D array where each row is a point (`Array1<f64>`) to add to the reference set.
    ///   The number of columns must match the problem dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any point has a different dimension than the problem
    /// - Any point is outside the variable bounds defined by the problem
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::oqnlp::OQNLP;
    /// use globalsearch::types::OQNLPParams;
    /// # use globalsearch::problem::Problem;
    /// # use globalsearch::types::EvaluationError;
    /// # use ndarray::{Array1, Array2, array};
    /// # #[derive(Clone)]
    /// # struct TestProblem;
    /// # impl Problem for TestProblem {
    /// #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
    /// #         Ok(x[0].powi(2) + x[1].powi(2))
    /// #     }
    /// #     fn variable_bounds(&self) -> Array2<f64> {
    /// #         array![[-5.0, 5.0], [-5.0, 5.0]]
    /// #     }
    /// # }
    ///
    /// let problem = TestProblem;
    /// let params = OQNLPParams::default();
    ///
    /// // Define custom starting points (2D array)
    /// let custom_points = array![
    ///     [1.0, 1.0],   // First point
    ///     [-1.0, -1.0], // Second point
    ///     [0.0, 0.0],   // Third point
    /// ];
    ///
    /// let mut optimizer = OQNLP::new(problem, params)
    ///     .unwrap()
    ///     .with_points(custom_points)?;
    ///
    /// let solutions = optimizer.run();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_points(mut self, points: Array2<f64>) -> Result<Self, OQNLPError> {
        let bounds = self.problem.variable_bounds();
        let n_dims = bounds.nrows();

        // Convert Array2 to Vec<Array1<f64>>
        let mut custom_points = Vec::new();
        for (i, point_row) in points.outer_iter().enumerate() {
            let point = point_row.to_owned();

            // Check dimension
            if point.len() != n_dims {
                return Err(OQNLPError::InvalidCustomPointsDimension {
                    expected: n_dims,
                    got: point.len(),
                });
            }

            // Check bounds
            for (j, &value) in point.iter().enumerate() {
                let lower = bounds[[j, 0]];
                let upper = bounds[[j, 1]];
                if value < lower || value > upper {
                    return Err(OQNLPError::CustomPointOutOfBounds { index: i });
                }
            }

            custom_points.push(point);
        }

        self.custom_points = Some(custom_points);
        Ok(self)
    }

    /// Run the OQNLP algorithm and return the solution set
    pub fn run(&mut self) -> Result<SolutionSet, OQNLPError> {
        // Start observer timing if enabled
        if let Some(ref mut observer) = self.observer {
            observer.start_timer();
            if observer.should_observe_stage1() {
                if let Some(stage1) = observer.stage1_mut() {
                    stage1.start();
                }
            }
        }

        // Try to resume from checkpoint if enabled
        #[cfg(feature = "checkpointing")]
        let resumed_from_checkpoint = self.try_resume_from_checkpoint()?;
        #[cfg(not(feature = "checkpointing"))]
        let resumed_from_checkpoint = false;

        // Set start time for elapsed time calculation
        #[cfg(feature = "checkpointing")]
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        // If we resumed from checkpoint, skip Stage 1 and jump to Stage 2
        let (mut ref_set, mut unchanged_cycles, ref_objectives) = if resumed_from_checkpoint {
            #[cfg(feature = "checkpointing")]
            {
                if self.verbose {
                    println!("Resuming from checkpoint at iteration {}", self.current_iteration);
                }
                let ref_set = self.current_reference_set.clone().unwrap_or_default();
                let unchanged_cycles = self.unchanged_cycles;
                // No pre-computed objectives when resuming from checkpoint
                (ref_set, unchanged_cycles, None)
            }
            #[cfg(not(feature = "checkpointing"))]
            unreachable!()
        } else {
            // Stage 1: Initial ScatterSearch iterations and first local call
            if self.verbose {
                println!("Starting Stage 1");
            }

            #[cfg(feature = "rayon")]
            let mut ss = ScatterSearch::new(self.problem.clone(), self.params.clone())?
                .parallel(self.enable_parallel);
            #[cfg(not(feature = "rayon"))]
            let mut ss = ScatterSearch::new(self.problem.clone(), self.params.clone())?;

            // Add custom points if provided
            if let Some(ref custom_points) = self.custom_points {
                ss = ss.with_custom_points(custom_points.clone());
            }

            // Attach observer if present and should observe Stage 1
            let (ref_set_with_objectives, scatter_candidate) = if let Some(ref mut observer) =
                self.observer
            {
                if observer.should_observe_stage1() {
                    ss.with_observer(observer).run().map_err(OQNLPError::ScatterSearchRunError)?
                } else {
                    ss.run().map_err(OQNLPError::ScatterSearchRunError)?
                }
            } else {
                ss.run().map_err(OQNLPError::ScatterSearchRunError)?
            };

            // Destructure the reference set to separate points and their pre-computed objectives
            let (ref_set, ref_objectives): (Vec<Array1<f64>>, Vec<f64>) =
                ref_set_with_objectives.into_iter().unzip();

            // Observer: Report scatter search complete
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage1() {
                    if let Some(stage1) = observer.stage1_mut() {
                        stage1.enter_substage("scatter_search_complete");
                        stage1.set_reference_set_size(ref_set.len());
                        // Find the best solution and its coordinates
                        if let Some((best_idx, &best_obj)) = ref_objectives
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        {
                            stage1.set_best_solution(best_obj, &ref_set[best_idx]);
                        }
                    }
                }
                observer.invoke_callback();
            }

            // Observer: Enter local optimization substage
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage1() {
                    if let Some(stage1) = observer.stage1_mut() {
                        stage1.enter_substage("local_optimization");
                    }
                }
            }

            let (local_sol, local_fn_evals) =
                if self.observer.as_ref().map(|obs| obs.should_observe_stage1()).unwrap_or(false) {
                    self.local_solver.solve_with_tracking(scatter_candidate, true)?
                } else {
                    let sol = self.local_solver.solve(scatter_candidate)?;
                    (sol, 0)
                };

            self.merit_filter.update_threshold(local_sol.objective);

            if self.verbose {
                println!(
                    "Stage 1: Local solution found with objective = {:.8}",
                    local_sol.objective
                );
            }

            // Observer: Track local optimization result
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage1() {
                    if let Some(stage1) = observer.stage1_mut() {
                        stage1.enter_substage("local_optimization_complete");
                        stage1.set_best_solution(local_sol.objective, &local_sol.point);
                        stage1.add_function_evaluations(local_fn_evals as usize);
                    }
                }
                // Invoke callback after local optimization
                observer.invoke_callback();
            }

            self.process_local_solution(local_sol)?;

            // Check if target objective has been reached after Stage 1
            if self.target_objective_reached() {
                if self.verbose {
                    println!(
                        "Stage 1: Target objective {:.8} reached. Stopping optimization.",
                        self.target_objective.unwrap()
                    );
                }

                // Observer: End Stage 1
                if let Some(ref mut observer) = self.observer {
                    if observer.should_observe_stage1() {
                        if let Some(stage1) = observer.stage1_mut() {
                            stage1.enter_substage("stage1_complete");
                            stage1.end();
                        }
                    }
                    // Invoke callback after Stage 1 completion (final Stage 1 log)
                    observer.invoke_callback();
                    // Mark Stage 1 as completed to prevent further logging
                    observer.mark_stage1_complete();
                }

                return self.solution_set.clone().ok_or(OQNLPError::NoFeasibleSolution);
            }

            // Store reference set for checkpointing
            #[cfg(feature = "checkpointing")]
            {
                self.current_reference_set = Some(ref_set.clone());
            }

            // Observer: End Stage 1
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage1() {
                    if let Some(stage1) = observer.stage1_mut() {
                        stage1.enter_substage("stage1_complete");
                        stage1.end();
                    }
                }
                // Invoke callback at the end of Stage 1 (final Stage 1 log)
                observer.invoke_callback();
                // Mark Stage 1 as completed to prevent further logging
                observer.mark_stage1_complete();
            }

            (ref_set, 0, Some(ref_objectives))
        };

        if self.verbose {
            println!("Starting Stage 2");
        }

        // Observer: Start Stage 2
        if let Some(ref mut observer) = self.observer {
            if observer.should_observe_stage2() {
                if let Some(stage2) = observer.stage2_mut() {
                    stage2.start();
                    if let Some(ref sol_set) = self.solution_set {
                        if let Some(best) = sol_set.best_solution() {
                            stage2.set_best_solution(best.objective, &best.point);
                        }
                        stage2.set_solution_set_size(sol_set.len());
                    }
                    stage2.set_threshold_value(self.merit_filter.threshold);
                }
            }
            // Mark Stage 2 as started to enable logging
            observer.mark_stage2_started();
        }

        #[cfg(feature = "progress_bar")]
        let mut stage2_bar = Bar::builder()
            .total(self.params.iterations)
            .desc("Stage 2")
            .unit("it")
            .postfix(format!(
                "Objective function: {:.6}",
                self.solution_set
                    .as_ref()
                    .and_then(|s| s.best_solution())
                    .map_or(f64::INFINITY, |s| s.objective)
            ))
            .build()
            .expect("Failed to create progress bar");

        // Adjust progress bar for resumed runs
        #[cfg(all(feature = "progress_bar", feature = "checkpointing"))]
        if resumed_from_checkpoint {
            for _ in 0..self.current_iteration {
                stage2_bar.update(1).expect("Failed to update progress bar");
            }
        }

        // Stage 2: Main iterative loop
        #[cfg(feature = "checkpointing")]
        let mut rng: StdRng = if resumed_from_checkpoint {
            // Use the saved seed when resuming from checkpoint
            StdRng::seed_from_u64(self.current_seed)
        } else {
            // Start fresh with base seed + current iteration
            let seed = self.params.seed + self.current_iteration as u64;
            self.current_seed = seed;
            StdRng::seed_from_u64(seed)
        };

        #[cfg(not(feature = "checkpointing"))]
        let mut rng: StdRng = StdRng::seed_from_u64(self.params.seed);

        // Shuffle reference set if starting fresh
        if !resumed_from_checkpoint {
            ref_set.shuffle(&mut rng);
        }

        let start_timer: Option<std::time::Instant> =
            self.max_time.map(|_| std::time::Instant::now());

        // Start from current iteration if resumed, otherwise from 0
        #[cfg(feature = "checkpointing")]
        let start_iter = if resumed_from_checkpoint { self.current_iteration } else { 0 };
        #[cfg(not(feature = "checkpointing"))]
        let start_iter = 0;

        // Calculate effective batch size for parallel processing
        #[cfg(feature = "rayon")]
        let effective_batch_size = if !self.enable_parallel {
            1 // Force sequential when parallel processing is disabled
        } else {
            self.batch_iterations.unwrap_or_else(|| {
                // Use a more intelligent default: balance between overhead and parallelism
                let thread_count = rayon::current_num_threads();
                let remaining_iterations = self.params.iterations.saturating_sub(start_iter);

                if remaining_iterations < 4 || thread_count == 1 {
                    1 // Use sequential for small workloads or single-threaded
                } else {
                    // Use a batch size that gives each thread meaningful work
                    (remaining_iterations / (thread_count * 2)).clamp(2, 8)
                }
            })
        };

        #[cfg(not(feature = "rayon"))]
        let effective_batch_size = 1; // Always sequential when rayon is not available

        // Process Stage 2 iterations in batches
        let trials_to_process: Vec<_> =
            ref_set.iter().take(self.params.iterations).enumerate().skip(start_iter).collect();

        for batch in trials_to_process.chunks(effective_batch_size) {
            // Check timeout before processing each batch
            if let (Some(max_secs), Some(start)) = (self.max_time, start_timer) {
                if start.elapsed().as_secs_f64() > max_secs {
                    if self.verbose {
                        println!("Timeout reached after {} seconds", max_secs);
                    }
                    break;
                }
            }

            // Process batch in parallel if rayon is enabled and batch size > 1
            #[cfg(feature = "rayon")]
            if effective_batch_size > 1 {
                self.process_batch_parallel(
                    batch,
                    &mut unchanged_cycles,
                    start_iter,
                    resumed_from_checkpoint,
                    ref_objectives.as_ref(),
                )?;
            } else {
                self.process_batch_sequential(
                    batch,
                    &mut unchanged_cycles,
                    start_iter,
                    resumed_from_checkpoint,
                    ref_objectives.as_ref(),
                )?;
            }

            #[cfg(not(feature = "rayon"))]
            {
                self.process_batch_sequential(
                    batch,
                    &mut unchanged_cycles,
                    start_iter,
                    resumed_from_checkpoint,
                    ref_objectives.as_ref(),
                )?;
            }

            // Check if target objective has been reached after processing the batch
            if self.target_objective_reached() {
                if self.verbose {
                    println!(
                        "Stage 2: Target objective {:.8} reached. Stopping optimization.",
                        self.target_objective.unwrap()
                    );
                }
                break;
            }
        }

        // Observer: End Stage 2
        if let Some(ref mut observer) = self.observer {
            if observer.should_observe_stage2() {
                if let Some(stage2) = observer.stage2_mut() {
                    stage2.end();
                }
            }
        }

        // Save final checkpoint at the end of optimization
        #[cfg(feature = "checkpointing")]
        self.maybe_save_final_checkpoint()?;

        self.solution_set.clone().ok_or(OQNLPError::NoFeasibleSolution)
    }

    // Helper methods
    /// Check if a local search should be started based on the merit and distance filters
    fn should_start_local(&self, point: &Array1<f64>, obj: f64) -> Result<bool, OQNLPError> {
        let passes_merit: bool = obj <= self.merit_filter.threshold;
        let passes_distance: bool = self.distance_filter.check(point);
        Ok(passes_merit && passes_distance)
    }

    /// Check if the target objective has been reached
    fn target_objective_reached(&self) -> bool {
        if let (Some(target), Some(solution_set)) = (self.target_objective, &self.solution_set) {
            if let Some(best) = solution_set.best_solution() {
                let target_reached = best.objective <= target;

                // If exclude_out_of_bounds is enabled, also check if the solution is within bounds
                if self.exclude_out_of_bounds {
                    return target_reached && self.is_within_bounds(&best.point);
                }

                return target_reached;
            }
        }
        false
    }

    /// Check if a point is within the variable bounds
    fn is_within_bounds(&self, point: &Array1<f64>) -> bool {
        let bounds = self.problem.variable_bounds();

        for (i, &value) in point.iter().enumerate() {
            let lower_bound = bounds[[i, 0]];
            let upper_bound = bounds[[i, 1]];

            if value < lower_bound || value > upper_bound {
                return false;
            }
        }

        true
    }

    /// Process a local solution, updating the best solution and filters
    fn process_local_solution(&mut self, solution: LocalSolution) -> Result<bool, OQNLPError> {
        const ABS_TOL: f64 = 1e-8;
        const REL_TOL: f64 = 1e-6;

        // If exclude_out_of_bounds is enabled, check if the solution is within bounds
        if self.exclude_out_of_bounds && !self.is_within_bounds(&solution.point) {
            if self.verbose {
                println!(
                    "Solution with objective {:.8} rejected: out of bounds",
                    solution.objective
                );
            }
            // Still add to distance filter to maintain diversity, but don't add to solution set
            self.distance_filter.add_solution(solution);
            return Ok(false);
        }

        let solutions = if let Some(existing) = &self.solution_set {
            existing.solutions.clone()
        } else {
            // First solution, initialize solution set
            if self.verbose {
                println!(
                    "New solution added to solution set: objective = {:.8}, point = {}",
                    solution.objective, solution.point
                );
                println!("Solution set size: 1");
            }

            self.solution_set =
                Some(SolutionSet { solutions: Array1::from(vec![solution.clone()]) });
            self.merit_filter.update_threshold(solution.objective);
            self.distance_filter.add_solution(solution);

            return Ok(true);
        };

        let current_best: &LocalSolution = &solutions[0];
        let obj1 = solution.objective;
        let obj2 = current_best.objective;

        let obj_diff = (obj1 - obj2).abs();
        let tol = ABS_TOL.max(REL_TOL * obj1.abs().max(obj2.abs()));

        let added: bool = if obj1 < obj2 - tol {
            // Found new best solution
            self.solution_set =
                Some(SolutionSet { solutions: Array1::from(vec![solution.clone()]) });
            self.merit_filter.update_threshold(solution.objective);

            // Update observer with the newly added solution
            if let Some(ref mut observer) = self.observer {
                if let Some(stage2) = observer.stage2_mut() {
                    stage2.set_last_added_solution(&solution.point);
                }
            }

            if self.verbose {
                println!(
                    "New best solution found (replacing solution set): objective = {:.8}, point = {}",
                    solution.objective, solution.point
                );
                println!("Solution set size: 1");
            }

            false
        } else if obj_diff <= tol && !self.is_duplicate_in_set(&solution, &solutions) {
            // Similar objective value and not duplicate, add to set
            let mut new_solutions: Vec<LocalSolution> = solutions.to_vec();
            new_solutions.push(solution.clone());
            self.solution_set = Some(SolutionSet { solutions: Array1::from(new_solutions) });

            // Update observer with the newly added solution
            if let Some(ref mut observer) = self.observer {
                if let Some(stage2) = observer.stage2_mut() {
                    stage2.set_last_added_solution(&solution.point);
                }
            }

            if self.verbose {
                println!(
                    "New solution added to solution set: objective = {:.8}, point = {}",
                    solution.objective, solution.point
                );
                println!("Solution set size: {}", self.solution_set.as_ref().unwrap().len());
            }

            true
        } else {
            false
        };

        self.distance_filter.add_solution(solution);
        Ok(added)
    }

    /// Check if a candidate solution is a duplicate in a set of solutions
    fn is_duplicate_in_set(&self, candidate: &LocalSolution, set: &Array1<LocalSolution>) -> bool {
        let distance_threshold = self.params.distance_factor;

        set.iter().any(|s| {
            let diff = &candidate.point - &s.point;
            let dist_squared = diff.dot(&diff);
            dist_squared < distance_threshold * distance_threshold
        })
    }

    /// Set the maximum time for the stage 2 of the OQNLP algorithm and return self for chaining.
    ///
    /// If the time limit is reached, the algorithm will stop and return the
    /// solution set found so far.
    ///
    /// The time limit is in seconds and it starts timing after the
    /// first local search.
    pub fn max_time(mut self, max_time: f64) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Set the batch size for parallel processing in Stage 2 and return self for chaining.
    ///
    /// This method is only available when the `rayon` feature is enabled.
    /// When enabled, Stage 2 iterations will be processed in batches of this size in parallel.
    /// This allows for efficient parallelization while maintaining convergence properties.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of iterations to process in each parallel batch
    ///   - If set to 1: Sequential processing (no parallelization)
    ///   - If set to larger values: Parallel processing with the specified batch size
    ///   - When not set (None): Uses sequential processing
    #[cfg(feature = "rayon")]
    pub fn batch_iterations(mut self, batch_size: usize) -> Self {
        // Warn if batch_size is larger than total iterations
        if batch_size > self.params.iterations {
            eprintln!(
                "Warning: batch_iterations ({}) is larger than total iterations ({}). \
                This may lead to suboptimal resource usage. Consider setting batch_iterations <= iterations.",
                batch_size, self.params.iterations
            );
        }

        self.batch_iterations = Some(batch_size);
        self
    }

    /// Enable verbose output for the OQNLP algorithm
    ///
    /// This will print additional information about the optimization process
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Control parallel processing when Rayon is available
    ///
    /// When the `rayon` feature is enabled, this controls whether parallel processing
    /// is used for applicable operations. Set to `false` to force sequential processing
    /// even when Rayon is available. Useful for benchmarking and Python bindings.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable parallel processing (default: `true`)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use globalsearch::*;
    /// # use globalsearch::types::*;
    /// # use globalsearch::problem::*;
    /// # use globalsearch::oqnlp::*;
    /// # #[derive(Clone)]
    /// # struct TestProblem;
    /// # impl Problem for TestProblem {
    /// #     fn objective(&self, x: &ndarray::Array1<f64>) -> Result<f64, EvaluationError> { Ok(0.0) }
    /// #     fn variable_bounds(&self) -> ndarray::Array2<f64> { ndarray::array![[-1.0, 1.0]] }
    /// # }
    /// let problem = TestProblem;
    /// let params = OQNLPParams::default();
    ///
    /// // Disable parallel processing for benchmarking
    /// let oqnlp = OQNLP::new(problem, params)
    ///     .unwrap()
    ///     .parallel(false);
    /// ```
    #[cfg(feature = "rayon")]
    pub fn parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Set a target objective function value to stop optimization early
    ///
    /// If the best solution found has an objective function value less than or equal to this target,
    /// the optimization will stop early. This can be useful when you know the global optimum
    /// or want to stop when a "good enough" solution is found.
    ///
    /// # Arguments
    ///
    /// * `target` - The target objective function value
    ///
    /// # Example
    ///
    /// ```rust
    /// # use globalsearch::oqnlp::OQNLP;
    /// # use globalsearch::types::OQNLPParams;
    /// # use globalsearch::problem::Problem;
    /// # use globalsearch::types::EvaluationError;
    /// # use ndarray::{Array1, Array2, array};
    /// # #[derive(Clone)]
    /// # struct TestProblem;
    /// # impl Problem for TestProblem {
    /// #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> { Ok(x.sum().powi(2)) }
    /// #     fn variable_bounds(&self) -> Array2<f64> { array![[-1.0, 1.0]] }
    /// # }
    /// let mut oqnlp = OQNLP::new(TestProblem, OQNLPParams::default())
    ///     .unwrap()
    ///     .target_objective(0.0);  // Stop when objective <= 0.0
    /// ```
    pub fn target_objective(mut self, target: f64) -> Self {
        self.target_objective = Some(target);
        self
    }

    /// Enable or disable exclusion of out-of-bounds solutions
    ///
    /// When enabled, the algorithm will check if solutions are within the variable bounds
    /// before adding them to the solution set.
    pub fn set_exclude_out_of_bounds(mut self, enable: bool) -> Self {
        self.exclude_out_of_bounds = enable;
        self
    }

    /// Enable exclusion of out-of-bounds solutions
    ///
    /// This is a convenience method equivalent to calling `set_exclude_out_of_bounds(true)`.
    /// When enabled, the algorithm will check if solutions are within the variable bounds
    /// before adding them to the solution set.
    pub fn exclude_out_of_bounds(self) -> Self {
        self.set_exclude_out_of_bounds(true)
    }

    /// Enable checkpointing with the given configuration
    ///
    /// This allows saving and resuming the optimization state
    #[cfg(feature = "checkpointing")]
    pub fn with_checkpointing(mut self, config: CheckpointConfig) -> Result<Self, OQNLPError> {
        self.checkpoint_manager = Some(CheckpointManager::new(config)?);
        Ok(self)
    }

    /// Try to resume from the latest checkpoint if available
    ///
    /// Returns true if resumed from checkpoint, false if starting fresh
    #[cfg(feature = "checkpointing")]
    pub fn try_resume_from_checkpoint(&mut self) -> Result<bool, OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if manager.config().auto_resume && manager.checkpoint_exists() {
                let checkpoint = manager.load_latest_checkpoint()?;
                self.restore_from_checkpoint(checkpoint)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Resume from the latest checkpoint with modified parameters
    ///
    /// This allows you to continue optimization with different parameters
    /// (e.g., more iterations, different population size, etc.)
    #[cfg(feature = "checkpointing")]
    pub fn resume_with_modified_params(
        &mut self,
        new_params: OQNLPParams,
    ) -> Result<bool, OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if manager.checkpoint_exists() {
                let mut checkpoint = manager.load_latest_checkpoint()?;

                // Keep the old params for comparison
                let old_iterations = checkpoint.params.iterations;
                let old_population_size = checkpoint.params.population_size;

                // Handle population size changes
                if new_params.population_size != old_population_size {
                    if new_params.population_size > old_population_size {
                        self.expand_reference_set(
                            &mut checkpoint.reference_set,
                            old_population_size,
                            new_params.population_size,
                        )?;

                        if self.verbose {
                            println!(
                                "Expanded reference set from {} to {} points",
                                old_population_size, new_params.population_size
                            );
                        }
                    } else {
                        // Population size decreased, issue warning but continue
                        eprintln!(
                            "Warning: New population size ({}) is smaller than original ({}). Using original reference set with {} points.",
                            new_params.population_size, old_population_size, old_population_size
                        );

                        if self.verbose {
                            println!(
                                "Keeping original reference set size of {} points despite smaller population_size parameter",
                                old_population_size
                            );
                        }
                    }
                }

                // Update with new parameters
                checkpoint.params = new_params.clone();

                self.restore_from_checkpoint(checkpoint)?;

                if self.verbose {
                    println!(
                        "Resumed with modified parameters. Iterations changed from {} to {}",
                        old_iterations, new_params.iterations
                    );
                }

                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Expand the reference set to a larger population size
    ///
    /// This method generates additional points to expand an existing reference set
    /// while maintaining diversity using the same strategy as ScatterSearch
    #[cfg(feature = "checkpointing")]
    fn expand_reference_set(
        &self,
        ref_set: &mut Vec<Array1<f64>>,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), OQNLPError> {
        if new_size <= old_size {
            return Ok(());
        }

        // Create a temporary ScatterSearch instance to reuse its methods
        let temp_params = OQNLPParams {
            population_size: new_size,
            seed: self.current_seed + ref_set.len() as u64,
            ..self.params.clone()
        };

        #[cfg(feature = "rayon")]
        let mut scatter_search =
            ScatterSearch::new(self.problem.clone(), temp_params)?.parallel(self.enable_parallel);
        #[cfg(not(feature = "rayon"))]
        let mut scatter_search = ScatterSearch::new(self.problem.clone(), temp_params)?;

        // ScatterSearch's diversify_reference_set method to expand our existing set
        scatter_search.diversify_reference_set(ref_set)?;

        Ok(())
    }

    /// Resume from a specific checkpoint file with modified parameters
    #[cfg(feature = "checkpointing")]
    pub fn resume_from_checkpoint_with_params(
        &mut self,
        checkpoint_path: &std::path::Path,
        new_params: OQNLPParams,
    ) -> Result<(), OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut checkpoint = manager.load_checkpoint_from_path(checkpoint_path)?;

            // Keep the old params for comparison
            let old_iterations = checkpoint.params.iterations;

            // Update with new parameters
            checkpoint.params = new_params.clone();

            self.restore_from_checkpoint(checkpoint)?;

            if self.verbose {
                println!(
                    "Resumed from {} with modified parameters. Iterations changed from {} to {}",
                    checkpoint_path.display(),
                    old_iterations,
                    new_params.iterations
                );
            }
        }
        Ok(())
    }

    /// Restore state from a checkpoint
    #[cfg(feature = "checkpointing")]
    fn restore_from_checkpoint(&mut self, checkpoint: OQNLPCheckpoint) -> Result<(), OQNLPError> {
        let solution_count = checkpoint.solution_set.as_ref().map_or(0, |s| s.len());

        self.params = checkpoint.params;
        self.current_iteration = checkpoint.current_iteration;
        self.merit_filter.update_threshold(checkpoint.merit_threshold);
        self.solution_set = checkpoint.solution_set;
        self.current_reference_set = Some(checkpoint.reference_set);
        self.unchanged_cycles = checkpoint.unchanged_cycles;

        // Restore the current seed for continuing RNG sequence
        self.current_seed = checkpoint.current_seed;

        // Restore target objective
        self.target_objective = checkpoint.target_objective;

        // Restore exclude_out_of_bounds setting
        self.exclude_out_of_bounds = checkpoint.exclude_out_of_bounds;

        // Restore batch_iterations setting
        #[cfg(feature = "rayon")]
        {
            self.batch_iterations = checkpoint.batch_iterations;
        }

        // Restore parallel execution setting
        #[cfg(feature = "rayon")]
        {
            self.enable_parallel = checkpoint.enable_parallel;
        }

        // Restore distance filter solutions
        self.distance_filter.set_solutions(checkpoint.distance_filter_solutions);

        if self.verbose {
            println!(
                "Resumed from checkpoint at iteration {} with {} solutions",
                checkpoint.current_iteration, solution_count
            );
        }

        Ok(())
    }

    /// Create a checkpoint of the current state
    #[cfg(feature = "checkpointing")]
    fn create_checkpoint(&self) -> OQNLPCheckpoint {
        let elapsed_time =
            self.start_time.map(|start| start.elapsed().as_secs_f64()).unwrap_or(0.0);

        OQNLPCheckpoint {
            params: self.params.clone(),
            current_iteration: self.current_iteration,
            merit_threshold: self.merit_filter.threshold,
            solution_set: self.solution_set.clone(),
            reference_set: self.current_reference_set.clone().unwrap_or_default(),
            unchanged_cycles: self.unchanged_cycles,
            elapsed_time,
            distance_filter_solutions: self.distance_filter.get_solutions().clone(),
            current_seed: self.current_seed,
            target_objective: self.target_objective,
            exclude_out_of_bounds: self.exclude_out_of_bounds,
            #[cfg(feature = "rayon")]
            batch_iterations: self.batch_iterations,
            #[cfg(feature = "rayon")]
            enable_parallel: self.enable_parallel,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Save a checkpoint if checkpointing is enabled and conditions are met
    #[cfg(feature = "checkpointing")]
    fn maybe_save_checkpoint(&self) -> Result<(), OQNLPError> {
        if let Some(ref manager) = self.checkpoint_manager {
            if self.current_iteration % manager.config().save_frequency == 0 {
                let checkpoint = self.create_checkpoint();
                let saved_path = manager.save_checkpoint(&checkpoint, self.current_iteration)?;

                if self.verbose {
                    println!("Checkpoint saved to: {}", saved_path.display());
                }
            }
        }
        Ok(())
    }

    /// Save final checkpoint at the end of optimization (regardless of save_frequency)
    #[cfg(feature = "checkpointing")]
    fn maybe_save_final_checkpoint(&self) -> Result<(), OQNLPError> {
        if let Some(manager) = &self.checkpoint_manager {
            let checkpoint = self.create_checkpoint();
            let saved_path = manager.save_checkpoint(&checkpoint, self.current_iteration)?;

            if self.verbose {
                println!("Final checkpoint saved to: {}", saved_path.display());
            }
        }
        Ok(())
    }

    /// Adjust the threshold for the merit filter
    ///
    /// The threshold is adjusted using the equation:
    /// `threshold = threshold + threshold_factor * (1 + abs(threshold))`
    fn adjust_threshold(&mut self, current_threshold: f64) {
        let new_threshold: f64 =
            current_threshold + self.params.threshold_factor * (1.0 + current_threshold.abs());
        self.merit_filter.update_threshold(new_threshold);
    }

    /// Process a batch of trials sequentially
    fn process_batch_sequential(
        &mut self,
        batch: &[(usize, &Array1<f64>)],
        unchanged_cycles: &mut usize,
        start_iter: usize,
        resumed_from_checkpoint: bool,
        ref_objectives: Option<&Vec<f64>>,
    ) -> Result<(), OQNLPError> {
        for &(local_iter, trial) in batch {
            // Skip iterations that were already processed when resuming from checkpoint
            if resumed_from_checkpoint && local_iter < start_iter {
                continue;
            }

            #[cfg(feature = "checkpointing")]
            {
                self.current_iteration = local_iter;
                self.unchanged_cycles = *unchanged_cycles;
            }

            // Update the current seed for checkpointing
            #[cfg(feature = "checkpointing")]
            {
                self.current_seed = self.params.seed + self.current_iteration as u64;
            }

            let trial = trial.clone();

            // Observer: Update iteration
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage2() {
                    if let Some(stage2) = observer.stage2_mut() {
                        stage2.set_iteration(local_iter);
                        stage2.set_unchanged_cycles(*unchanged_cycles);
                        stage2.set_threshold_value(self.merit_filter.threshold);
                    }
                }
            }

            // Use pre-computed objective if available, otherwise evaluate
            let obj: f64 = if let Some(objectives) = ref_objectives {
                // Map iteration index to reference set index (cycle through ref_set)
                let ref_set_index = local_iter % objectives.len();

                objectives[ref_set_index]
            } else {
                // Fallback to evaluating objective function (e.g., when resuming from checkpoint)
                self.problem
                    .objective(&trial)
                    .map_err(|_| OQNLPError::ObjectiveFunctionEvaluationFailed)?
            };

            // Observer: Track function evaluation only if we actually evaluated (not cached)
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage2() && ref_objectives.is_none() {
                    if let Some(stage2) = observer.stage2_mut() {
                        stage2.add_function_evaluations(1);
                    }
                }
            }

            if self.should_start_local(&trial, obj)? {
                self.merit_filter.update_threshold(obj);

                // Only track evaluations if observer is active
                let (local_trial, eval_count) = if let Some(ref mut observer) = self.observer {
                    if observer.should_observe_stage2() {
                        self.local_solver.solve_with_tracking(trial, true)?
                    } else {
                        let sol = self.local_solver.solve(trial)?;
                        (sol, 0)
                    }
                } else {
                    let sol = self.local_solver.solve(trial)?;
                    (sol, 0)
                };

                let added: bool = self.process_local_solution(local_trial.clone())?;

                // Observer: Track local solver call and results
                if let Some(ref mut observer) = self.observer {
                    if observer.should_observe_stage2() {
                        if let Some(stage2) = observer.stage2_mut() {
                            stage2.add_local_solver_call(added);
                            stage2.add_function_evaluations(eval_count as usize);
                            if let Some(ref sol_set) = self.solution_set {
                                if let Some(best) = sol_set.best_solution() {
                                    stage2.set_best_solution(best.objective, &best.point);
                                }
                                stage2.set_solution_set_size(sol_set.len());
                            }
                        }
                    }
                }

                if self.verbose && added {
                    println!(
                        "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                        local_iter, local_trial.objective
                    );
                    println!("x0 = {}", local_trial.point);
                }

                // Check if target objective has been reached
                if self.target_objective_reached() {
                    if self.verbose {
                        println!(
                            "Stage 2, iteration {}: Target objective {:.8} reached. Stopping optimization.",
                            local_iter,
                            self.target_objective.unwrap()
                        );
                    }
                    return Ok(());
                }
            } else {
                *unchanged_cycles += 1;

                if *unchanged_cycles >= self.params.wait_cycle {
                    if self.verbose {
                        println!(
                            "Stage 2, iteration {}: Adjusting threshold from {:.8} to {:.8}",
                            local_iter,
                            self.merit_filter.threshold,
                            self.merit_filter.threshold + 0.1 * self.merit_filter.threshold.abs()
                        );
                    }

                    self.adjust_threshold(self.merit_filter.threshold);
                    *unchanged_cycles = 0;
                }
            }

            // Invoke observer callback if set and frequency matches
            if let Some(ref mut observer) = self.observer {
                if observer.should_invoke_callback(local_iter) {
                    observer.invoke_callback();
                }
            }

            // Save checkpoint if enabled and conditions are met
            #[cfg(feature = "checkpointing")]
            self.maybe_save_checkpoint()?;
        }
        Ok(())
    }

    /// Process a batch of trials in parallel using Rayon
    #[cfg(feature = "rayon")]
    fn process_batch_parallel(
        &mut self,
        batch: &[(usize, &Array1<f64>)],
        unchanged_cycles: &mut usize,
        start_iter: usize,
        resumed_from_checkpoint: bool,
        ref_objectives: Option<&Vec<f64>>,
    ) -> Result<(), OQNLPError> {
        // Filter out iterations that were already processed when resuming from checkpoint
        let filtered_batch: Vec<_> = if resumed_from_checkpoint {
            batch.iter().filter(|&&(local_iter, _)| local_iter >= start_iter).copied().collect()
        } else {
            batch.to_vec()
        };

        // Skip if no iterations need processing
        if filtered_batch.is_empty() {
            return Ok(());
        }

        // For parallel processing, we need to collect the objective evaluations first
        let batch_results: Result<Vec<(usize, Array1<f64>, f64)>, OQNLPError> = filtered_batch
            .par_iter()
            .map(|&(local_iter, trial)| {
                // Use pre-computed objective if available, otherwise evaluate
                let obj = if let Some(objectives) = ref_objectives {
                    // Map iteration index to reference set index (cycle through ref_set)
                    let ref_set_index = local_iter % objectives.len();
                    objectives[ref_set_index]
                } else {
                    // Fallback to evaluating objective function (e.g., when resuming from checkpoint)
                    self.problem
                        .objective(trial)
                        .map_err(|_| OQNLPError::ObjectiveFunctionEvaluationFailed)?
                };
                Ok((local_iter, trial.clone(), obj))
            })
            .collect();

        let batch_results = batch_results?;

        // Observer: Track function evaluations only for actually evaluated points (not cached)
        if let Some(ref mut observer) = self.observer {
            if observer.should_observe_stage2() && ref_objectives.is_none() {
                if let Some(stage2) = observer.stage2_mut() {
                    stage2.add_function_evaluations(batch_results.len());
                }
            }
        }

        // Process results with parallel local solver calls where beneficial
        // Group by whether local search is needed to optimize parallelization
        let (local_candidates, quick_candidates): (Vec<_>, Vec<_>) =
            batch_results.into_iter().partition(|(_, trial, obj)| {
                let passes_merit = *obj <= self.merit_filter.threshold;
                let passes_distance = self.distance_filter.check(trial);
                passes_merit && passes_distance
            });

        // Process quick candidates (no local search needed) sequentially for efficiency
        for (local_iter, _trial, _obj) in quick_candidates {
            #[cfg(feature = "checkpointing")]
            {
                self.current_iteration = local_iter;
                self.unchanged_cycles = *unchanged_cycles;
                self.current_seed = self.params.seed + self.current_iteration as u64;
            }

            *unchanged_cycles += 1;
            if *unchanged_cycles >= self.params.wait_cycle {
                if self.verbose {
                    println!(
                        "Stage 2, iteration {}: Adjusting threshold from {:.8} to {:.8}",
                        local_iter,
                        self.merit_filter.threshold,
                        self.merit_filter.threshold + 0.1 * self.merit_filter.threshold.abs()
                    );
                }
                self.adjust_threshold(self.merit_filter.threshold);
                *unchanged_cycles = 0;
            }

            // Observer: Update iteration before callback
            if let Some(ref mut observer) = self.observer {
                if observer.should_observe_stage2() {
                    if let Some(stage2) = observer.stage2_mut() {
                        stage2.set_iteration(local_iter);
                        stage2.set_unchanged_cycles(*unchanged_cycles);
                        stage2.set_threshold_value(self.merit_filter.threshold);
                    }
                }
            }

            // Invoke observer callback if set and frequency matches
            if let Some(ref mut observer) = self.observer {
                if observer.should_invoke_callback(local_iter) {
                    observer.invoke_callback();
                }
            }

            #[cfg(feature = "checkpointing")]
            self.maybe_save_checkpoint()?;
        }

        // Process local search candidates in parallel when there are enough to justify overhead
        if !local_candidates.is_empty() {
            #[cfg(feature = "rayon")]
            if local_candidates.len() >= 2 && self.enable_parallel {
                // Clone once for sharing across threads
                let problem = self.problem.clone();
                let solver_type = self.params.local_solver_type.clone();
                let solver_config = self.params.local_solver_config.clone();
                let track_evals =
                    self.observer.as_ref().map(|obs| obs.should_observe_stage2()).unwrap_or(false);

                // Parallel processing for multiple local searches
                let local_results: Result<Vec<_>, OQNLPError> = local_candidates
                    .par_iter()
                    .map(|(local_iter, trial, obj)| {
                        // Each thread gets its own local solver instance
                        let local_solver = LocalSolver::new(
                            problem.clone(),
                            solver_type.clone(),
                            solver_config.clone(),
                        );

                        let (local_solution, eval_count) = if track_evals {
                            local_solver.solve_with_tracking(trial.clone(), true)?
                        } else {
                            let sol = local_solver.solve(trial.clone())?;
                            (sol, 0)
                        };

                        Ok((*local_iter, trial.clone(), *obj, local_solution, eval_count))
                    })
                    .collect();

                let local_results = local_results?;

                // Process local results sequentially to maintain state consistency
                for (local_iter, _trial, obj, local_solution, eval_count) in local_results {
                    #[cfg(feature = "checkpointing")]
                    {
                        self.current_iteration = local_iter;
                        self.unchanged_cycles = *unchanged_cycles;
                        self.current_seed = self.params.seed + self.current_iteration as u64;
                    }

                    self.merit_filter.update_threshold(obj);
                    let added = self.process_local_solution(local_solution.clone())?;

                    // Observer: Track local solver call and results (parallel case)
                    if let Some(ref mut observer) = self.observer {
                        if observer.should_observe_stage2() {
                            if let Some(stage2) = observer.stage2_mut() {
                                stage2.add_local_solver_call(added);
                                stage2.add_function_evaluations(eval_count as usize);
                                if let Some(ref sol_set) = self.solution_set {
                                    if let Some(best) = sol_set.best_solution() {
                                        stage2.set_best_solution(best.objective, &best.point);
                                    }
                                    stage2.set_solution_set_size(sol_set.len());
                                }
                            }
                        }
                    }

                    if self.verbose && added {
                        println!(
                            "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                            local_iter, local_solution.objective
                        );
                        println!("x0 = {}", local_solution.point);
                    }

                    // Observer: Update iteration before callback
                    if let Some(ref mut observer) = self.observer {
                        if observer.should_observe_stage2() {
                            if let Some(stage2) = observer.stage2_mut() {
                                stage2.set_iteration(local_iter);
                                stage2.set_unchanged_cycles(*unchanged_cycles);
                                stage2.set_threshold_value(self.merit_filter.threshold);
                            }
                        }
                    }

                    // Invoke observer callback if set and frequency matches
                    if let Some(ref mut observer) = self.observer {
                        if observer.should_invoke_callback(local_iter) {
                            observer.invoke_callback();
                        }
                    }

                    if self.target_objective_reached() {
                        if self.verbose {
                            println!(
                                "Stage 2, iteration {}: Target objective {:.8} reached. Stopping optimization.",
                                local_iter,
                                self.target_objective.unwrap()
                            );
                        }
                        return Ok(());
                    }

                    #[cfg(feature = "checkpointing")]
                    self.maybe_save_checkpoint()?;
                }
            } else {
                // Single local search - process directly without parallel overhead
                for (local_iter, trial, obj) in local_candidates {
                    #[cfg(feature = "checkpointing")]
                    {
                        self.current_iteration = local_iter;
                        self.unchanged_cycles = *unchanged_cycles;
                        self.current_seed = self.params.seed + self.current_iteration as u64;
                    }

                    self.merit_filter.update_threshold(obj);

                    // Only track evaluations if observer is active
                    let (local_trial, eval_count) = if let Some(ref mut observer) = self.observer {
                        if observer.should_observe_stage2() {
                            self.local_solver.solve_with_tracking(trial, true)?
                        } else {
                            let sol = self.local_solver.solve(trial)?;
                            (sol, 0)
                        }
                    } else {
                        let sol = self.local_solver.solve(trial)?;
                        (sol, 0)
                    };

                    let added = self.process_local_solution(local_trial.clone())?;

                    // Observer: Track local solver call and results (single local search case)
                    if let Some(ref mut observer) = self.observer {
                        if observer.should_observe_stage2() {
                            if let Some(stage2) = observer.stage2_mut() {
                                stage2.add_local_solver_call(added);
                                stage2.add_function_evaluations(eval_count as usize);
                                if let Some(ref sol_set) = self.solution_set {
                                    if let Some(best) = sol_set.best_solution() {
                                        stage2.set_best_solution(best.objective, &best.point);
                                    }
                                    stage2.set_solution_set_size(sol_set.len());
                                }
                            }
                        }
                    }

                    // Observer: Update iteration before callback
                    if let Some(ref mut observer) = self.observer {
                        if observer.should_observe_stage2() {
                            if let Some(stage2) = observer.stage2_mut() {
                                stage2.set_iteration(local_iter);
                                stage2.set_unchanged_cycles(*unchanged_cycles);
                                stage2.set_threshold_value(self.merit_filter.threshold);
                            }
                        }
                    }

                    // Invoke observer callback for stage 2 progress
                    if let Some(ref mut observer) = self.observer {
                        observer.invoke_callback();
                    }

                    if self.verbose && added {
                        println!(
                            "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                            local_iter, local_trial.objective
                        );
                        println!("x0 = {}", local_trial.point);
                    }

                    if self.target_objective_reached() {
                        if self.verbose {
                            println!(
                                "Stage 2, iteration {}: Target objective {:.8} reached. Stopping optimization.",
                                local_iter,
                                self.target_objective.unwrap()
                            );
                        }
                        return Ok(());
                    }

                    #[cfg(feature = "checkpointing")]
                    self.maybe_save_checkpoint()?;
                }
            }

            #[cfg(not(feature = "rayon"))]
            {
                // Sequential processing when Rayon is not available
                for (local_iter, trial, obj) in local_candidates {
                    #[cfg(feature = "checkpointing")]
                    {
                        self.current_iteration = local_iter;
                        self.unchanged_cycles = *unchanged_cycles;
                        self.current_seed = self.params.seed + self.current_iteration as u64;
                    }

                    self.merit_filter.update_threshold(obj);
                    let local_trial = self
                        .local_solver
                        .solve(trial)
                        .map_err(|e| OQNLPError::LocalSolverError(e.to_string()))?;
                    let added = self.process_local_solution(local_trial.clone())?;

                    if self.verbose && added {
                        println!(
                            "Stage 2, iteration {}: Added local solution found with objective = {:.8}",
                            local_iter, local_trial.objective
                        );
                        println!("x0 = {}", local_trial.point);
                    }

                    if self.target_objective_reached() {
                        if self.verbose {
                            println!(
                                "Stage 2, iteration {}: Target objective {:.8} reached. Stopping optimization.",
                                local_iter,
                                self.target_objective.unwrap()
                            );
                        }
                        return Ok(());
                    }

                    #[cfg(feature = "checkpointing")]
                    self.maybe_save_checkpoint()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_oqnlp {
    use super::*;
    use crate::types::EvaluationError;
    use ndarray::{array, Array1, Array2};

    // Dummy problem for testing, sum of variables with bounds from -5 to 5
    #[derive(Clone)]
    struct DummyProblem;
    impl Problem for DummyProblem {
        fn objective(&self, trial: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(trial.sum())
        }

        fn gradient(&self, trial: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(Array1::ones(trial.len()))
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        }
    }

    #[derive(Clone)]
    struct SixHumpCamel;
    impl Problem for SixHumpCamel {
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

    #[test]
    /// Test processing a new local solution
    fn test_process_local_solution_new() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        // Create a new OQNLP instance manually.
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        let trial = Array1::from(vec![1.0, 2.0, 3.0]);
        let ls: LocalSolution = LocalSolution { objective: trial.sum(), point: trial.clone() };

        let added: bool = oqnlp.process_local_solution(ls.clone()).unwrap();
        // For the first solution, no duplicate exists so added should be true
        // and the solution set should contain one solution
        assert!(added);

        let sol_set: SolutionSet = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - ls.objective).abs() < 1e-6);
    }

    #[test]
    /// Test that processing a duplicate solution does not add it
    fn test_process_local_solution_duplicate() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        let trial = Array1::from(vec![1.0, 2.0, 3.0]);
        let ls: LocalSolution = LocalSolution { objective: trial.sum(), point: trial.clone() };

        // Process the solution for the first time
        oqnlp.process_local_solution(ls.clone()).unwrap();

        // Process the duplicate solution, it should not be added
        let added: bool = oqnlp.process_local_solution(ls.clone()).unwrap();
        let sol_set = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!(!added);
    }

    #[test]
    /// Test that a better (lower objective) solution replaces the previous best.
    fn test_process_local_solution_better() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // First solution with objective 6.0
        let trial1 = Array1::from(vec![2.0, 2.0, 2.0]);
        let ls1: LocalSolution = LocalSolution { objective: trial1.sum(), point: trial1.clone() };
        oqnlp.process_local_solution(ls1).unwrap();

        // Second solution with objective 3.0 (better)
        let trial2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let ls2: LocalSolution = LocalSolution { objective: trial2.sum(), point: trial2.clone() };
        let added: bool = oqnlp.process_local_solution(ls2.clone()).unwrap();

        // When a new best is found, the solution set is replaced
        let sol_set: SolutionSet = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - ls2.objective).abs() < 1e-6);

        // The best solution replacement does not mark the solution as "added"
        // since it changes the existing solution set
        assert!(!added);
    }

    #[test]
    /// Test the helper for deciding whether to start a local search or not
    fn test_should_start_local() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // Set the merit filter threshold to a specific value for testing
        oqnlp.merit_filter.update_threshold(10.0);

        // A trial far enough in space but with objective above the threshold
        let trial = Array1::from(vec![10.0, 10.0, 10.0]);
        let obj: f64 = trial.sum(); // 30.0 > 10.0 threshold
        let start: bool = oqnlp.should_start_local(&trial, obj).unwrap();
        assert!(!start);

        // A trial with objective below threshold and spatially acceptable
        let trial2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let obj2: f64 = trial2.sum(); // 3.0 < 10.0 threshold
        let start2: bool = oqnlp.should_start_local(&trial2, obj2).unwrap();
        assert!(start2);
    }

    #[test]
    /// Test adjusting the merit filter threshold.
    fn test_adjust_threshold() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let mut oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        oqnlp.adjust_threshold(10.0);

        // New threshold = 10.0 + 0.2*(1 + 10.0) = 12.2
        assert!((oqnlp.merit_filter.threshold - 12.2).abs() < f64::EPSILON);
    }

    #[test]
    /// Test max_time method for setting the time limit
    fn test_max_time() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance manually
        let oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // Test setting the time limit
        let oqnlp: OQNLP<DummyProblem> = oqnlp.max_time(10.0);
        assert_eq!(oqnlp.max_time, Some(10.0));

        // Test using it in SixHumpCamel example
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams =
            OQNLPParams { iterations: 250, population_size: 1500, ..Default::default() };

        // With normal time limits, the algorithm should find both solutions
        let mut oqnlp: OQNLP<SixHumpCamel> =
            OQNLP::new(problem.clone(), params.clone()).unwrap().verbose().max_time(60.0);

        let sol_set: SolutionSet = oqnlp.run().unwrap();
        assert_eq!(sol_set.len(), 2);

        // With a very low time limit, the algorithm should stop before finding both solutions
        let mut oqnlp: OQNLP<SixHumpCamel> =
            OQNLP::new(problem, params).unwrap().verbose().max_time(0.00000000001);

        let sol_set: SolutionSet = oqnlp.run().unwrap();
        assert_eq!(sol_set.len(), 1);
    }

    #[test]
    /// Test the invalid population size for the OQNLP algorithm
    fn test_oqnlp_params_invalid_population_size() {
        let problem: DummyProblem = DummyProblem {};
        let params: OQNLPParams = OQNLPParams {
            population_size: 1, // Population size must be greater or equal to 3
            ..OQNLPParams::default()
        };

        let oqnlp = OQNLP::new(problem, params);
        assert!(matches!(oqnlp, Err(OQNLPError::InvalidPopulationSize(1))));
    }

    #[test]
    /// Test the invalid threshold factor for the OQNLP algorithm
    fn test_oqnlp_params_invalid_threshold_factor_zero() {
        let problem: DummyProblem = DummyProblem {};
        let params: OQNLPParams = OQNLPParams {
            threshold_factor: 0.0, // Threshold factor must be positive
            ..OQNLPParams::default()
        };

        let oqnlp = OQNLP::new(problem, params);
        assert!(matches!(oqnlp, Err(OQNLPError::InvalidThresholdFactor(0.0))));
    }

    #[test]
    /// Test the invalid threshold factor (negative) for the OQNLP algorithm
    fn test_oqnlp_params_invalid_threshold_factor_negative() {
        let problem: DummyProblem = DummyProblem {};
        let params: OQNLPParams = OQNLPParams {
            threshold_factor: -0.5, // Threshold factor must be positive
            ..OQNLPParams::default()
        };

        let oqnlp = OQNLP::new(problem, params);
        assert!(matches!(oqnlp, Err(OQNLPError::InvalidThresholdFactor(_))));
    }

    #[test]
    #[cfg(feature = "progress_bar")]
    /// Test the progress bar functionality
    fn test_progress_bar() {
        use kdam::term;
        use std::io::{stderr, IsTerminal};

        // Initialize terminal
        term::init(stderr().is_terminal());

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,       // Small number for testing
            population_size: 10, // Must be >= iterations
            ..Default::default()
        };

        let mut oqnlp = OQNLP::new(problem, params).unwrap().verbose();

        // Run OQNLP with progress bar
        let result = oqnlp.run();

        assert!(result.is_ok(), "OQNLP should run successfully with progress bar");

        // Verify that a solution was found
        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution");
    }

    #[test]
    /// Test the target objective functionality
    fn test_target_objective() {
        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 50, population_size: 100, ..Default::default() };

        // Test with a target that should be reached (SixHumpCamel has global minimum around -1.03)
        let mut oqnlp = OQNLP::new(problem.clone(), params.clone()).unwrap().target_objective(-0.5); // Set target higher than global minimum

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should run successfully");

        let sol_set = result.unwrap();
        let best = sol_set.best_solution().unwrap();

        // The algorithm should have stopped when it found a solution <= -0.5
        assert!(
            best.objective <= -0.5,
            "Best objective {} should be <= target -0.5",
            best.objective
        );

        // Test with a target that will never be reached
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap().target_objective(-10.0); // Set target much lower than possible

        let result2 = oqnlp2.run();
        assert!(result2.is_ok(), "OQNLP should run successfully even if target not reached");

        let sol_set2 = result2.unwrap();
        let best2 = sol_set2.best_solution().unwrap();

        // The algorithm should have run all iterations without reaching the target
        assert!(
            best2.objective > -10.0,
            "Best objective {} should be > impossible target -10.0",
            best2.objective
        );
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test the resume_with_modified_params functionality
    fn test_resume_with_modified_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params =
            OQNLPParams { iterations: 10, population_size: 20, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP with checkpointing to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        // Run for a few iterations to create checkpoints
        let _result = oqnlp.run();

        // Create a new OQNLP instance with modified parameters
        let modified_params = OQNLPParams {
            iterations: 25,      // Increased iterations
            population_size: 30, // Increased population size
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem.clone(), modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with modified parameters
        let resumed = oqnlp2.resume_with_modified_params(modified_params.clone());

        assert!(resumed.is_ok(), "Resume with modified params should succeed");
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.iterations, 25);
        assert_eq!(oqnlp2.params.population_size, 30);

        // Test case where no checkpoint exists
        let empty_checkpoint_dir = env::temp_dir().join("globalsearch_test_empty");
        std::fs::create_dir_all(&empty_checkpoint_dir).expect("Failed to create test directory");

        let empty_checkpoint_config = CheckpointConfig {
            checkpoint_dir: empty_checkpoint_dir.clone(),
            checkpoint_name: "nonexistent".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        let mut oqnlp3 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(empty_checkpoint_config)
            .unwrap();

        let not_resumed = oqnlp3.resume_with_modified_params(modified_params);
        assert!(not_resumed.is_ok(), "Should handle no checkpoint gracefully");
        assert!(!not_resumed.unwrap(), "Should return false when no checkpoint exists");

        // Clean up test directories
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let _ = std::fs::remove_dir_all(&empty_checkpoint_dir);
    }

    #[test]
    /// Test the exclude_out_of_bounds functionality
    fn test_exclude_out_of_bounds() {
        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        // Test with exclude_out_of_bounds enabled
        let mut oqnlp =
            OQNLP::new(problem.clone(), params.clone()).unwrap().exclude_out_of_bounds().verbose();

        // Verify the flag is set
        assert!(oqnlp.exclude_out_of_bounds);

        // Create a solution that is out of bounds
        // Problem bounds are [-5.0, 5.0] for each dimension
        let out_of_bounds_solution = LocalSolution {
            point: Array1::from(vec![10.0, 10.0, 10.0]), // All values exceed upper bound of 5.0
            objective: 30.0,
        };

        // Create a solution that is within bounds
        let within_bounds_solution = LocalSolution {
            point: Array1::from(vec![1.0, 2.0, 3.0]), // All values within [-5.0, 5.0]
            objective: 6.0,
        };

        // Test that out-of-bounds solution is rejected
        let added_out_of_bounds = oqnlp.process_local_solution(out_of_bounds_solution).unwrap();
        assert!(!added_out_of_bounds);
        assert!(oqnlp.solution_set.is_none()); // No solution should be added

        // Test that within-bounds solution is accepted
        let added_within_bounds =
            oqnlp.process_local_solution(within_bounds_solution.clone()).unwrap();
        assert!(added_within_bounds);
        assert!(oqnlp.solution_set.is_some());
        let sol_set = oqnlp.solution_set.unwrap();
        assert_eq!(sol_set.len(), 1);
        assert!((sol_set[0].objective - within_bounds_solution.objective).abs() < 1e-6);

        // Test with exclude_out_of_bounds disabled (default behavior)
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap();
        assert!(!oqnlp2.exclude_out_of_bounds); // Should be false by default

        // Out-of-bounds solution should be accepted when flag is disabled
        let out_of_bounds_solution2 =
            LocalSolution { point: Array1::from(vec![15.0, 20.0, 25.0]), objective: 60.0 };

        let added_out_of_bounds2 =
            oqnlp2.process_local_solution(out_of_bounds_solution2.clone()).unwrap();
        assert!(added_out_of_bounds2);
        assert!(oqnlp2.solution_set.is_some());
        let sol_set2 = oqnlp2.solution_set.unwrap();
        assert_eq!(sol_set2.len(), 1);
        assert!((sol_set2[0].objective - out_of_bounds_solution2.objective).abs() < 1e-6);
    }

    #[test]
    /// Test exclude_out_of_bounds with target_objective
    fn test_exclude_out_of_bounds_with_target_objective() {
        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .exclude_out_of_bounds()
            .target_objective(50.0)
            .verbose();

        // Create an out-of-bounds solution that meets the target objective
        let out_of_bounds_good_obj = LocalSolution {
            point: Array1::from(vec![10.0, 10.0, 10.0]), // Out of bounds
            objective: 30.0,                             // Meets target objective (< 50.0)
        };

        // Create a within-bounds solution that meets the target objective
        let within_bounds_good_obj = LocalSolution {
            point: Array1::from(vec![1.0, 2.0, 3.0]), // Within bounds
            objective: 40.0,                          // Meets target objective (< 50.0)
        };

        // Create a within-bounds solution that doesn't meet the target objective
        let within_bounds_bad_obj = LocalSolution {
            point: Array1::from(vec![0.0, 0.0, 0.0]), // Within bounds
            objective: 60.0,                          // Doesn't meet target objective (> 50.0)
        };

        // Process out-of-bounds solution - should be rejected even if it meets target
        oqnlp.process_local_solution(out_of_bounds_good_obj).unwrap();
        assert!(!oqnlp.target_objective_reached()); // Should not reach target due to bounds

        // Process within-bounds solution that meets target - should be accepted
        oqnlp.process_local_solution(within_bounds_good_obj).unwrap();
        assert!(oqnlp.target_objective_reached()); // Should reach target

        // Reset for next test
        oqnlp.solution_set = None;

        // Process within-bounds solution that doesn't meet target - should be accepted but target not reached
        oqnlp.process_local_solution(within_bounds_bad_obj).unwrap();
        assert!(!oqnlp.target_objective_reached()); // Should not reach target due to objective
    }

    #[test]
    /// Test is_within_bounds helper method
    fn test_is_within_bounds() {
        let problem = DummyProblem; // Bounds are [-5.0, 5.0] for each dimension
        let params = OQNLPParams::default();
        let oqnlp = OQNLP::new(problem, params).unwrap();

        // Test point within bounds
        let within_bounds = Array1::from(vec![1.0, 2.0, 3.0]);
        assert!(oqnlp.is_within_bounds(&within_bounds));

        // Test point at lower bound
        let at_lower_bound = Array1::from(vec![-5.0, -5.0, -5.0]);
        assert!(oqnlp.is_within_bounds(&at_lower_bound));

        // Test point at upper bound
        let at_upper_bound = Array1::from(vec![5.0, 5.0, 5.0]);
        assert!(oqnlp.is_within_bounds(&at_upper_bound));

        // Test point below lower bound
        let below_lower_bound = Array1::from(vec![-6.0, 0.0, 0.0]);
        assert!(!oqnlp.is_within_bounds(&below_lower_bound));

        // Test point above upper bound
        let above_upper_bound = Array1::from(vec![0.0, 6.0, 0.0]);
        assert!(!oqnlp.is_within_bounds(&above_upper_bound));

        // Test empty point (edge case)
        let empty_point = Array1::from(vec![]);
        assert!(oqnlp.is_within_bounds(&empty_point)); // Empty point should be considered within bounds
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test that exclude_out_of_bounds is properly saved and restored in checkpoints
    fn test_exclude_out_of_bounds_checkpointing() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_exclude_bounds");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_exclude_bounds".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP with exclude_out_of_bounds enabled
        let oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .exclude_out_of_bounds()
            .verbose();

        // Create a checkpoint
        let checkpoint = oqnlp.create_checkpoint();
        assert!(checkpoint.exclude_out_of_bounds);

        // Create new OQNLP instance and restore from checkpoint
        let mut oqnlp2 =
            OQNLP::new(problem, params).unwrap().with_checkpointing(checkpoint_config).unwrap();

        // Before restoration, exclude_out_of_bounds should be false
        assert!(!oqnlp2.exclude_out_of_bounds);

        // Restore from checkpoint
        oqnlp2.restore_from_checkpoint(checkpoint).unwrap();

        // After restoration, exclude_out_of_bounds should be true
        assert!(oqnlp2.exclude_out_of_bounds);

        // Clean up
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_with_modified_params with decreased population size
    fn test_resume_with_modified_params_decreased_population() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_decrease");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params =
            OQNLPParams { iterations: 10, population_size: 30, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_decrease".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP with checkpointing
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Create modified parameters with decreased population size
        let modified_params = OQNLPParams {
            iterations: 15,
            population_size: 20, // Decreased from 30 to 20
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with decreased population size
        let resumed = oqnlp2.resume_with_modified_params(modified_params);

        assert!(resumed.is_ok(), "Resume with decreased population should succeed");
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // The reference set should keep the original size despite the smaller population_size parameter
        // This is tested implicitly by ensuring the function succeeds without error

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test the expand_reference_set method functionality
    fn test_expand_reference_set() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        // Create OQNLP instance
        let oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Create a small reference set to expand
        let mut ref_set = vec![
            Array1::from(vec![-1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![1.0, 1.0]),
        ];
        let old_size = ref_set.len();
        let new_size = 8;

        // Test expanding the reference set
        let result = oqnlp.expand_reference_set(&mut ref_set, old_size, new_size);

        assert!(result.is_ok(), "expand_reference_set should succeed");
        assert_eq!(ref_set.len(), new_size, "Reference set should be expanded to new size");

        // Verify that original points are still there
        assert!(ref_set.contains(&Array1::from(vec![-1.0, -1.0])));
        assert!(ref_set.contains(&Array1::from(vec![0.0, 0.0])));
        assert!(ref_set.contains(&Array1::from(vec![1.0, 1.0])));

        // Verify that new points are within bounds
        let bounds = problem.variable_bounds();
        for point in &ref_set {
            assert!(point.len() == bounds.nrows(), "Point should have correct dimensions");
            for (i, &val) in point.iter().enumerate() {
                assert!(
                    val >= bounds[[i, 0]] && val <= bounds[[i, 1]],
                    "Point value {} should be within bounds [{}, {}]",
                    val,
                    bounds[[i, 0]],
                    bounds[[i, 1]]
                );
            }
        }

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test expand_reference_set with edge cases
    fn test_expand_reference_set_edge_cases() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        let oqnlp =
            OQNLP::new(problem, params).unwrap().with_checkpointing(checkpoint_config).unwrap();

        // Test case 1: new_size <= old_size (should return immediately)
        let mut ref_set1 = vec![
            Array1::from(vec![-1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![1.0, 1.0]),
        ];
        let original_len = ref_set1.len();

        let result1 = oqnlp.expand_reference_set(&mut ref_set1, 5, 3);
        assert!(result1.is_ok(), "expand_reference_set should handle new_size <= old_size");
        assert_eq!(
            ref_set1.len(),
            original_len,
            "Reference set should not change when new_size <= old_size"
        );

        // Test case 2: new_size == old_size (should return immediately)
        let mut ref_set2 = vec![Array1::from(vec![-1.0, -1.0]), Array1::from(vec![0.0, 0.0])];
        let original_len2 = ref_set2.len();

        let result2 = oqnlp.expand_reference_set(&mut ref_set2, original_len2, original_len2);
        assert!(result2.is_ok(), "expand_reference_set should handle new_size == old_size");
        assert_eq!(
            ref_set2.len(),
            original_len2,
            "Reference set should not change when new_size == old_size"
        );

        // Test case 3: Empty reference set expansion
        let mut ref_set3: Vec<Array1<f64>> = vec![];
        let result3 = oqnlp.expand_reference_set(&mut ref_set3, 0, 5);
        assert!(result3.is_ok(), "expand_reference_set should handle empty reference set");
        assert_eq!(ref_set3.len(), 5, "Reference set should be expanded from empty to new size");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test expand_reference_set integration through resume_with_modified_params
    fn test_expand_reference_set_integration() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_expand_integration");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params =
            OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_expand_integration".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: true,
        };

        // Create and run initial OQNLP to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Create modified parameters with significantly larger population size
        let modified_params = OQNLPParams {
            iterations: 8,
            population_size: 25, // Significantly increased from 10 to 25
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume with modified parameters (this should trigger expand_reference_set internally)
        let resumed = oqnlp2.resume_with_modified_params(modified_params);

        assert!(resumed.is_ok(), "Resume with larger population should succeed");
        assert!(resumed.unwrap(), "Should have resumed from checkpoint");

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.population_size, 25);

        // The internal reference set should have been expanded
        // This is tested indirectly by ensuring the resume operation succeeds
        // and the algorithm can continue with the larger population size

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params functionality
    fn test_resume_from_checkpoint_with_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_specific");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params =
            OQNLPParams { iterations: 8, population_size: 15, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume_specific".to_string(),
            save_frequency: 2,
            keep_all: true,     // Keep all checkpoints for specific file testing
            auto_resume: false, // Don't auto-resume to test specific file loading
        };

        // Create and run initial OQNLP with checkpointing to create checkpoint files
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Find the checkpoint file that was created
        let checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoint_dir)
            .expect("Failed to read checkpoint directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "bin"
                    && path.file_name()?.to_str()?.contains("test_resume_specific")
                {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        assert!(!checkpoint_files.is_empty(), "Should have created at least one checkpoint file");

        // Use the first checkpoint file found
        let checkpoint_path = &checkpoint_files[0];

        // Create modified parameters
        let modified_params = OQNLPParams {
            iterations: 20,      // Increased iterations
            population_size: 25, // Increased population size
            ..initial_params
        };

        // Create a new OQNLP instance
        let mut oqnlp2 = OQNLP::new(problem.clone(), modified_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Test resume from specific checkpoint with modified parameters
        let result =
            oqnlp2.resume_from_checkpoint_with_params(checkpoint_path, modified_params.clone());

        assert!(result.is_ok(), "Resume from specific checkpoint should succeed");

        // Verify that the parameters were updated
        assert_eq!(oqnlp2.params.iterations, 20);
        assert_eq!(oqnlp2.params.population_size, 25);

        // Verify that the algorithm can continue running with the loaded state
        let continued_result = oqnlp2.run();
        assert!(continued_result.is_ok(), "Should be able to continue optimization after resume");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params with nonexistent file
    fn test_resume_from_checkpoint_with_params_nonexistent_file() {
        use crate::types::CheckpointConfig;
        use std::env;
        use std::path::PathBuf;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_nonexistent");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_nonexistent".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Try to resume from a nonexistent checkpoint file
        let nonexistent_path = PathBuf::from("nonexistent_checkpoint.bin");
        let result = oqnlp.resume_from_checkpoint_with_params(&nonexistent_path, params);

        // Should return an error for nonexistent file
        assert!(result.is_err(), "Should return error for nonexistent checkpoint file");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params without checkpoint manager
    fn test_resume_from_checkpoint_with_params_no_manager() {
        use std::path::PathBuf;

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        // Create OQNLP without checkpointing
        let mut oqnlp = OQNLP::new(problem, params.clone()).unwrap();

        // Try to resume from checkpoint without checkpoint manager
        let dummy_path = PathBuf::from("dummy_checkpoint.bin");
        let result = oqnlp.resume_from_checkpoint_with_params(&dummy_path, params);

        // Should succeed but do nothing when no checkpoint manager is present
        assert!(result.is_ok(), "Should handle absence of checkpoint manager gracefully");
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test resume_from_checkpoint_with_params with different parameter combinations
    fn test_resume_from_checkpoint_with_params_various_params() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_resume_various");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 6,
            population_size: 12,
            distance_factor: 0.1,
            threshold_factor: 0.3,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_resume_various".to_string(),
            save_frequency: 1,
            keep_all: true,
            auto_resume: false,
        };

        // Create and run initial OQNLP to create a checkpoint
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose();

        let _result = oqnlp.run();

        // Find a checkpoint file
        let checkpoint_files: Vec<_> = std::fs::read_dir(&checkpoint_dir)
            .expect("Failed to read checkpoint directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()? == "bin" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        assert!(!checkpoint_files.is_empty(), "Should have created checkpoint files");
        let checkpoint_path = &checkpoint_files[0];

        // Test 1: Modify only iterations (keeping it within population_size constraint)
        let modified_params1 = OQNLPParams {
            iterations: 10, // Increase iterations but keep <= population_size (12)
            ..initial_params.clone()
        };

        let mut oqnlp1 = OQNLP::new(problem.clone(), modified_params1.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap();

        let result1 = oqnlp1.resume_from_checkpoint_with_params(checkpoint_path, modified_params1);
        assert!(result1.is_ok(), "Should handle iterations-only modification");
        assert_eq!(oqnlp1.params.iterations, 10);

        // Test 2: Modify multiple parameters
        let modified_params2 = OQNLPParams {
            iterations: 25,
            population_size: 30,
            distance_factor: 0.05,
            threshold_factor: 0.4,
            ..initial_params
        };

        let mut oqnlp2 = OQNLP::new(problem, modified_params2.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        let result2 = oqnlp2.resume_from_checkpoint_with_params(checkpoint_path, modified_params2);
        assert!(result2.is_ok(), "Should handle multiple parameter modifications");
        assert_eq!(oqnlp2.params.iterations, 25);
        assert_eq!(oqnlp2.params.population_size, 30);
        assert!((oqnlp2.params.distance_factor - 0.05).abs() < 1e-10);
        assert!((oqnlp2.params.threshold_factor - 0.4).abs() < 1e-10);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint functionality
    fn test_restore_from_checkpoint() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let initial_params = OQNLPParams {
            iterations: 15,
            population_size: 20,
            distance_factor: 0.15,
            threshold_factor: 0.25,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance
        let mut oqnlp = OQNLP::new(problem.clone(), initial_params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose()
            .target_objective(-0.8);

        // Create some solutions for the checkpoint
        let solution1 = LocalSolution { objective: -0.5, point: Array1::from(vec![0.1, -0.7]) };
        let solution2 = LocalSolution { objective: -0.3, point: Array1::from(vec![-0.9, 0.2]) };

        let solution_set =
            SolutionSet { solutions: Array1::from(vec![solution1.clone(), solution2.clone()]) };

        // Create a mock checkpoint with comprehensive state
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams {
                iterations: 25,
                population_size: 30,
                distance_factor: 0.08,
                threshold_factor: 0.35,
                ..initial_params.clone()
            },
            current_iteration: 12,
            merit_threshold: -0.2,
            solution_set: Some(solution_set.clone()),
            reference_set: vec![
                Array1::from(vec![-1.5, 1.0]),
                Array1::from(vec![0.5, -1.2]),
                Array1::from(vec![2.0, 0.8]),
            ],
            unchanged_cycles: 3,
            elapsed_time: 45.67,
            distance_filter_solutions: vec![solution1.clone(), solution2.clone()],
            current_seed: 98765,
            target_objective: Some(-0.9),
            exclude_out_of_bounds: true,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(2),
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Test restore_from_checkpoint
        let result = oqnlp.restore_from_checkpoint(checkpoint.clone());

        assert!(result.is_ok(), "restore_from_checkpoint should succeed");

        // Verify all state was restored correctly
        assert_eq!(oqnlp.params.iterations, 25, "Iterations should be restored");
        assert_eq!(oqnlp.params.population_size, 30, "Population size should be restored");
        assert!(
            (oqnlp.params.distance_factor - 0.08).abs() < 1e-10,
            "Distance factor should be restored"
        );
        assert!(
            (oqnlp.params.threshold_factor - 0.35).abs() < 1e-10,
            "Threshold factor should be restored"
        );

        assert_eq!(oqnlp.current_iteration, 12, "Current iteration should be restored");
        assert!(
            (oqnlp.merit_filter.threshold - (-0.2)).abs() < 1e-10,
            "Merit threshold should be restored"
        );
        assert_eq!(oqnlp.unchanged_cycles, 3, "Unchanged cycles should be restored");
        assert_eq!(oqnlp.current_seed, 98765, "Current seed should be restored");
        assert_eq!(oqnlp.target_objective, Some(-0.9), "Target objective should be restored");

        // Verify solution set was restored
        let restored_solution_set =
            oqnlp.solution_set.as_ref().expect("Solution set should be restored");
        assert_eq!(restored_solution_set.len(), 2, "Solution set should have 2 solutions");
        assert!(
            (restored_solution_set[0].objective - (-0.5)).abs() < 1e-10,
            "First solution objective should match"
        );
        assert!(
            (restored_solution_set[1].objective - (-0.3)).abs() < 1e-10,
            "Second solution objective should match"
        );

        // Verify reference set was restored
        let restored_ref_set =
            oqnlp.current_reference_set.as_ref().expect("Reference set should be restored");
        assert_eq!(restored_ref_set.len(), 3, "Reference set should have 3 points");
        assert_eq!(
            restored_ref_set[0],
            Array1::from(vec![-1.5, 1.0]),
            "First reference point should match"
        );
        assert_eq!(
            restored_ref_set[1],
            Array1::from(vec![0.5, -1.2]),
            "Second reference point should match"
        );
        assert_eq!(
            restored_ref_set[2],
            Array1::from(vec![2.0, 0.8]),
            "Third reference point should match"
        );

        // Verify distance filter solutions were restored
        let distance_filter_solutions = oqnlp.distance_filter.get_solutions();
        assert_eq!(distance_filter_solutions.len(), 2, "Distance filter should have 2 solutions");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with empty solution set
    fn test_restore_from_checkpoint_empty_solution_set() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_empty");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 10, population_size: 15, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_empty".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .verbose();

        // Create checkpoint with no solution set
        let checkpoint = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 5,
            merit_threshold: 100.0,
            solution_set: None, // No solutions
            reference_set: vec![Array1::from(vec![0.0, 0.0])],
            unchanged_cycles: 0,
            elapsed_time: 10.0,
            distance_filter_solutions: vec![],
            current_seed: 12345,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result = oqnlp.restore_from_checkpoint(checkpoint);

        assert!(result.is_ok(), "Should handle empty solution set gracefully");
        assert!(oqnlp.solution_set.is_none(), "Solution set should remain None");
        assert_eq!(oqnlp.current_iteration, 5, "Current iteration should be restored");
        assert!(
            (oqnlp.merit_filter.threshold - 100.0).abs() < 1e-10,
            "Merit threshold should be restored"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with verbose output
    fn test_restore_from_checkpoint_verbose_output() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_verbose");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 8, population_size: 12, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_verbose".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Test with verbose enabled
        let mut oqnlp_verbose = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .verbose(); // Enable verbose output

        let solution = LocalSolution { objective: -1.2, point: Array1::from(vec![0.5, -0.3]) };

        let solution_set = SolutionSet { solutions: Array1::from(vec![solution]) };

        let checkpoint_with_solutions = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 7,
            merit_threshold: -1.0,
            solution_set: Some(solution_set),
            reference_set: vec![Array1::from(vec![1.0, 1.0])],
            unchanged_cycles: 2,
            elapsed_time: 25.0,
            distance_filter_solutions: vec![],
            current_seed: 54321,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(6),
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // This should print verbose output (iteration 7 with 1 solutions)
        let result_verbose = oqnlp_verbose.restore_from_checkpoint(checkpoint_with_solutions);
        assert!(result_verbose.is_ok(), "Verbose restore should succeed");

        // Test with verbose disabled
        let mut oqnlp_quiet = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();
        // No .verbose() call - should be quiet

        let checkpoint_no_solutions = OQNLPCheckpoint {
            params,
            current_iteration: 3,
            merit_threshold: 50.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![-1.0, -1.0])],
            unchanged_cycles: 1,
            elapsed_time: 15.0,
            distance_filter_solutions: vec![],
            current_seed: 11111,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // This should not print any output (verbose is false)
        let result_quiet = oqnlp_quiet.restore_from_checkpoint(checkpoint_no_solutions);
        assert!(result_quiet.is_ok(), "Quiet restore should succeed");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test restore_from_checkpoint with various edge cases
    fn test_restore_from_checkpoint_edge_cases() {
        use crate::types::{CheckpointConfig, OQNLPCheckpoint};
        use chrono;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_restore_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 8, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_restore_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Test case 1: Checkpoint at iteration 0
        let checkpoint_zero = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 0,
            merit_threshold: 1000.0,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 1,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result1 = oqnlp.restore_from_checkpoint(checkpoint_zero);
        assert!(result1.is_ok(), "Should handle iteration 0 checkpoint");
        assert_eq!(oqnlp.current_iteration, 0);
        assert_eq!(oqnlp.unchanged_cycles, 0);

        // Test case 2: Checkpoint with very large merit threshold
        let checkpoint_large_threshold = OQNLPCheckpoint {
            params: params.clone(),
            current_iteration: 2,
            merit_threshold: f64::MAX / 2.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![0.0, 0.0])],
            unchanged_cycles: 1,
            elapsed_time: 5.0,
            distance_filter_solutions: vec![],
            current_seed: 999,
            target_objective: Some(f64::MIN / 2.0),
            exclude_out_of_bounds: true,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(8),
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result2 = oqnlp.restore_from_checkpoint(checkpoint_large_threshold);
        assert!(result2.is_ok(), "Should handle large threshold values");
        assert!((oqnlp.merit_filter.threshold - (f64::MAX / 2.0)).abs() < f64::MAX / 4.0);
        assert_eq!(oqnlp.target_objective, Some(f64::MIN / 2.0));

        // Test case 3: Checkpoint with many unchanged cycles
        let checkpoint_many_cycles = OQNLPCheckpoint {
            params,
            current_iteration: 4,
            merit_threshold: 0.0,
            solution_set: None,
            reference_set: vec![Array1::from(vec![1.0, -1.0])],
            unchanged_cycles: 1000,
            elapsed_time: 100.0,
            distance_filter_solutions: vec![],
            current_seed: 777,
            target_objective: Some(0.0),
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(1),
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let result3 = oqnlp.restore_from_checkpoint(checkpoint_many_cycles);
        assert!(result3.is_ok(), "Should handle many unchanged cycles");
        assert_eq!(oqnlp.unchanged_cycles, 1000);
        assert_eq!(oqnlp.current_seed, 777);

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint functionality
    fn test_create_checkpoint() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_checkpoint");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 20,
            population_size: 25,
            distance_factor: 0.12,
            threshold_factor: 0.28,
            seed: 42,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance and set up some state
        let mut oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .target_objective(-0.75);

        // Manually set up some internal state to test checkpoint creation
        oqnlp.current_iteration = 8;
        oqnlp.unchanged_cycles = 2;
        oqnlp.current_seed = 98765;
        oqnlp.merit_filter.update_threshold(-0.3);

        // Set up some solutions
        let solution1 = LocalSolution { objective: -0.6, point: Array1::from(vec![0.2, -0.8]) };
        let solution2 = LocalSolution { objective: -0.4, point: Array1::from(vec![-0.7, 0.3]) };

        let solution_set =
            SolutionSet { solutions: Array1::from(vec![solution1.clone(), solution2.clone()]) };
        oqnlp.solution_set = Some(solution_set);

        // Set up reference set
        oqnlp.current_reference_set = Some(vec![
            Array1::from(vec![-2.0, 1.5]),
            Array1::from(vec![1.0, -1.0]),
            Array1::from(vec![0.0, 0.0]),
        ]);

        // Add solutions to distance filter
        oqnlp.distance_filter.add_solution(solution1.clone());
        oqnlp.distance_filter.add_solution(solution2.clone());

        // Set start time to test elapsed time calculation
        oqnlp.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(30));

        // Create checkpoint
        let checkpoint = oqnlp.create_checkpoint();

        // Verify all checkpoint fields are correct
        assert_eq!(checkpoint.params.iterations, 20, "Parameters should be captured");
        assert_eq!(checkpoint.params.population_size, 25, "Parameters should be captured");
        assert!(
            (checkpoint.params.distance_factor - 0.12).abs() < 1e-10,
            "Parameters should be captured"
        );
        assert!(
            (checkpoint.params.threshold_factor - 0.28).abs() < 1e-10,
            "Parameters should be captured"
        );
        assert_eq!(checkpoint.params.seed, 42, "Parameters should be captured");

        assert_eq!(checkpoint.current_iteration, 8, "Current iteration should be captured");
        assert!(
            (checkpoint.merit_threshold - (-0.3)).abs() < 1e-10,
            "Merit threshold should be captured"
        );
        assert_eq!(checkpoint.unchanged_cycles, 2, "Unchanged cycles should be captured");
        assert_eq!(checkpoint.current_seed, 98765, "Current seed should be captured");
        assert_eq!(checkpoint.target_objective, Some(-0.75), "Target objective should be captured");

        // Verify solution set
        let checkpoint_solutions =
            checkpoint.solution_set.as_ref().expect("Solution set should be captured");
        assert_eq!(checkpoint_solutions.len(), 2, "Solution set should have 2 solutions");
        assert!(
            (checkpoint_solutions[0].objective - (-0.6)).abs() < 1e-10,
            "First solution should match"
        );
        assert!(
            (checkpoint_solutions[1].objective - (-0.4)).abs() < 1e-10,
            "Second solution should match"
        );

        // Verify reference set
        assert_eq!(checkpoint.reference_set.len(), 3, "Reference set should have 3 points");
        assert_eq!(
            checkpoint.reference_set[0],
            Array1::from(vec![-2.0, 1.5]),
            "Reference point should match"
        );
        assert_eq!(
            checkpoint.reference_set[1],
            Array1::from(vec![1.0, -1.0]),
            "Reference point should match"
        );
        assert_eq!(
            checkpoint.reference_set[2],
            Array1::from(vec![0.0, 0.0]),
            "Reference point should match"
        );

        // Verify distance filter solutions
        assert_eq!(
            checkpoint.distance_filter_solutions.len(),
            2,
            "Distance filter should have 2 solutions"
        );

        // Verify elapsed time (should be approximately 30 seconds, give or take a few milliseconds)
        assert!(
            checkpoint.elapsed_time >= 29.0 && checkpoint.elapsed_time <= 31.0,
            "Elapsed time should be approximately 30 seconds, got {}",
            checkpoint.elapsed_time
        );

        // Verify timestamp is valid RFC3339 format
        assert!(
            chrono::DateTime::parse_from_rfc3339(&checkpoint.timestamp).is_ok(),
            "Timestamp should be valid RFC3339 format"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint with empty state
    fn test_create_checkpoint_empty_state() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_empty");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 5, population_size: 8, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create_empty".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create OQNLP instance with minimal state
        let oqnlp = OQNLP::new(problem, params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap();

        // Create checkpoint with default/empty state
        let checkpoint = oqnlp.create_checkpoint();

        // Verify checkpoint captures empty/default state correctly
        assert_eq!(checkpoint.params.iterations, 5, "Parameters should be captured");
        assert_eq!(checkpoint.params.population_size, 8, "Parameters should be captured");
        assert_eq!(checkpoint.current_iteration, 0, "Should start at iteration 0");
        assert_eq!(checkpoint.unchanged_cycles, 0, "Should start with 0 unchanged cycles");
        assert_eq!(checkpoint.current_seed, params.seed, "Should have initial seed");
        assert_eq!(checkpoint.target_objective, None, "Should have no target objective");

        // Verify empty collections
        assert!(checkpoint.solution_set.is_none(), "Solution set should be None initially");
        assert!(checkpoint.reference_set.is_empty(), "Reference set should be empty initially");
        assert!(
            checkpoint.distance_filter_solutions.is_empty(),
            "Distance filter should be empty initially"
        );

        // Verify elapsed time is 0 when no start time is set
        assert_eq!(checkpoint.elapsed_time, 0.0, "Elapsed time should be 0 when no start time");

        // Verify timestamp is valid
        assert!(
            chrono::DateTime::parse_from_rfc3339(&checkpoint.timestamp).is_ok(),
            "Timestamp should be valid RFC3339 format"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test create_checkpoint with various edge cases
    fn test_create_checkpoint_edge_cases() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_create_edge");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 10, population_size: 15, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_create_edge".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .target_objective(f64::NEG_INFINITY); // Edge case: extreme target

        // Test case 1: Very large iteration number
        oqnlp.current_iteration = usize::MAX / 2;
        oqnlp.unchanged_cycles = 999999;
        oqnlp.current_seed = u64::MAX / 2;

        // Set extreme merit threshold
        oqnlp.merit_filter.update_threshold(f64::MAX / 2.0);

        // Set start time in the past (edge case for elapsed time)
        oqnlp.start_time = Some(
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(60))
                .unwrap_or(std::time::Instant::now()),
        ); // 1 minute ago

        let checkpoint1 = oqnlp.create_checkpoint();

        assert_eq!(
            checkpoint1.current_iteration,
            usize::MAX / 2,
            "Should handle large iteration numbers"
        );
        assert_eq!(checkpoint1.unchanged_cycles, 999999, "Should handle large unchanged cycles");
        assert_eq!(checkpoint1.current_seed, u64::MAX / 2, "Should handle large seed values");
        assert!(
            (checkpoint1.merit_threshold - (f64::MAX / 2.0)).abs() < f64::MAX / 4.0,
            "Should handle large threshold"
        );
        assert_eq!(
            checkpoint1.target_objective,
            Some(f64::NEG_INFINITY),
            "Should handle extreme target objective"
        );
        assert!(
            checkpoint1.elapsed_time >= 55.0 && checkpoint1.elapsed_time <= 65.0,
            "Should handle elapsed time, got {}",
            checkpoint1.elapsed_time
        );

        // Test case 2: Reset to minimal values
        oqnlp.current_iteration = 0;
        oqnlp.unchanged_cycles = 0;
        oqnlp.current_seed = 1;
        oqnlp.merit_filter.update_threshold(0.0);
        oqnlp.target_objective = Some(0.0);
        oqnlp.start_time = Some(std::time::Instant::now()); // Just started

        let checkpoint2 = oqnlp.create_checkpoint();

        assert_eq!(checkpoint2.current_iteration, 0, "Should handle zero iteration");
        assert_eq!(checkpoint2.unchanged_cycles, 0, "Should handle zero unchanged cycles");
        assert_eq!(checkpoint2.current_seed, 1, "Should handle minimal seed");
        assert!((checkpoint2.merit_threshold - 0.0).abs() < 1e-10, "Should handle zero threshold");
        assert_eq!(checkpoint2.target_objective, Some(0.0), "Should handle zero target objective");
        assert!(
            checkpoint2.elapsed_time >= 0.0 && checkpoint2.elapsed_time <= 1.0,
            "Should handle minimal elapsed time, got {}",
            checkpoint2.elapsed_time
        );

        // Verify timestamps are different (showing time progression)
        assert_ne!(checkpoint1.timestamp, checkpoint2.timestamp, "Timestamps should be different");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test the batch_iterations method for setting batch size
    fn test_batch_iterations() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams::default();

        // Create a new OQNLP instance and test batch_iterations method
        let oqnlp: OQNLP<DummyProblem> = OQNLP::new(problem, params).unwrap();

        // Test setting batch_iterations
        let oqnlp = oqnlp.batch_iterations(4);
        assert_eq!(oqnlp.batch_iterations, Some(4));

        // Test chaining with other methods
        let problem2: DummyProblem = DummyProblem;
        let params2: OQNLPParams = OQNLPParams::default();
        let oqnlp2 =
            OQNLP::new(problem2, params2).unwrap().batch_iterations(8).max_time(30.0).verbose();

        assert_eq!(oqnlp2.batch_iterations, Some(8));
        assert_eq!(oqnlp2.max_time, Some(30.0));
        assert!(oqnlp2.verbose);
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch processing with sequential execution
    fn test_batch_processing_sequential() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams = OQNLPParams {
            iterations: 6,       // Small number for testing
            population_size: 10, // Must be >= iterations
            ..Default::default()
        };

        // Test with batch_iterations = 1 (should force sequential processing)
        let mut oqnlp =
            OQNLP::new(problem.clone(), params.clone()).unwrap().batch_iterations(1).verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should run successfully with batch_iterations = 1");

        // Verify that solutions were found
        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution");

        // Test with default (no batch_iterations set)
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap().verbose();

        let result2 = oqnlp2.run();
        assert!(result2.is_ok(), "OQNLP should run successfully without batch_iterations");

        let sol_set2 = result2.unwrap();
        assert!(!sol_set2.is_empty(), "Should find at least one solution");
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch processing with parallel execution (when rayon feature is enabled)
    fn test_batch_processing_parallel() {
        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 8,       // Small number for testing
            population_size: 12, // Must be >= iterations
            ..Default::default()
        };

        // Test with batch_iterations > 1 (should enable parallel processing when rayon is available)
        let mut oqnlp =
            OQNLP::new(problem.clone(), params.clone()).unwrap().batch_iterations(4).verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should run successfully with parallel batch processing");

        // Verify that solutions were found
        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution with parallel processing");

        // Test with auto-determined batch size (should use number of CPUs)
        let mut oqnlp2 = OQNLP::new(problem, params).unwrap().verbose();

        let result2 = oqnlp2.run();
        assert!(result2.is_ok(), "OQNLP should run successfully with auto batch size");

        let sol_set2 = result2.unwrap();
        assert!(!sol_set2.is_empty(), "Should find solutions with auto batch size");
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch_iterations with target_objective
    fn test_batch_iterations_with_target_objective() {
        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 20, population_size: 30, ..Default::default() };

        // Test that batch processing respects target_objective stopping condition
        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .batch_iterations(3)
            .target_objective(-0.5) // Should stop when this target is reached
            .verbose();

        let result = oqnlp.run();
        assert!(
            result.is_ok(),
            "OQNLP should run successfully with batch_iterations and target_objective"
        );

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution");

        // Check that the best solution meets the target
        let best = sol_set.best_solution().unwrap();
        assert!(
            best.objective <= -0.5,
            "Best objective {} should be <= target -0.5",
            best.objective
        );
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch_iterations with max_time
    fn test_batch_iterations_with_max_time() {
        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 100, // Large number that would take a while
            population_size: 200,
            ..Default::default()
        };

        // Test that batch processing respects max_time limit
        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .batch_iterations(5)
            .max_time(0.1) // Very short time limit
            .verbose();

        let start_time = std::time::Instant::now();
        let result = oqnlp.run();
        let elapsed = start_time.elapsed().as_secs_f64();

        assert!(result.is_ok(), "OQNLP should run successfully with batch_iterations and max_time");

        // Should stop due to time limit (allowing some overhead)
        assert!(elapsed < 2.0, "Should stop within reasonable time due to max_time limit");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution before timeout");
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test various batch_iterations values
    fn test_various_batch_iterations_values() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams =
            OQNLPParams { iterations: 4, population_size: 8, ..Default::default() };

        // Test batch_iterations = 1
        let mut oqnlp1 = OQNLP::new(problem.clone(), params.clone()).unwrap().batch_iterations(1);
        let result1 = oqnlp1.run();
        assert!(result1.is_ok(), "Should work with batch_iterations = 1");

        // Test batch_iterations = 2
        let mut oqnlp2 = OQNLP::new(problem.clone(), params.clone()).unwrap().batch_iterations(2);
        let result2 = oqnlp2.run();
        assert!(result2.is_ok(), "Should work with batch_iterations = 2");

        // Test batch_iterations larger than iterations
        let mut oqnlp3 = OQNLP::new(problem.clone(), params.clone()).unwrap().batch_iterations(10); // Larger than iterations (4)
        let result3 = oqnlp3.run();
        assert!(result3.is_ok(), "Should work with batch_iterations > iterations");

        // Test without setting batch_iterations (should use default)
        let mut oqnlp4 = OQNLP::new(problem, params).unwrap();
        let result4 = oqnlp4.run();
        assert!(result4.is_ok(), "Should work without setting batch_iterations");
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test that parallel processing actually uses multiple threads
    fn test_parallel_processing_thread_usage() {
        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 16, population_size: 20, ..Default::default() };

        // Set batch size to use available threads
        let num_threads = rayon::current_num_threads();
        let mut oqnlp =
            OQNLP::new(problem, params).unwrap().batch_iterations(num_threads).verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should run successfully with parallel processing");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find solutions with parallel processing");

        // This test mainly verifies that the code compiles and runs without errors
        // when using rayon features. Actual parallel execution verification would
        // require more complex instrumentation.
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch_iterations error handling
    fn test_batch_iterations_error_handling() {
        let problem: DummyProblem = DummyProblem;
        let params: OQNLPParams =
            OQNLPParams { iterations: 5, population_size: 10, ..Default::default() };

        // Test that batch_iterations doesn't interfere with normal error conditions
        let mut oqnlp = OQNLP::new(problem, params).unwrap().batch_iterations(3);

        // The algorithm should still work normally and find solutions
        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should handle batch processing correctly");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find solutions even with batch processing");
    }

    #[test]
    #[cfg(all(feature = "rayon", feature = "checkpointing"))]
    /// Test batch_iterations with checkpointing
    fn test_batch_iterations_with_checkpointing() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_batch_checkpoint");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams { iterations: 8, population_size: 12, ..Default::default() };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_batch".to_string(),
            save_frequency: 2,
            keep_all: false,
            auto_resume: false,
        };

        // Test that batch_iterations works with checkpointing
        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .with_checkpointing(checkpoint_config)
            .unwrap()
            .batch_iterations(4)
            .verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should work with batch_iterations and checkpointing");

        let sol_set = result.unwrap();
        assert!(
            !sol_set.is_empty(),
            "Should find solutions with batch processing and checkpointing"
        );

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "checkpointing")]
    /// Test that create_checkpoint produces consistent checkpoints that can be restored
    fn test_create_checkpoint_consistency() {
        use crate::types::CheckpointConfig;
        use std::env;

        let checkpoint_dir = env::temp_dir().join("globalsearch_test_checkpoint_consistency");
        std::fs::create_dir_all(&checkpoint_dir).expect("Failed to create test directory");

        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 15,
            population_size: 20,
            distance_factor: 0.08,
            threshold_factor: 0.32,
            seed: 12345,
            ..Default::default()
        };

        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            checkpoint_name: "test_consistency".to_string(),
            save_frequency: 1,
            keep_all: false,
            auto_resume: false,
        };

        // Create initial OQNLP instance with comprehensive settings
        let mut oqnlp1 = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint_config.clone())
            .unwrap()
            .target_objective(-0.6)
            .exclude_out_of_bounds()
            .verbose();

        // Add batch_iterations if rayon feature is available
        #[cfg(feature = "rayon")]
        {
            oqnlp1 = oqnlp1.batch_iterations(3);
        }

        // Set up some internal state to create a meaningful checkpoint
        oqnlp1.current_iteration = 7;
        oqnlp1.unchanged_cycles = 2;
        oqnlp1.current_seed = 98765;
        oqnlp1.merit_filter.update_threshold(-0.4);
        oqnlp1.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(25));

        // Create some solutions
        let solution1 = LocalSolution { objective: -0.7, point: Array1::from(vec![0.3, -0.6]) };
        let solution2 = LocalSolution { objective: -0.5, point: Array1::from(vec![-0.8, 0.4]) };

        let solution_set =
            SolutionSet { solutions: Array1::from(vec![solution1.clone(), solution2.clone()]) };
        oqnlp1.solution_set = Some(solution_set.clone());

        // Set reference set
        oqnlp1.current_reference_set = Some(vec![
            Array1::from(vec![-1.5, 1.2]),
            Array1::from(vec![0.8, -0.9]),
            Array1::from(vec![2.1, 0.3]),
        ]);

        // Add solutions to distance filter
        oqnlp1.distance_filter.add_solution(solution1.clone());
        oqnlp1.distance_filter.add_solution(solution2.clone());

        // Create checkpoint from first instance
        let checkpoint = oqnlp1.create_checkpoint();

        // Create second OQNLP instance and restore from checkpoint
        let mut oqnlp2 =
            OQNLP::new(problem, params).unwrap().with_checkpointing(checkpoint_config).unwrap();

        let restore_result = oqnlp2.restore_from_checkpoint(checkpoint);
        assert!(restore_result.is_ok(), "Checkpoint restoration should succeed");

        // Verify all state was consistently restored
        assert_eq!(oqnlp1.params.iterations, oqnlp2.params.iterations, "Iterations should match");
        assert_eq!(
            oqnlp1.params.population_size, oqnlp2.params.population_size,
            "Population size should match"
        );
        assert!(
            (oqnlp1.params.distance_factor - oqnlp2.params.distance_factor).abs() < 1e-10,
            "Distance factor should match"
        );
        assert!(
            (oqnlp1.params.threshold_factor - oqnlp2.params.threshold_factor).abs() < 1e-10,
            "Threshold factor should match"
        );
        assert_eq!(oqnlp1.params.seed, oqnlp2.params.seed, "Seed should match");

        assert_eq!(
            oqnlp1.current_iteration, oqnlp2.current_iteration,
            "Current iteration should match"
        );
        assert!(
            (oqnlp1.merit_filter.threshold - oqnlp2.merit_filter.threshold).abs() < 1e-10,
            "Merit threshold should match"
        );
        assert_eq!(
            oqnlp1.unchanged_cycles, oqnlp2.unchanged_cycles,
            "Unchanged cycles should match"
        );
        assert_eq!(oqnlp1.current_seed, oqnlp2.current_seed, "Current seed should match");
        assert_eq!(
            oqnlp1.target_objective, oqnlp2.target_objective,
            "Target objective should match"
        );
        assert_eq!(
            oqnlp1.exclude_out_of_bounds, oqnlp2.exclude_out_of_bounds,
            "Exclude out of bounds should match"
        );

        // Verify batch_iterations is restored if rayon feature is available
        #[cfg(feature = "rayon")]
        {
            assert_eq!(
                oqnlp1.batch_iterations, oqnlp2.batch_iterations,
                "Batch iterations should match"
            );
        }

        // Verify solution sets match
        let sol_set1 = oqnlp1.solution_set.as_ref().expect("Original should have solution set");
        let sol_set2 = oqnlp2.solution_set.as_ref().expect("Restored should have solution set");
        assert_eq!(sol_set1.len(), sol_set2.len(), "Solution set lengths should match");

        for (s1, s2) in sol_set1.solutions.iter().zip(sol_set2.solutions.iter()) {
            assert!(
                (s1.objective - s2.objective).abs() < 1e-10,
                "Solution objectives should match"
            );
            assert_eq!(s1.point.len(), s2.point.len(), "Solution point dimensions should match");
            for (p1, p2) in s1.point.iter().zip(s2.point.iter()) {
                assert!((p1 - p2).abs() < 1e-10, "Solution point values should match");
            }
        }

        // Verify reference sets match
        let ref_set1 =
            oqnlp1.current_reference_set.as_ref().expect("Original should have reference set");
        let ref_set2 =
            oqnlp2.current_reference_set.as_ref().expect("Restored should have reference set");
        assert_eq!(ref_set1.len(), ref_set2.len(), "Reference set lengths should match");

        for (r1, r2) in ref_set1.iter().zip(ref_set2.iter()) {
            assert_eq!(r1.len(), r2.len(), "Reference point dimensions should match");
            for (p1, p2) in r1.iter().zip(r2.iter()) {
                assert!((p1 - p2).abs() < 1e-10, "Reference point values should match");
            }
        }

        // Verify distance filter solutions match
        let dist_filter1 = oqnlp1.distance_filter.get_solutions();
        let dist_filter2 = oqnlp2.distance_filter.get_solutions();
        assert_eq!(
            dist_filter1.len(),
            dist_filter2.len(),
            "Distance filter solution counts should match"
        );

        for (d1, d2) in dist_filter1.iter().zip(dist_filter2.iter()) {
            assert!(
                (d1.objective - d2.objective).abs() < 1e-10,
                "Distance filter objectives should match"
            );
            assert_eq!(
                d1.point.len(),
                d2.point.len(),
                "Distance filter point dimensions should match"
            );
            for (p1, p2) in d1.point.iter().zip(d2.point.iter()) {
                assert!((p1 - p2).abs() < 1e-10, "Distance filter point values should match");
            }
        }

        // Verify that both instances can continue optimization successfully
        let result1 = oqnlp1.run();
        let result2 = oqnlp2.run();

        assert!(result1.is_ok(), "Original instance should continue successfully");
        assert!(result2.is_ok(), "Restored instance should continue successfully");

        // Clean up test directory
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch iterations edge case: small remaining iterations vs large batch size
    fn test_batch_iterations_edge_case_small_remaining() {
        let problem = DummyProblem;
        let params = OQNLPParams {
            iterations: 5,       // Small total iterations
            population_size: 10, // Must be >= iterations
            ..Default::default()
        };

        // Test with batch size larger than remaining iterations
        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .batch_iterations(10) // Batch size = 10, but only 5 total iterations
            .verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should handle large batch size gracefully");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution");
    }

    #[test]
    #[cfg(feature = "rayon")]
    /// Test batch iterations with exactly matching batch size and iterations
    fn test_batch_iterations_exact_match() {
        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 8, population_size: 12, ..Default::default() };

        // Test with batch size exactly matching iterations
        let mut oqnlp = OQNLP::new(problem, params)
            .unwrap()
            .batch_iterations(8) // Batch size = iterations
            .verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "OQNLP should handle exact match gracefully");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find at least one solution");
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_control() {
        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 5, population_size: 8, ..Default::default() };

        // Test default behavior (parallel enabled)
        let oqnlp1 = OQNLP::new(problem, params.clone()).unwrap();
        assert!(oqnlp1.enable_parallel, "Parallel should be enabled by default");

        // Test disabling parallel
        let mut oqnlp2 = OQNLP::new(DummyProblem, params.clone()).unwrap().parallel(false);
        assert!(!oqnlp2.enable_parallel, "Parallel should be disabled");

        // Test enabling parallel explicitly
        let oqnlp3 = OQNLP::new(DummyProblem, params.clone()).unwrap().parallel(true);
        assert!(oqnlp3.enable_parallel, "Parallel should be enabled");

        // Test that optimization still works with parallel disabled
        let result = oqnlp2.run();
        assert!(result.is_ok(), "OQNLP should work with parallel disabled");
        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find solutions even with parallel disabled");
    }

    #[test]
    /// Test with_points method with valid custom points
    fn test_with_points_valid() {
        let problem = DummyProblem; // Bounds are [-5.0, 5.0] for each dimension (3D)
        let params = OQNLPParams { iterations: 5, population_size: 20, ..Default::default() };

        // Create custom points within bounds
        let custom_points = array![[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0],];

        let result = OQNLP::new(problem, params).unwrap().with_points(custom_points);

        assert!(result.is_ok(), "with_points should succeed with valid points");
        let oqnlp = result.unwrap();
        assert!(oqnlp.custom_points.is_some(), "Custom points should be stored");
        assert_eq!(oqnlp.custom_points.as_ref().unwrap().len(), 3, "Should have 3 custom points");
    }

    #[test]
    /// Test with_points with points outside bounds
    fn test_with_points_out_of_bounds() {
        let problem = DummyProblem; // Bounds are [-5.0, 5.0] for each dimension
        let params = OQNLPParams::default();

        // Create points with one dimension out of bounds
        let custom_points = array![
            [1.0, 2.0, 3.0],  // Valid
            [10.0, 0.0, 0.0], // Invalid - first coordinate exceeds upper bound
        ];

        let result = OQNLP::new(problem, params).unwrap().with_points(custom_points);

        assert!(result.is_err(), "with_points should fail with out-of-bounds points");
        match result {
            Err(OQNLPError::CustomPointOutOfBounds { index }) => {
                assert_eq!(index, 1, "Should report the second point (index 1) as out of bounds");
            }
            _ => panic!("Expected CustomPointOutOfBounds error"),
        }
    }

    #[test]
    /// Test with_points with wrong dimension
    fn test_with_points_wrong_dimension() {
        let problem = DummyProblem; // 3D problem
        let params = OQNLPParams::default();

        // Create points with wrong dimension (2D instead of 3D)
        let custom_points = array![
            [1.0, 2.0],   // Only 2 dimensions, should be 3
        ];

        let result = OQNLP::new(problem, params).unwrap().with_points(custom_points);

        assert!(result.is_err(), "with_points should fail with wrong dimension");
        match result {
            Err(OQNLPError::InvalidCustomPointsDimension { expected, got }) => {
                assert_eq!(expected, 3, "Expected 3 dimensions");
                assert_eq!(got, 2, "Got 2 dimensions");
            }
            _ => panic!("Expected InvalidCustomPointsDimension error"),
        }
    }

    #[test]
    /// Test that custom points are actually used in optimization
    fn test_with_points_integration() {
        let problem = DummyProblem; // Objective is sum of variables, minimum at lower bounds
        let params = OQNLPParams {
            iterations: 10,
            population_size: 20,
            seed: 42, // Fixed seed for reproducibility
            ..Default::default()
        };

        // Create custom points near the actual optimum (lower bounds: [-5, -5, -5])
        let custom_points = array![
            [-4.5, -4.5, -4.5], // Near optimum
            [-4.8, -4.8, -4.8], // Near optimum
        ];

        let mut oqnlp = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_points(custom_points)
            .unwrap()
            .verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "Optimization should succeed with custom points");

        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find solutions");

        // The custom points are near the optimum, so the best solution should be good
        let best = sol_set.best_solution().unwrap();
        assert!(
            best.objective < -10.0,
            "Best objective should be good with custom points near optimum, got {}",
            best.objective
        );
    }

    #[test]
    /// Test with_points with empty array
    fn test_with_points_empty() {
        let problem = DummyProblem;
        let params = OQNLPParams::default();

        // Create empty 2D array (0 rows, 3 columns)
        let custom_points = Array2::<f64>::zeros((0, 3));

        let result = OQNLP::new(problem, params).unwrap().with_points(custom_points);

        assert!(result.is_ok(), "with_points should accept empty array");
        let oqnlp = result.unwrap();
        assert!(oqnlp.custom_points.is_some(), "Custom points should be set");
        assert_eq!(oqnlp.custom_points.as_ref().unwrap().len(), 0, "Should have 0 custom points");
    }

    #[test]
    /// Test with_points with at bounds points
    fn test_with_points_at_bounds() {
        let problem = DummyProblem; // Bounds are [-5.0, 5.0] for each dimension
        let params = OQNLPParams::default();

        // Create points exactly at the bounds
        let custom_points = array![
            [-5.0, -5.0, -5.0], // At lower bound
            [5.0, 5.0, 5.0],    // At upper bound
            [-5.0, 0.0, 5.0],   // Mix of bounds
        ];

        let result = OQNLP::new(problem, params).unwrap().with_points(custom_points);

        assert!(result.is_ok(), "with_points should accept points at bounds");
        let oqnlp = result.unwrap();
        assert_eq!(oqnlp.custom_points.as_ref().unwrap().len(), 3, "Should have 3 custom points");
    }

    #[test]
    /// Test with_points combined with other builder methods
    fn test_with_points_with_other_methods() {
        let problem = DummyProblem;
        let params = OQNLPParams { iterations: 5, population_size: 15, ..Default::default() };

        let custom_points = array![[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0],];

        let result = OQNLP::new(problem, params)
            .unwrap()
            .with_points(custom_points)
            .unwrap()
            .verbose()
            .max_time(60.0)
            .target_objective(0.5)
            .exclude_out_of_bounds();

        assert!(result.custom_points.is_some(), "Custom points should be set");
        assert!(result.verbose, "Verbose should be set");
        assert_eq!(result.max_time, Some(60.0), "Max time should be set");
        assert_eq!(result.target_objective, Some(0.5), "Target objective should be set");
        assert!(result.exclude_out_of_bounds, "Exclude out of bounds should be set");
    }

    #[test]
    /// Test that custom points affect the reference set size
    fn test_with_points_affects_reference_set() {
        let problem = SixHumpCamel;
        let params = OQNLPParams {
            iterations: 5,
            population_size: 15, // Small population to see the effect
            seed: 123,
            ..Default::default()
        };

        // Create several custom points
        let custom_points = array![[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [-0.5, 0.5],];

        let mut oqnlp =
            OQNLP::new(problem, params).unwrap().with_points(custom_points).unwrap().verbose();

        let result = oqnlp.run();
        assert!(result.is_ok(), "Optimization should succeed");

        // The custom points should have been included in the reference set initialization
        // We can't directly verify the reference set, but we can verify the optimization ran successfully
        let sol_set = result.unwrap();
        assert!(!sol_set.is_empty(), "Should find solutions");
    }
}
