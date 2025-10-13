//! # Types Module
//!
//! This module defines the core data structures and types used throughout the OQNLP
//! (OptQuest for Nonlinear Programming) global optimization algorithm.
//!
//! ## Core Types
//!
//! - [`OQNLPParams`] - Configuration parameters for the optimization algorithm
//! - [`LocalSolution`] - Represents a single solution point with its objective value
//! - [`SolutionSet`] - Collection of local solutions with utility methods
//! - [`FilterParams`] - Parameters for filtering mechanisms (distance and merit filters)
//! - [`LocalSolverType`] - Enumeration of available local optimization solvers
//! - [`LocalSolverConfig`] - Configuration options for local solvers
//!
//! ## Example
//!
//! ```rust
//! use globalsearch::types::{OQNLPParams, LocalSolverType};
//! use globalsearch::local_solver::builders::COBYLABuilder;
//!
//! // Create optimization parameters
//! let params = OQNLPParams {
//!     iterations: 500,
//!     population_size: 1000,
//!     local_solver_type: LocalSolverType::COBYLA,
//!     local_solver_config: COBYLABuilder::default().build(),
//!     ..OQNLPParams::default()
//! };
//! ```

use crate::local_solver::builders::{COBYLABuilder, LocalSolverConfig};
use crate::problem::Problem;
use ndarray::Array1;
use std::fmt;
use std::ops::Index;
use thiserror::Error;

#[cfg(feature = "checkpointing")]
use std::path::PathBuf;

// TODO: Implement SR1 when it is fixed in argmin (https://github.com/argmin-rs/argmin/issues/221)
// Or add it now and print a warning that it is not working as expected in some cases

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Configuration parameters for the OQNLP (OptQuest for Nonlinear Programming) algorithm.
///
/// This struct contains all the tunable parameters that control the behavior of the
/// global optimization process. The algorithm combines scatter search with local
/// optimization methods to find global optima.
///
/// # Parameter Guidelines
///
/// - **iterations**: Should be set based on problem complexity and available computation time
/// - **population_size**: Larger values improve exploration but increase computation time
/// - **wait_cycle**: Controls balance between exploration and exploitation
/// - **threshold_factor**: Higher values make merit filter more permissive
/// - **distance_factor**: Higher values enforce more diversity between solutions
pub struct OQNLPParams {
    /// Total number of iterations for the optimization process
    pub iterations: usize,

    /// Number of population size
    ///
    /// Population size is the number of points in the reference set.
    /// The reference set is created in Stage 1, where we optimize the best objective
    /// function value found so far.
    ///
    /// In stage 2, we optimize random `iterations` points of the reference set.
    pub population_size: usize,

    /// Number of iterations to wait before updating the threshold criteria and reference set
    ///
    /// This is used to determine the number of iterations to wait before updating the
    /// threshold criteria (Stage 2) and the reference set (Stage 1).
    pub wait_cycle: usize,

    /// Threshold factor multiplier
    ///
    /// The new threshold is calculated as `threshold = threshold + threshold_factor * (1 + abs(threshold))`
    pub threshold_factor: f64,

    /// Factor that influences the minimum required distance between candidate solutions
    pub distance_factor: f64,

    /// Type of local solver to use from argmin
    pub local_solver_type: LocalSolverType,

    /// Configuration for the local solver
    pub local_solver_config: LocalSolverConfig,

    /// Random seed for the algorithm
    pub seed: u64,
}

impl Default for OQNLPParams {
    /// Default parameters for the OQNLP algorithm
    ///
    /// It is highly recommended to change these parameters based on the problem at hand.
    ///
    /// The default parameters are:
    /// - `iterations`: 300
    /// - `population_size`: 1000
    /// - `wait_cycle`: 15
    /// - `threshold_factor`: 0.2
    /// - `distance_factor`: 0.75
    /// - `local_solver_type`: `LocalSolverType::COBYLA`
    /// - `local_solver_config`: `COBYLABuilder::default().build()`
    /// - `seed`: 0
    fn default() -> Self {
        Self {
            iterations: 300,
            population_size: 1000,
            wait_cycle: 15,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            local_solver_type: LocalSolverType::COBYLA,
            local_solver_config: COBYLABuilder::default().build(),
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Configuration parameters for filtering mechanisms in the OQNLP algorithm.
///
/// These parameters control how candidate solutions are filtered during the
/// optimization process to maintain solution diversity and quality.
///
/// # Filtering Mechanisms
///
/// ## Distance Filter
/// Uses `distance_factor` to enforce minimum separation between solutions,
/// preventing clustering and maintaining population diversity.
///
/// ## Merit Filter  
/// Uses `threshold_factor` and `wait_cycle` to dynamically adjust acceptance
/// criteria based on solution quality.
///
/// # Parameter Details
///
/// - **distance_factor**: Controls minimum Euclidean distance between solutions
///   - Range: [0.0, ∞)
///   - Higher values → more diverse solutions
///   - Lower values → solutions can be closer together
///
/// - **wait_cycle**: Number of iterations between filter parameter updates
///   - Typical range: [5, 50]
///   - Higher values → more stable filtering
///   - Lower values → more adaptive filtering
///
/// - **threshold_factor**: Controls merit filter sensitivity
///   - Range: (0.0, 1.0]
///   - Higher values → more permissive acceptance
///   - Lower values → stricter solution quality requirements
pub struct FilterParams {
    /// Factor that influences the minimum required distance between candidate solutions
    ///
    /// The distance factor is used to determine the minimum required distance between candidate solutions.
    /// If the distance between two solutions is less than the distance factor, one of the solutions is removed.
    ///
    /// The distance factor is used in the `DistanceFilter` mechanism and it is a positive value or zero.
    pub distance_factor: f64,
    /// Number of iterations to wait before updating the threshold criteria
    pub wait_cycle: usize,
    /// Threshold factor
    pub threshold_factor: f64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Represents a single solution point in the optimization parameter space.
///
/// A `LocalSolution` encapsulates both the parameter values (point) and the
/// corresponding objective function value. This is the fundamental unit of
/// solution information in the optimization algorithm.
///
/// # Fields
///
/// - `point`: The solution coordinates in the parameter space as an N-dimensional array
/// - `objective`: The objective function value at this point (lower is better for minimization)
///
/// # Methods
///
/// The struct provides convenience methods compatible with scipy.optimize conventions:
/// - [`fun()`](LocalSolution::fun) - Returns the objective value (alias for `objective`)
/// - [`x()`](LocalSolution::x) - Returns a clone of the parameter vector (alias for `point`)
///
/// # Example
///
/// ```rust
/// use globalsearch::types::LocalSolution;
/// use ndarray::array;
///
/// let solution = LocalSolution {
///     point: array![1.0, 2.0, 3.0],
///     objective: -5.2,
/// };
///
/// // Access using convenience methods
/// assert_eq!(solution.fun(), -5.2);
/// assert_eq!(solution.x(), array![1.0, 2.0, 3.0]);
/// ```
pub struct LocalSolution {
    /// The solution point in the parameter space
    pub point: Array1<f64>,
    /// The objective function value at the solution point
    pub objective: f64,
}

impl LocalSolution {
    /// Returns the objective function value (f64) at the solution point
    ///
    /// Same as `objective` field
    ///
    /// This method is similar to the `fun` method in `SciPy.optimize` result
    pub fn fun(&self) -> f64 {
        self.objective
    }

    /// Returns the solution point (`Array1<f64>`) in the parameter space
    ///
    /// Same as `point` field
    /// Returns a clone of the point to avoid moving it
    ///
    /// This method is similar to the `x` method in `SciPy.optimize` result
    pub fn x(&self) -> Array1<f64> {
        self.point.clone()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// A collection of local solutions with utility methods for analysis and display.
///
/// `SolutionSet` is the primary data structure for storing and manipulating
/// optimization results. It provides methods for accessing, analyzing, and
/// displaying multiple solutions found during the optimization process.
///
/// # Key Features
///
/// - **Indexing**: Direct access to individual solutions via `solution_set[index]`
/// - **Best solution**: Automatic identification of the best (lowest objective) solution
/// - **Iteration**: Support for iterating over all solutions
/// - **Display**: Pretty-printing with constraint violation information when applicable
///
/// # Storage
///
/// Solutions are stored internally as an `Array1<LocalSolution>` for efficient
/// vectorized operations and memory layout.
///
/// # Example
///
/// ```rust
/// use globalsearch::types::{LocalSolution, SolutionSet};
/// use ndarray::{array, Array1};
///
/// let solutions = Array1::from_vec(vec![
///     LocalSolution { point: array![1.0, 2.0], objective: 5.0 },
///     LocalSolution { point: array![2.0, 1.0], objective: 3.0 },
///     LocalSolution { point: array![0.0, 0.0], objective: 1.0 },
/// ]);
///
/// let solution_set = SolutionSet { solutions };
///
/// // Access the best solution
/// let best = solution_set.best_solution().unwrap();
/// assert_eq!(best.objective, 1.0);
///
/// // Check size and iterate
/// assert_eq!(solution_set.len(), 3);
/// for solution in solution_set.solutions() {
///     println!("Point: {:?}, Objective: {}", solution.point, solution.objective);
/// }
/// ```
pub struct SolutionSet {
    pub solutions: Array1<LocalSolution>,
}

impl SolutionSet {
    /// Returns the number of solutions stored in the set.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns true if the solution set contains no solutions.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Returns the best solution in the set based on the objective function value.
    pub fn best_solution(&self) -> Option<&LocalSolution> {
        self.solutions.iter().min_by(|a, b| a.objective.partial_cmp(&b.objective).unwrap())
    }

    /// Returns an iterator over the solutions in the set.
    pub fn solutions(&self) -> impl Iterator<Item = &LocalSolution> {
        self.solutions.iter()
    }

    /// Display solution set with constraint violations for problems that have constraints.
    ///
    /// This method formats the solution set similarly to the Display trait but includes
    /// constraint violation information when the problem has constraints defined.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem that was solved, used to evaluate constraints
    /// * `constraint_descriptions` - Optional descriptions for each constraint (e.g., "x + y <= 1.5")
    ///
    /// # Returns
    ///
    /// A formatted string showing solutions with constraint violations
    pub fn display_with_constraints<P: Problem>(
        &self,
        problem: &P,
        constraint_descriptions: Option<&[&str]>,
    ) -> String {
        let mut result = String::new();
        let constraints = problem.constraints();

        result.push_str("━━━━━━━━━━━ Solution Set ━━━━━━━━━━━\n");
        result.push_str(&format!("Total solutions: {}\n", self.solutions.len()));

        if !self.solutions.is_empty() {
            if let Some(best) = self.best_solution() {
                result.push_str(&format!("Best objective value: {:.8e}\n", best.objective));
            }
        }
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (i, solution) in self.solutions.iter().enumerate() {
            result.push_str(&format!("Solution #{}\n", i + 1));
            result.push_str(&format!("  Objective: {:.8e}\n", solution.objective));
            result.push_str("  Parameters:\n");
            result.push_str(&format!("    {:.8e}\n", solution.point));

            // Add constraint violations if constraints exist
            if !constraints.is_empty() {
                result.push_str("  Constraint violations:\n");
                for (j, constraint_fn) in constraints.iter().enumerate() {
                    let x_slice: Vec<f64> = solution.point.to_vec();
                    let constraint_value = constraint_fn(&x_slice, &mut ());

                    // Format constraint status
                    let status = if constraint_value >= 0.0 { "✓" } else { "✗" };
                    let violation = if constraint_value < 0.0 {
                        format!(" (violated by {:.6})", -constraint_value)
                    } else {
                        " (satisfied)".to_string()
                    };

                    // Add constraint description if provided
                    let description = if let Some(descriptions) = constraint_descriptions {
                        if j < descriptions.len() {
                            format!(" [{}]", descriptions[j])
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    };

                    result.push_str(&format!(
                        "    Constraint {}{}: {} {:.6e}{}\n",
                        j + 1,
                        description,
                        status,
                        constraint_value,
                        violation
                    ));
                }
            }

            if i < self.solutions.len() - 1 {
                result.push_str("――――――――――――――――――――――――――――――――――――\n");
            }
        }

        result
    }

    /// Display solution set with constraint violations if the problem has constraints.
    ///
    /// This is a convenience method that automatically detects if the problem has constraints
    /// and displays them if present, otherwise displays normally.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem that was solved, used to evaluate constraints
    ///
    /// # Returns
    ///
    /// A formatted string showing solutions with or without constraint violations
    pub fn display_with_problem<P: Problem>(&self, problem: &P) -> String {
        let constraints = problem.constraints();
        if constraints.is_empty() {
            // No constraints, use regular display
            format!("{}", self)
        } else {
            // Has constraints, use constraint-aware display
            self.display_with_constraints(problem, None)
        }
    }
}

impl Index<usize> for SolutionSet {
    type Output = LocalSolution;

    /// Returns the solution at the given index.
    fn index(&self, index: usize) -> &Self::Output {
        &self.solutions[index]
    }
}

impl fmt::Display for SolutionSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len: usize = self.solutions.len();
        writeln!(f, "━━━━━━━━━━━ Solution Set ━━━━━━━━━━━")?;
        writeln!(f, "Total solutions: {}", self.solutions.len())?;
        if len > 0 {
            if let Some(best) = self.best_solution() {
                writeln!(f, "Best objective value: {:.8e}", best.objective)?;
            }
        }
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;

        for (i, solution) in self.solutions.iter().enumerate() {
            writeln!(f, "Solution #{}", i + 1)?;
            writeln!(f, "  Objective: {:.8e}", solution.objective)?;
            writeln!(f, "  Parameters:")?;
            writeln!(f, "    {:.8e}", solution.point)?;

            if i < self.solutions.len() - 1 {
                writeln!(f, "――――――――――――――――――――――――――――――――――――")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Local solver implementation types for the OQNLP algorithm
///
/// This enum defines the types of local solvers that can be used in the OQNLP algorithm, including L-BFGS, Nelder-Mead, and Gradient Descent (argmin's implementations).
pub enum LocalSolverType {
    /// L-BFGS local solver
    ///
    /// Requires `CostFunction` and `Gradient`
    LBFGS,

    /// Nelder-Mead local solver
    ///
    /// Requires `CostFunction`
    NelderMead,

    /// Steepest Descent local solver
    ///
    /// Requires `CostFunction` and `Gradient`
    SteepestDescent,

    /// Trust Region local solver
    ///
    /// Requires `CostFunction`, `Gradient` and `Hessian`
    TrustRegion,

    /// Newton-Conjugate-Gradient method local solver
    ///
    /// Requires `CostFunction`, `Gradient` and `Hessian`
    NewtonCG,

    /// COBYLA (Constrained Optimization BY Linear Approximations) local solver
    ///
    /// Requires only `CostFunction`
    COBYLA,
}

impl LocalSolverType {
    /// Returns the local solver type from a string
    ///
    /// This method is used to convert a string to a `LocalSolverType` enum.
    /// It is used to set the local solver type for the Python bindings.
    pub fn from_string(s: &str) -> Result<Self, &'static str> {
        match s.to_lowercase().as_str() {
            "lbfgs" => Ok(Self::LBFGS),
            "nelder-mead" => Ok(Self::NelderMead),
            "neldermead" => Ok(Self::NelderMead),
            "steepestdescent" => Ok(Self::SteepestDescent),
            "trustregion" => Ok(Self::TrustRegion),
            "newton-cg" => Ok(Self::NewtonCG),
            "newtoncg" => Ok(Self::NewtonCG),
            "cobyla" => Ok(Self::COBYLA),
            _ => Err("Invalid solver type."),
        }
    }
}

#[derive(Debug, Error)]
/// Error type for function, gradient and hessian evaluation
pub enum EvaluationError {
    /// Error when the input is invalid
    #[error("Invalid input: {0}.")]
    InvalidInput(String),

    /// Error when dividing by zero
    #[error("Division by zero found.")]
    DivisionByZero,

    /// Error when having a negative square root
    #[error("Negative square root found.")]
    NegativeSqrt,

    /// Error when the objective function is not implemented
    #[error("Objective function not implemented and needed for local solver.")]
    ObjectiveFunctionNotImplemented,

    /// Error when the gradient is not implemented
    #[error("Gradient not implemented and needed for local solver.")]
    GradientNotImplemented,

    /// Error when the hessian is not implemented
    #[error("Hessian not implemented and needed for local solver.")]
    HessianNotImplemented,

    /// Error when the objective function can't be evaluated
    #[error("Objective function evaluation failed.")]
    ObjectiveFunctionEvaluationFailed,

    /// Error when the gradient can't be evaluated
    #[error("Gradient evaluation failed.")]
    GradientEvaluationFailed,

    /// Error when the hessian can't be evaluated
    #[error("Hessian evaluation failed.")]
    HessianEvaluationFailed,

    /// Error when constraints are not implemented
    #[error("Constraints not implemented and needed for constrained solver.")]
    ConstraintNotImplemented,

    /// Error when an invalid constraint index is provided
    #[error("Invalid constraint index provided.")]
    InvalidConstraintIndex,

    /// Error when constraint evaluation fails
    #[error("Constraint evaluation failed.")]
    ConstraintEvaluationFailed,
}

#[cfg(feature = "checkpointing")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Configuration for checkpointing behavior
///
/// This struct defines the configuration for checkpointing behavior in the OQNLP algorithm.
/// It includes the directory where checkpoint files will be saved, the base name for checkpoint files,
/// the frequency of checkpointing, whether to keep all checkpoints or only the latest one,
/// and whether to automatically resume from the latest checkpoint if found.
pub struct CheckpointConfig {
    /// Directory where checkpoint files will be saved
    pub checkpoint_dir: PathBuf,

    /// Base name for checkpoint files (without extension)
    pub checkpoint_name: String,

    /// Frequency of checkpointing (every N iterations)
    pub save_frequency: usize,

    /// Whether to keep all checkpoints or only the latest one
    pub keep_all: bool,

    /// Whether to automatically resume from the latest checkpoint if found
    pub auto_resume: bool,
}

#[cfg(feature = "checkpointing")]
impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("./checkpoints"),
            checkpoint_name: "oqnlp_checkpoint".to_string(),
            save_frequency: 25,
            keep_all: false,
            auto_resume: true,
        }
    }
}

#[cfg(feature = "checkpointing")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
/// Complete OQNLP state that can be saved and restored
///
/// This struct represents the complete state of the OQNLP algorithm at a given point in time,
/// including the algorithm parameters, current iteration, merit threshold, solution set,
/// current reference set from scatter search, number of unchanged cycles,
/// elapsed time, distance filter solutions for maintaining diversity, current seed value,
/// timestamp of the checkpoint, and other relevant information.
/// It is used to save the state of the algorithm to a file and restore it later.
pub struct OQNLPCheckpoint {
    /// Algorithm parameters
    pub params: OQNLPParams,

    /// Current iteration number
    pub current_iteration: usize,

    /// Current threshold value for merit filter
    pub merit_threshold: f64,

    /// Current solution set (if any)
    pub solution_set: Option<SolutionSet>,

    /// Current reference set from scatter search
    #[cfg_attr(
        feature = "checkpointing",
        serde(
            serialize_with = "serialize_vec_array1",
            deserialize_with = "deserialize_vec_array1"
        )
    )]
    pub reference_set: Vec<Array1<f64>>,

    /// Number of unchanged cycles
    pub unchanged_cycles: usize,

    /// Elapsed time in seconds
    pub elapsed_time: f64,

    /// Distance filter solutions for maintaining diversity
    pub distance_filter_solutions: Vec<LocalSolution>,

    /// Current seed value for continuing RNG sequence
    pub current_seed: u64,

    /// Target objective function value to stop optimization
    pub target_objective: Option<f64>,

    /// Whether to exclude out-of-bounds solutions from being considered valid
    pub exclude_out_of_bounds: bool,

    /// Batch size for parallel processing in Stage 2 (only available with rayon feature)
    #[cfg(feature = "rayon")]
    pub batch_iterations: Option<usize>,

    /// Whether parallel processing is enabled when Rayon is available
    #[cfg(feature = "rayon")]
    pub enable_parallel: bool,

    /// Timestamp of the checkpoint
    pub timestamp: String,
}

#[cfg(feature = "checkpointing")]
/// Serializes a vector of `Array1<f64>` to a format suitable for serialization
fn serialize_vec_array1<S>(vec: &Vec<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(vec.len()))?;
    for array in vec {
        let vec_data: Vec<f64> = array.to_vec();
        seq.serialize_element(&vec_data)?;
    }
    seq.end()
}

#[cfg(feature = "checkpointing")]
/// Deserializes a vector of `Array1<f64>` from a format suitable for serialization
fn deserialize_vec_array1<'de, D>(deserializer: D) -> Result<Vec<Array1<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let vec_of_vecs: Vec<Vec<f64>> = Vec::deserialize(deserializer)?;
    Ok(vec_of_vecs.into_iter().map(Array1::from).collect())
}

#[cfg(feature = "checkpointing")]
/// Implements the Display trait for OQNLPCheckpoint
///
/// This trait provides a formatted string representation of the OQNLP checkpoint,
/// including the timestamp, current iteration, elapsed time, unchanged cycles,
/// merit threshold, reference set size, distance filter solutions,
/// current seed value, and parameters.
impl fmt::Display for OQNLPCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "━━━━━━━━━━━ OQNLP Checkpoint ━━━━━━━━━━━")?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f, "Current iteration: {}", self.current_iteration)?;
        writeln!(f, "Elapsed time: {:.2}s", self.elapsed_time)?;
        writeln!(f, "Unchanged cycles: {}", self.unchanged_cycles)?;
        writeln!(f, "Merit threshold: {:.8e}", self.merit_threshold)?;
        writeln!(f, "Reference set size: {}", self.reference_set.len())?;
        writeln!(f, "Distance filter solutions: {}", self.distance_filter_solutions.len())?;
        writeln!(f, "Current seed: {}", self.current_seed)?;

        if let Some(ref solution_set) = self.solution_set {
            writeln!(f, "Solution set: {} solutions", solution_set.len())?;
            if let Some(best) = solution_set.best_solution() {
                writeln!(f, "Best objective: {:.8e}", best.objective)?;
            }
        } else {
            writeln!(f, "Solution set: None")?;
        }

        writeln!(f, "Parameters:")?;
        writeln!(f, "  Population size: {}", self.params.population_size)?;
        writeln!(f, "  Iterations: {}", self.params.iterations)?;
        writeln!(f, "  Wait cycle: {}", self.params.wait_cycle)?;
        writeln!(f, "  Threshold factor: {}", self.params.threshold_factor)?;
        writeln!(f, "  Distance factor: {}", self.params.distance_factor)?;
        writeln!(f, "  Local solver: {:?}", self.params.local_solver_type)?;
        writeln!(f, "  Seed: {}", self.params.seed)?;

        if let Some(target) = self.target_objective {
            writeln!(f, "  Target objective: {:.8e}", target)?;
        } else {
            writeln!(f, "  Target objective: None")?;
        }

        writeln!(f, "  Exclude out of bounds: {}", self.exclude_out_of_bounds)?;

        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests_types {
    use super::*;
    use ndarray::array;

    #[test]
    /// Test the default parameters for the OQNLP algorithm
    fn test_oqnlp_params_default() {
        let params = OQNLPParams::default();
        assert_eq!(params.iterations, 300);
        assert_eq!(params.population_size, 1000);
        assert_eq!(params.wait_cycle, 15);
        assert_eq!(params.threshold_factor, 0.2);
        assert_eq!(params.distance_factor, 0.75);
        assert_eq!(params.seed, 0);
    }

    #[test]
    /// Test the len method for the SolutionSet struct
    fn test_solution_set_len() {
        let solutions = Array1::from_vec(vec![
            LocalSolution { point: array![1.0, 2.0], objective: -1.0 },
            LocalSolution { point: array![3.0, 4.0], objective: -2.0 },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert_eq!(solution_set.len(), 2);
    }

    #[test]
    /// Test the is_empty method for the SolutionSet struct
    fn test_solution_set_is_empty() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert!(solution_set.is_empty());

        let solutions: Array1<LocalSolution> =
            Array1::from_vec(vec![LocalSolution { point: array![1.0], objective: -1.0 }]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        assert!(!solution_set.is_empty());
    }

    #[test]
    /// Test indexing into the SolutionSet struct
    fn test_solution_set_index() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![
            LocalSolution { point: array![1.0, 2.0], objective: -1.0 },
            LocalSolution { point: array![3.0, 4.0], objective: -2.0 },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        assert_eq!(solution_set[0].objective, -1.0);
        assert_eq!(solution_set[1].objective, -2.0);
    }

    #[test]
    /// Test the Display trait for the SolutionSet struct
    fn test_solution_set_display() {
        let solutions: Array1<LocalSolution> =
            Array1::from_vec(vec![LocalSolution { point: array![1.0], objective: -1.0 }]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        println!("{}", solution_set);

        let display_output: String = format!("{}", solution_set);
        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 1"));
        assert!(display_output.contains("Best objective value"));
        assert!(display_output.contains("Solution #1"));
    }

    #[test]
    /// Test the display of empty solution set
    fn test_empty_solution_set_display() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        let display_output: String = format!("{}", solution_set);
        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 0"));
    }

    #[test]
    #[should_panic]
    fn test_solution_set_index_out_of_bounds() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![]);
        let solution_set: SolutionSet = SolutionSet { solutions };
        let _should_panic: LocalSolution = solution_set[0].clone();
    }

    #[test]
    /// Test the from_string method for the LocalSolverType enum
    fn test_local_solver_type_from_string() {
        assert_eq!(LocalSolverType::from_string("LBFGS"), Ok(LocalSolverType::LBFGS));
        assert_eq!(LocalSolverType::from_string("Nelder-Mead"), Ok(LocalSolverType::NelderMead));
        assert_eq!(
            LocalSolverType::from_string("SteepestDescent"),
            Ok(LocalSolverType::SteepestDescent)
        );
        assert_eq!(LocalSolverType::from_string("TrustRegion"), Ok(LocalSolverType::TrustRegion));
        assert_eq!(LocalSolverType::from_string("NewtonCG"), Ok(LocalSolverType::NewtonCG));
        assert_eq!(LocalSolverType::from_string("Invalid"), Err("Invalid solver type."));
    }

    #[test]
    /// Test f() and x() methods from LocalSolution
    fn test_local_solution_f_x() {
        let local_solution = LocalSolution { point: array![1.0], objective: -1.0 };

        assert_eq!(local_solution.fun(), -1.0);
        assert_eq!(local_solution.x(), array![1.0]);
    }

    #[test]
    /// Test best_solution from SolutionSet
    fn test_solution_set_best_solution() {
        let solutions: Array1<LocalSolution> = Array1::from_vec(vec![
            LocalSolution { point: array![1.0], objective: -1.0 },
            LocalSolution { point: array![2.0], objective: -1.0 },
            LocalSolution { point: array![3.0], objective: -1.0 },
        ]);
        let solution_set: SolutionSet = SolutionSet { solutions };

        let best_solution = solution_set.best_solution().unwrap();
        assert_eq!(best_solution.objective, -1.0);
    }

    #[cfg(feature = "checkpointing")]
    #[test]
    /// Test the Display trait for OQNLPCheckpoint
    fn test_oqnlp_checkpoint_display() {
        let solution_set = SolutionSet {
            solutions: Array1::from_vec(vec![
                LocalSolution { point: array![1.0, 2.0], objective: -1.5 },
                LocalSolution { point: array![3.0, 4.0], objective: -2.0 },
            ]),
        };

        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams {
                iterations: 100,
                population_size: 500,
                wait_cycle: 20,
                threshold_factor: 0.3,
                distance_factor: 0.8,
                seed: 42,
                local_solver_type: LocalSolverType::COBYLA,
                local_solver_config: crate::local_solver::builders::COBYLABuilder::default()
                    .build(),
            },
            current_iteration: 50,
            merit_threshold: 1.25,
            solution_set: Some(solution_set),
            reference_set: vec![array![1.0, 2.0], array![3.0, 4.0]],
            unchanged_cycles: 5,
            elapsed_time: 123.45,
            distance_filter_solutions: vec![],
            current_seed: 42,
            target_objective: Some(-1.5),
            exclude_out_of_bounds: true,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(4),
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            timestamp: "2025-07-27T12:00:00Z".to_string(),
        };

        let display_output = format!("{}", checkpoint);

        // Check that display output contains expected elements
        assert!(display_output.contains("OQNLP Checkpoint"));
        assert!(display_output.contains("2025-07-27T12:00:00Z"));
        assert!(display_output.contains("Current iteration: 50"));
        assert!(display_output.contains("Merit threshold: 1.25"));
        assert!(display_output.contains("Solution set: 2 solutions"));
        assert!(display_output.contains("Best objective: -2"));
        assert!(display_output.contains("Reference set size: 2"));
        assert!(display_output.contains("Unchanged cycles: 5"));
        assert!(display_output.contains("Elapsed time: 123.45s"));
        assert!(display_output.contains("Population size: 500"));
        assert!(display_output.contains("Wait cycle: 20"));
        assert!(display_output.contains("Local solver: COBYLA"));
    }

    #[test]
    /// Test constraint-aware display with constraints
    fn test_solution_set_display_with_constraints() {
        use crate::problem::Problem;
        use crate::types::EvaluationError;

        // Create a mock problem with constraints
        #[derive(Debug, Clone)]
        struct TestProblemWithConstraints;

        impl Problem for TestProblemWithConstraints {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x[0].powi(2) + x[1].powi(2))
            }

            fn variable_bounds(&self) -> ndarray::Array2<f64> {
                ndarray::array![[-2.0, 2.0], [-2.0, 2.0]]
            }

            fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
                vec![
                    |x: &[f64], _: &mut ()| 1.0 - x[0] - x[1], // x[0] + x[1] <= 1.0
                ]
            }
        }

        let solutions =
            Array1::from_vec(vec![LocalSolution { point: array![0.3, 0.3], objective: 0.18 }]);
        let solution_set = SolutionSet { solutions };
        let problem = TestProblemWithConstraints;

        let constraint_descriptions = ["x[0] + x[1] <= 1.0"];
        let display_output =
            solution_set.display_with_constraints(&problem, Some(&constraint_descriptions));

        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 1"));
        assert!(display_output.contains("Constraint violations:"));
        assert!(display_output.contains("Constraint 1 [x[0] + x[1] <= 1.0]"));
        assert!(display_output.contains("✓")); // Should be satisfied
        assert!(display_output.contains("(satisfied)"));
    }

    #[test]
    /// Test constraint-aware display without constraints
    fn test_solution_set_display_with_problem_no_constraints() {
        use crate::problem::Problem;
        use crate::types::EvaluationError;

        // Create a mock problem without constraints
        #[derive(Debug, Clone)]
        struct TestProblemNoConstraints;

        impl Problem for TestProblemNoConstraints {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x[0].powi(2) + x[1].powi(2))
            }

            fn variable_bounds(&self) -> ndarray::Array2<f64> {
                ndarray::array![[-2.0, 2.0], [-2.0, 2.0]]
            }
        }

        let solutions =
            Array1::from_vec(vec![LocalSolution { point: array![1.0, 1.0], objective: 2.0 }]);
        let solution_set = SolutionSet { solutions };
        let problem = TestProblemNoConstraints;

        let display_output = solution_set.display_with_problem(&problem);

        assert!(display_output.contains("Solution Set"));
        assert!(display_output.contains("Total solutions: 1"));
        assert!(!display_output.contains("Constraint violations:")); // Should not have constraints
    }

    #[cfg(feature = "checkpointing")]
    #[test]
    /// Test the Display trait for OQNLPCheckpoint with no solutions
    fn test_oqnlp_checkpoint_display_no_solutions() {
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 10,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 15.5,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-07-27T10:00:00Z".to_string(),
        };

        let display_output = format!("{}", checkpoint);

        // Check that display output handles no solutions case
        assert!(display_output.contains("OQNLP Checkpoint"));
        assert!(display_output.contains("2025-07-27T10:00:00Z"));
        assert!(display_output.contains("Current iteration: 10"));
        assert!(display_output.contains("Solution set: None"));
        assert!(display_output.contains("Reference set size: 0"));
    }
}
