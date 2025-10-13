//! # Optimization Problem Trait Module
//!
//! This module defines the [`Problem`] trait, which provides a standardized interface
//! for optimization problems in the globalsearch-rs library. Any optimization problem
//! must implement this trait to be compatible with the OQNLP algorithm.
//!
//! ## Problem Trait Overview
//!
//! The [`Problem`] trait defines the mathematical structure of an optimization problem
//! through several key methods:
//!
//! ### Required Methods
//! - [`objective`](Problem::objective): The objective function to minimize
//! - [`variable_bounds`](Problem::variable_bounds): Box constraints for variables
//!
//! ### Optional Methods (Depending on Local Solver Requirements)
//! - [`gradient`](Problem::gradient): First-order derivatives
//! - [`hessian`](Problem::hessian): Second-order derivatives  
//! - [`constraints`](Problem::constraints): General inequality constraints, only valid with the COBYLA local solver
//!
//! ## Implementation Guidelines
//!
//! ### Objective Function
//! - **Return Type**: `Result<f64, EvaluationError>` for error handling
//! - **Convention**: Lower values indicate better solutions (minimization)
//! - **Error Handling**: Return `EvaluationError` for invalid inputs or computation failures
//!
//! ### Variable Bounds
//! - **Format**: 2D array where each row is `[lower_bound, upper_bound]`
//! - **Requirement**: Must be finite and well-defined
//! - **Purpose**: Defines the feasible region for optimization
//!
//! ### Constraints (Optional, only valid with the COBYLA local solver)
//! - **Sign Convention**:
//!   - `g(x) ≥ 0`: Constraint satisfied
//!   - `g(x) < 0`: Constraint violated
//! - **Return Type**: Vector of constraint function closures
//! - **Use Cases**: Nonlinear inequality constraints beyond simple bounds
//!
//! ## Example: Six-Hump Camel Function
//!
//! ```rust
//! /// References:
//! ///
//! /// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 11-12. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
//!
//! use globalsearch::problem::Problem;
//! use globalsearch::types::EvaluationError;
//! use ndarray::{array, Array1, Array2};
//!
//! #[derive(Debug, Clone)]
//! pub struct SixHumpCamel;
//!
//! impl Problem for SixHumpCamel {
//!     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//!        Ok(
//!              (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
//!                  + x[0] * x[1]
//!                  + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2),
//!          )
//!     }
//!
//!     // Calculated analytically, reference didn't provide gradient
//!     fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
//!         Ok(array![
//!              (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
//!              x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
//!         ])
//!     }
//!
//!     // Calculated analytically, reference didn't provide hessian
//!     fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
//!         Ok(array![
//!             [
//!                 (4.0 * x[0].powi(2) - 4.2) * x[0].powi(2)
//!                     + 4.0 * (4.0 / 3.0 * x[0].powi(3) - 4.2 * x[0]) * x[0]
//!                     + 2.0 * (x[0].powi(4) / 3.0 - 2.1 * x[0].powi(2) + 4.0),
//!                 1.0
//!             ],
//!             [1.0, 40.0 * x[1].powi(2) + 2.0 * (4.0 * x[1].powi(2) - 4.0)],
//!         ])
//!     }
//!
//!     fn variable_bounds(&self) -> Array2<f64> {
//!         array![[-3.0, 3.0], [-2.0, 2.0]]
//!     }
//! }
//! ```

use crate::types::EvaluationError;
use ndarray::{Array1, Array2};

/// # Trait for optimization problems
///
/// This trait defines the methods that an optimization problem must implement, including the objective function, gradient, hessian and variable bounds.
///
/// The objective function is the function to minimize, evaluated at a given point x (`Array1<f64>`).
///
/// The gradient is the derivative of the objective function, evaluated at a given point x (`Array1<f64>`).
///
/// The hessian is the square matrix of the second order partial derivatives of the objective function, evaluated at a given point x (`Array1<f64>`).
///
/// The variable bounds are the lower and upper bounds for the optimization problem.
///
/// Constraint functions for constrained optimization problems can also be defined using the `constraints` method.
///
/// The default implementation of the gradient and hessian returns an error indicating the gradient and hessian are not implemented.
/// Some local solvers require the gradient and hessian to be implemented, while for others it isn't needed.
/// You should check the documentation of the local solver you are using to know if the gradient and hessian are needed.
pub trait Problem {
    /// Objective function to minimize, given at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<f64, EvaluationError>` of the value of the objective function at x
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError>;

    /// Gradient of the objective function at point x (`Array1<f64>`)
    ///
    /// Returns a `Result<Array1<f64>, EvaluationError>` of the gradient of the objective function at x
    ///
    /// The default implementation returns an error indicating the gradient is not implemented
    /// in case it is needed
    fn gradient(&self, _x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Err(EvaluationError::GradientNotImplemented)
    }

    /// Returns the Hessian at point x (`Array1<f64>`).
    ///
    /// Returns a `Result<Array2<f64>, EvaluationError>` of the hessian of the objective function at x
    ///
    /// The default implementation returns an error indicating the hessian is not implemented
    /// in case it is needed
    fn hessian(&self, _x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Err(EvaluationError::HessianNotImplemented)
    }

    /// Variable bounds for the optimization problem
    ///
    /// Returns a `Result<Array2<f64>>` of the variable bounds for the optimization problem.
    ///
    /// This bounds are only used in the scatter search phase of the algorithm.
    /// The local solver is unconstrained (See [argmin issue #137](https://github.com/argmin-rs/argmin/issues/137)) and therefor can return solutions out of the bounds.
    /// You may be able to guide your solutions to your desired bounds/constraints by using a penalty method.
    fn variable_bounds(&self) -> Array2<f64>;

    /// Constraint functions for constrained optimization
    ///
    /// Returns constraint functions in the format expected by optimization solvers.
    /// Function pointers that take (&[f64], &mut ()) and return f64.
    ///
    /// **Sign Convention**:
    /// - **Positive or zero**: constraint satisfied  
    /// - **Negative**: constraint violated
    ///
    /// The default implementation returns an empty vector (no constraints).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use globalsearch::problem::Problem;
    /// use globalsearch::types::EvaluationError;
    /// use ndarray::Array1;
    ///
    /// struct MyProblem;
    /// impl Problem for MyProblem {
    ///     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
    ///         Ok(x[0].powi(2) + x[1].powi(2))
    ///     }
    ///
    ///     fn variable_bounds(&self) -> ndarray::Array2<f64> {
    ///         ndarray::array![[-1.0, 1.0], [-1.0, 1.0]]
    ///     }
    ///
    ///     fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
    ///         vec![
    ///             |x: &[f64], _: &mut ()| 1.0 - x[0] - x[1], // x[0] + x[1] <= 1.0 -> 1.0 - x[0] - x[1] >= 0
    ///         ]
    ///     }
    /// }
    /// ```
    fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
        vec![]
    }
}
