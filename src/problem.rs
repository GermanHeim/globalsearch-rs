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
//! - [`constraints`](Problem::constraints): Nonlinear inequalities (`g(x) >= 0`), COBYLA and SLSQP
//! - [`nonlinear_equalities`](Problem::nonlinear_equalities): Nonlinear equalities, SLSQP and COBYLA
//! - [`linear_inequalities`](Problem::linear_inequalities): `A x <= b`, SLSQP, Barrier, COBYLA
//! - [`linear_equalities`](Problem::linear_equalities): `A x = b`, SLSQP, AugmentedLagrangian, COBYLA
//! - [`constraint_jacobian`](Problem::constraint_jacobian): Nonlinear constraint Jacobian, SLSQP
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
//! ### Constraints (Optional, solver-dependent)
//! - **Sign Convention**:
//!   - `g(x) ≥ 0`: Constraint satisfied
//!   - `g(x) < 0`: Constraint violated
//! - **Return Type**: A vector containing all constraint values at the given point
//! - **Requirement**: The number and order of constraint values must remain stable
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
use std::sync::OnceLock;

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
    /// COBYLA and the L-BFGS-B, bounded Nelder-Mead, and BOBYQA solvers
    /// enforce these bounds during local optimization. Other local solvers are
    /// unconstrained and can return solutions outside them.
    fn variable_bounds(&self) -> Array2<f64>;

    /// Evaluates the constraints at `x`.
    ///
    /// The returned array must contain the same number of values in the same order
    /// on every successful evaluation. The first evaluation determines the number
    /// of constraints for an optimization run.
    ///
    /// **Sign Convention**:
    /// - **Positive or zero**: constraint satisfied  
    /// - **Negative**: constraint violated
    ///
    /// The default implementation returns an empty array (no constraints).
    /// COBYLA and SLSQP accept nonempty nonlinear constraints; the other
    /// Basin-backed solvers reject them. Use COBYLA for derivative-free
    /// constrained problems and SLSQP when gradients are available.
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
    ///     fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
    ///         Ok(ndarray::array![
    ///             1.0 - x[0] - x[1], // x[0] + x[1] <= 1.0
    ///         ])
    ///     }
    /// }
    /// ```
    fn constraints(&self, _x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(Array1::from_vec(Vec::new()))
    }

    /// Linear inequality constraints `A x <= b`.
    ///
    /// Returns `None` (the default) when the problem has no linear
    /// inequalities. `A` is `m x n` and `b` has length `m`, where `n` is the
    /// problem dimension. Shapes must remain fixed throughout a solve.
    ///
    /// Accepted by SLSQP, Barrier, LINCOA-style solvers, and COBYLA (folded
    /// into the nonlinear system). The classic Basin-backed solvers
    /// (L-BFGS, gradient descent, trust region, Nelder-Mead, L-BFGS-B,
    /// bounded Nelder-Mead, BOBYQA) reject problems with linear constraints.
    fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        None
    }

    /// Linear equality constraints `A x = b`.
    ///
    /// Returns `None` (the default) when the problem has no linear
    /// equalities. `A` is `m x n` and `b` has length `m`. Shapes must remain
    /// fixed throughout a solve.
    ///
    /// Accepted by SLSQP, AugmentedLagrangian, and COBYLA (each equality is
    /// folded into a pair of inequalities). Other solvers reject problems
    /// with linear equalities.
    fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        None
    }

    /// Nonlinear equality constraints `h(x) = 0`.
    ///
    /// The default implementation returns an empty array (no equalities).
    /// Counts and order must remain stable across evaluations. Accepted by
    /// SLSQP and COBYLA; other solvers reject nonempty nonlinear equalities.
    fn nonlinear_equalities(&self, _x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(Array1::from_vec(Vec::new()))
    }

    /// Jacobian of the nonlinear constraint blocks at `x`.
    ///
    /// Rows hold nonlinear equalities first, then nonlinear inequalities in
    /// callback order; columns correspond to variables. The matrix has
    /// `num_nonlinear_equalities + num_nonlinear_constraints` rows and `n`
    /// columns (zero rows when unconstrained). Signs follow the public
    /// convention: differentiate `h(x)` and `g(x)` (with `g(x) >= 0`
    /// satisfied) directly; the SLSQP adapter negates inequality rows
    /// internally to match Basin's `c(x) <= 0` form.
    ///
    /// Required by SLSQP. The default returns
    /// [`EvaluationError::ConstraintJacobianNotImplemented`].
    fn constraint_jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Err(EvaluationError::ConstraintJacobianNotImplemented)
    }
}

pub(crate) fn evaluate_constraints<P: Problem>(
    problem: &P,
    x: &Array1<f64>,
    dimension: &OnceLock<usize>,
) -> Result<Array1<f64>, EvaluationError> {
    if dimension.get() == Some(&0) {
        return Ok(Array1::from_vec(Vec::new()));
    }

    let constraints = problem.constraints(x)?;
    let actual = constraints.len();
    let expected = *dimension.get_or_init(|| actual);

    if actual != expected {
        return Err(EvaluationError::ConstraintDimensionMismatch { expected, actual });
    }

    Ok(constraints)
}

pub(crate) fn evaluate_nonlinear_equalities<P: Problem>(
    problem: &P,
    x: &Array1<f64>,
    dimension: &OnceLock<usize>,
) -> Result<Array1<f64>, EvaluationError> {
    if dimension.get() == Some(&0) {
        return Ok(Array1::from_vec(Vec::new()));
    }

    let equalities = problem.nonlinear_equalities(x)?;
    let actual = equalities.len();
    let expected = *dimension.get_or_init(|| actual);

    if actual != expected {
        return Err(EvaluationError::ConstraintDimensionMismatch { expected, actual });
    }

    Ok(equalities)
}

/// Tolerance for treating linear/nonlinear equality residuals as satisfied
/// in global-search feasibility filtering. Local solvers apply their own
/// duality/feasibility tolerances once started.
pub(crate) const EQUALITY_FEASIBILITY_TOLERANCE: f64 = 1e-8;

/// Tolerance for `A x <= b` feasibility filtering.
pub(crate) const LINEAR_INEQUALITY_FEASIBILITY_TOLERANCE: f64 = 1e-9;

/// Validates the shapes of the optional linear constraint blocks.
///
/// Returns `(m_ineq, m_eq)`. Errors with `InvalidInput` on column or
/// right-hand-side mismatches.
pub(crate) fn validate_linear_blocks<P: Problem>(
    problem: &P,
    n: usize,
) -> Result<(usize, usize), EvaluationError> {
    let mut m_ineq = 0;
    let mut m_eq = 0;
    if let Some((a, b)) = problem.linear_inequalities() {
        if a.ncols() != n {
            return Err(EvaluationError::InvalidInput {
                reason: format!(
                    "Linear inequalities matrix must have {n} columns, got {}.",
                    a.ncols()
                ),
            });
        }
        if a.nrows() != b.len() {
            return Err(EvaluationError::InvalidInput {
                reason: format!(
                    "Linear inequalities shape mismatch: A has {} rows, b has {}.",
                    a.nrows(),
                    b.len()
                ),
            });
        }
        m_ineq = a.nrows();
    }
    if let Some((a, b)) = problem.linear_equalities() {
        if a.ncols() != n {
            return Err(EvaluationError::InvalidInput {
                reason: format!(
                    "Linear equalities matrix must have {n} columns, got {}.",
                    a.ncols()
                ),
            });
        }
        if a.nrows() != b.len() {
            return Err(EvaluationError::InvalidInput {
                reason: format!(
                    "Linear equalities shape mismatch: A has {} rows, b has {}.",
                    a.nrows(),
                    b.len()
                ),
            });
        }
        m_eq = a.nrows();
    }
    Ok((m_ineq, m_eq))
}

/// Projects `point` onto the affine subspace `A x = b` defined by
/// [`Problem::linear_equalities`], alternating with box clamping.
///
/// Scatter search rejection-samples the feasible region, which has zero
/// volume for equalities. A few rounds of alternating projections (affine
/// subspace and box are both convex) move random samples onto the feasible
/// affine patch when one exists inside the box. Returns the original point
/// when there are no equalities, shapes are invalid (validated later), or
/// the normal system is singular.
pub(crate) fn project_onto_linear_equalities<P: Problem>(
    problem: &P,
    point: &Array1<f64>,
) -> Array1<f64> {
    let Some((a, b)) = problem.linear_equalities() else {
        return point.clone();
    };
    let n = point.len();
    if a.ncols() != n || a.nrows() != b.len() || a.nrows() == 0 {
        return point.clone();
    }
    let m = a.nrows();
    // Normal matrix G = A A^T (m x m).
    let mut g = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[[i, k]] * a[[j, k]];
            }
            g[i][j] = sum;
            g[j][i] = sum;
        }
    }
    let bounds = problem.variable_bounds();
    let in_box = bounds.dim() == (n, 2);
    let mut x = point.to_vec();
    for _ in 0..3 {
        // Affine step: x <- x - A^T y with G y = (A x - b).
        let mut residual = vec![0.0; m];
        for i in 0..m {
            let mut dot = 0.0;
            for k in 0..n {
                dot += a[[i, k]] * x[k];
            }
            residual[i] = dot - b[i];
        }
        let Some(step) = solve_dense(g.clone(), residual) else {
            return point.clone();
        };
        for k in 0..n {
            let mut correction = 0.0;
            for i in 0..m {
                correction += a[[i, k]] * step[i];
            }
            x[k] -= correction;
        }
        if in_box {
            for k in 0..n {
                x[k] = x[k].clamp(bounds[[k, 0]], bounds[[k, 1]]);
            }
        }
    }
    Array1::from_vec(x)
}

/// Solves a small dense system by Gaussian elimination with partial pivoting.
///
/// Returns `None` when the matrix is (numerically) singular.
fn solve_dense(mut g: Vec<Vec<f64>>, mut residual: Vec<f64>) -> Option<Vec<f64>> {
    let m = residual.len();
    for col in 0..m {
        let mut pivot = col;
        for row in (col + 1)..m {
            if g[row][col].abs() > g[pivot][col].abs() {
                pivot = row;
            }
        }
        if g[pivot][col].abs() < 1e-12 {
            return None;
        }
        g.swap(col, pivot);
        residual.swap(col, pivot);
        let inv = 1.0 / g[col][col];
        for row in (col + 1)..m {
            let factor = g[row][col] * inv;
            let (above, below) = g.split_at_mut(row);
            let pivot_row = &above[col];
            let target_row = &mut below[0];
            for (target, &pivot) in target_row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *target -= factor * pivot;
            }
            residual[row] -= factor * residual[col];
        }
    }
    let mut solution = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = residual[i];
        for (k, &coeff) in g[i].iter().enumerate().skip(i + 1) {
            sum -= coeff * solution[k];
        }
        if g[i][i].abs() < 1e-12 {
            return None;
        }
        solution[i] = sum / g[i][i];
    }
    Some(solution)
}

/// Full feasibility check across all constraint blocks.
///
/// Nonlinear inequalities use the `g(x) >= 0` convention; nonlinear and
/// linear equalities use absolute tolerances; linear inequalities use `A x
/// <= b` with a small tolerance for rounding.
pub(crate) fn is_feasible_point<P: Problem>(
    problem: &P,
    point: &Array1<f64>,
    constraint_dimension: &OnceLock<usize>,
    equality_dimension: &OnceLock<usize>,
) -> Result<bool, EvaluationError> {
    if !evaluate_constraints(problem, point, constraint_dimension)?
        .iter()
        .all(|&value| value >= 0.0)
    {
        return Ok(false);
    }
    if !evaluate_nonlinear_equalities(problem, point, equality_dimension)?
        .iter()
        .all(|&value| value.abs() <= EQUALITY_FEASIBILITY_TOLERANCE)
    {
        return Ok(false);
    }
    validate_linear_blocks(problem, point.len())?;
    if let Some((a, b)) = problem.linear_inequalities() {
        let residual = a.dot(point) - &b;
        if !residual.iter().all(|&value| value <= LINEAR_INEQUALITY_FEASIBILITY_TOLERANCE) {
            return Ok(false);
        }
    }
    if let Some((a, b)) = problem.linear_equalities() {
        let residual = a.dot(point) - &b;
        if !residual.iter().all(|&value| value.abs() <= EQUALITY_FEASIBILITY_TOLERANCE) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct ParameterizedConstraint {
        limit: f64,
    }

    impl Problem for ParameterizedConstraint {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(x.sum())
        }

        fn variable_bounds(&self) -> Array2<f64> {
            array![[0.0, 2.0], [0.0, 2.0]]
        }

        fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![self.limit - x.sum()])
        }
    }

    #[test]
    fn constraints_can_use_problem_data() {
        let problem = ParameterizedConstraint { limit: 1.5 };

        assert_eq!(problem.constraints(&array![0.5, 0.25]).unwrap(), array![0.75]);
    }

    #[test]
    fn constraints_default_to_an_empty_vector() {
        struct Unconstrained;

        impl Problem for Unconstrained {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0]]
            }
        }

        let problem = Unconstrained;

        assert!(problem.constraints(&array![0.5]).unwrap().is_empty());
    }

    #[test]
    fn constraint_dimension_is_validated() {
        struct VariableDimensionConstraints;

        impl Problem for VariableDimensionConstraints {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0]]
            }

            fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
                if x[0] < 0.5 { Ok(array![1.0, 2.0]) } else { Ok(array![1.0]) }
            }
        }

        let dimension = OnceLock::new();
        let constraints =
            evaluate_constraints(&VariableDimensionConstraints, &array![0.25], &dimension).unwrap();
        assert_eq!(constraints, array![1.0, 2.0]);

        let error = evaluate_constraints(&VariableDimensionConstraints, &array![0.75], &dimension)
            .unwrap_err();

        assert!(matches!(
            error,
            EvaluationError::ConstraintDimensionMismatch { expected: 2, actual: 1 }
        ));
    }

    #[test]
    /// New constraint blocks default to absent/empty so existing problems
    /// keep working unchanged.
    fn new_constraint_blocks_default_to_absent() {
        struct Unconstrained;

        impl Problem for Unconstrained {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0]]
            }
        }

        let problem = Unconstrained;
        assert!(problem.linear_inequalities().is_none());
        assert!(problem.linear_equalities().is_none());
        assert!(problem.nonlinear_equalities(&array![0.5]).unwrap().is_empty());
        assert!(matches!(
            problem.constraint_jacobian(&array![0.5]).unwrap_err(),
            EvaluationError::ConstraintJacobianNotImplemented
        ));
        // No blocks means every point is feasible.
        assert!(
            is_feasible_point(&problem, &array![0.5], &OnceLock::new(), &OnceLock::new()).unwrap()
        );
    }

    #[test]
    /// Linear block shapes are validated against the problem dimension.
    fn linear_block_shapes_are_validated() {
        struct Shape {
            ineq: Option<(Array2<f64>, Array1<f64>)>,
            eq: Option<(Array2<f64>, Array1<f64>)>,
        }

        impl Problem for Shape {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0], [0.0, 1.0]]
            }

            fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                self.ineq.clone()
            }

            fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                self.eq.clone()
            }
        }

        let valid = Shape {
            ineq: Some((array![[1.0, 1.0]], array![1.0])),
            eq: Some((array![[1.0, -1.0]], array![0.0])),
        };
        assert_eq!(validate_linear_blocks(&valid, 2).unwrap(), (1, 1));

        // Wrong column count.
        let bad_columns = Shape { ineq: Some((array![[1.0, 1.0, 1.0]], array![1.0])), eq: None };
        assert!(matches!(
            validate_linear_blocks(&bad_columns, 2).unwrap_err(),
            EvaluationError::InvalidInput { .. }
        ));

        // Right-hand side length differs from the row count.
        let bad_rhs = Shape { ineq: None, eq: Some((array![[1.0, 0.0]], array![0.0, 0.0])) };
        assert!(matches!(
            validate_linear_blocks(&bad_rhs, 2).unwrap_err(),
            EvaluationError::InvalidInput { .. }
        ));
    }

    #[test]
    /// Nonlinear equality counts are locked after the first evaluation.
    fn nonlinear_equality_dimension_is_validated() {
        struct VariableEqualities;

        impl Problem for VariableEqualities {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0]]
            }

            fn nonlinear_equalities(
                &self,
                x: &Array1<f64>,
            ) -> Result<Array1<f64>, EvaluationError> {
                if x[0] < 0.5 { Ok(array![1.0]) } else { Ok(Array1::from_vec(vec![])) }
            }
        }

        let dimension = OnceLock::new();
        let equalities =
            evaluate_nonlinear_equalities(&VariableEqualities, &array![0.25], &dimension).unwrap();
        assert_eq!(equalities, array![1.0]);

        let error = evaluate_nonlinear_equalities(&VariableEqualities, &array![0.75], &dimension)
            .unwrap_err();
        assert!(matches!(
            error,
            EvaluationError::ConstraintDimensionMismatch { expected: 1, actual: 0 }
        ));
    }

    #[test]
    /// Each constraint block is checked in order and can reject a point.
    fn feasibility_covers_all_blocks() {
        struct AllBlocks;

        impl Problem for AllBlocks {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 2.0], [0.0, 2.0]]
            }

            fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
                Ok(array![1.0 - x[0]])
            }

            fn nonlinear_equalities(
                &self,
                x: &Array1<f64>,
            ) -> Result<Array1<f64>, EvaluationError> {
                Ok(array![x[1] - 0.5])
            }

            fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                Some((array![[1.0, 0.0]], array![0.75]))
            }

            fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                Some((array![[1.0, 1.0]], array![1.0]))
            }
        }

        let problem = AllBlocks;
        // Satisfies every block: x0 = 0.5, x1 = 0.5.
        let feasible = array![0.5, 0.5];
        assert!(
            is_feasible_point(&problem, &feasible, &OnceLock::new(), &OnceLock::new()).unwrap()
        );

        // Blocks are checked in order, so each point passes the earlier
        // blocks and is rejected by the named one.
        for (point, block) in [
            (array![1.5, -0.5], "nonlinear inequality"),
            (array![0.5, 0.6], "nonlinear equality"),
            (array![0.8, 0.5], "linear inequality"),
            (array![0.2, 0.5], "linear equality"),
        ] {
            assert!(
                !is_feasible_point(&problem, &point, &OnceLock::new(), &OnceLock::new()).unwrap(),
                "{point:?} should be rejected by its {block}"
            );
        }

        // Malformed linear shapes surface as errors, not silent filtering.
        struct BadShapes;
        impl Problem for BadShapes {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0], [0.0, 1.0]]
            }

            fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                Some((array![[1.0]], array![1.0]))
            }
        }
        assert!(matches!(
            is_feasible_point(&BadShapes, &array![0.5, 0.5], &OnceLock::new(), &OnceLock::new())
                .unwrap_err(),
            EvaluationError::InvalidInput { .. }
        ));
    }

    #[test]
    /// Projection lands on `A x = b` and degrades gracefully.
    fn projection_onto_linear_equalities() {
        struct WithEqualities;

        impl Problem for WithEqualities {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[-5.0, 5.0], [-5.0, 5.0]]
            }

            fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                Some((array![[1.0, 1.0]], array![1.0]))
            }
        }

        let problem = WithEqualities;
        let projected = project_onto_linear_equalities(&problem, &array![2.0, 3.0]);
        assert!((projected[0] + projected[1] - 1.0).abs() < 1e-12);
        // Minimum-norm correction of (2, 3) onto x + y = 1 is (0, 1).
        assert!((&projected - array![0.0, 1.0]).mapv(f64::abs).sum() < 1e-12);

        // Already-feasible points are (numerically) unchanged.
        let projected = project_onto_linear_equalities(&problem, &array![0.25, 0.75]);
        assert!((&projected - array![0.25, 0.75]).mapv(f64::abs).sum() < 1e-12);

        // Without equalities the point passes through untouched.
        struct WithoutEqualities;
        impl Problem for WithoutEqualities {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0]]
            }
        }
        let untouched = array![0.7];
        assert_eq!(project_onto_linear_equalities(&WithoutEqualities, &untouched), untouched);

        // A rank-deficient row system cannot be solved; the point is kept so
        // the caller (feasibility filter) can drop it normally.
        struct Singular;
        impl Problem for Singular {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                array![[0.0, 1.0], [0.0, 1.0]]
            }

            fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
                Some((array![[0.0, 0.0]], array![1.0]))
            }
        }
        let kept = array![0.2, 0.3];
        assert_eq!(project_onto_linear_equalities(&Singular, &kept), kept);
    }
}
