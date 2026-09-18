//! Adapters for Basin's linearly/nonlinearly constrained local solvers.
//!
//! - SLSQP: gradient-based, handles box + linear + nonlinear constraints.
//! - Barrier: log-barrier over a BFGS inner solver; box bounds are folded
//!   into the linear inequality system, so `A x <= b` and the box are both
//!   enforced exactly.
//! - AugmentedLagrangian: ALM over a BFGS inner solver for `A x = b`. Box
//!   bounds are *not* enforced during local search (like the existing
//!   unconstrained solvers); use SLSQP when box and equalities must both
//!   hold strictly.
//!
//! The problem parameter is `Vec<f64>` with [`basin::DenseMatrix`] so no
//! extra linear-algebra backend feature is required.

use super::builders::LocalSolverConfig;
use super::runner::LocalSolverError;
use crate::problem::{Problem, validate_linear_blocks};
use crate::types::{EvaluationError, LocalSolution, LocalSolverType};
use basin::State as _;
use ndarray::{Array1, Array2};

fn invalid(solver_type: &LocalSolverType, reason: impl Into<String>) -> LocalSolverError {
    LocalSolverError::InvalidConfig {
        solver_type: format!("{solver_type:?}"),
        reason: reason.into(),
    }
}

fn failed(solver_type: &LocalSolverType, error: EvaluationError) -> LocalSolverError {
    LocalSolverError::RunFailed {
        solver_type: format!("{solver_type:?}"),
        reason: error.to_string(),
    }
}

fn to_dense(a: &Array2<f64>) -> basin::DenseMatrix {
    basin::DenseMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[[i, j]])
}

fn check_accuracy(
    solver_type: &LocalSolverType,
    name: &str,
    value: Option<f64>,
) -> Result<(), LocalSolverError> {
    if value.is_none_or(|v| v.is_finite() && v >= 0.0) {
        Ok(())
    } else {
        Err(invalid(solver_type, format!("`{name}` must be finite and nonnegative, or None.")))
    }
}

fn check_positive(
    solver_type: &LocalSolverType,
    name: &str,
    value: f64,
) -> Result<(), LocalSolverError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(solver_type, format!("`{name}` must be finite and positive.")))
    }
}

fn check_strictly_above_one(
    solver_type: &LocalSolverType,
    name: &str,
    value: f64,
) -> Result<(), LocalSolverError> {
    if value.is_finite() && value > 1.0 {
        Ok(())
    } else {
        Err(invalid(solver_type, format!("`{name}` must be finite and exceed 1.")))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ConstrainedMode {
    Slsqp,
    Barrier,
    AugmentedLagrangian,
}

/// Cached constraint data for one local solve.
struct ConstrainedProblem<'a, P> {
    problem: &'a P,
    lower: Vec<f64>,
    upper: Vec<f64>,
    a_ineq: Option<(basin::DenseMatrix, Vec<f64>)>,
    a_eq: Option<(basin::DenseMatrix, Vec<f64>)>,
    /// Barrier only: box rows folded into the inequality system.
    ineq_combined: Option<(basin::DenseMatrix, Vec<f64>)>,
    n_nonlin_ineq: usize,
    n_nonlin_eq: usize,
    /// Barrier/ALM: evaluate user callbacks at the box-projected point so
    /// out-of-bounds trial points stay inside the callback domain (the same
    /// tradeoff as the COBYLA adapter). Linear residuals always use the raw
    /// trial point so the solver observes true violations.
    project_callbacks: bool,
}

impl<'a, P: Problem> ConstrainedProblem<'a, P> {
    fn new(
        problem: &'a P,
        initial: &Array1<f64>,
        solver_type: &LocalSolverType,
        mode: ConstrainedMode,
    ) -> Result<(Self, Vec<f64>), LocalSolverError> {
        if initial.is_empty() || initial.iter().any(|v| !v.is_finite()) {
            return Err(invalid(solver_type, "The starting point must be nonempty and finite."));
        }
        let n = initial.len();
        let bounds = problem.variable_bounds();
        if bounds.dim() != (n, 2) {
            return Err(invalid(solver_type, format!("Bounds must have shape ({n}, 2).")));
        }
        let mut lower = Vec::with_capacity(n);
        let mut upper = Vec::with_capacity(n);
        for i in 0..n {
            let (lo, hi) = (bounds[[i, 0]], bounds[[i, 1]]);
            if !lo.is_finite() || !hi.is_finite() || lo >= hi || !(hi - lo).is_finite() {
                return Err(invalid(
                    solver_type,
                    "Bounds must be finite, with lower < upper and a finite width.",
                ));
            }
            lower.push(lo);
            upper.push(hi);
        }
        validate_linear_blocks(problem, n).map_err(|error| failed(solver_type, error))?;

        let fail = |error: EvaluationError| failed(solver_type, error);
        let start_point = Array1::from_vec(initial.to_vec());
        let n_nonlin_ineq = problem.constraints(&start_point).map_err(fail)?.len();
        let n_nonlin_eq = problem.nonlinear_equalities(&start_point).map_err(fail)?.len();

        let a_ineq = problem.linear_inequalities().map(|(a, b)| (to_dense(&a), b.to_vec()));
        let a_eq = problem.linear_equalities().map(|(a, b)| (to_dense(&a), b.to_vec()));

        // Barrier folds the box into A x <= b: rows e_i (x_i <= upper_i) and
        // -e_i (-x_i <= -lower_i) ahead of the user rows.
        let ineq_combined = if mode == ConstrainedMode::Barrier {
            let m_user = a_ineq.as_ref().map_or(0, |(a, _)| a.nrows());
            let combined = basin::DenseMatrix::from_fn(2 * n + m_user, n, |i, j| {
                if i < n {
                    if i == j { 1.0 } else { 0.0 }
                } else if i < 2 * n {
                    if i - n == j { -1.0 } else { 0.0 }
                } else if let Some((a, _)) = a_ineq.as_ref() {
                    a.get(i - 2 * n, j)
                } else {
                    0.0
                }
            });
            let mut rhs = Vec::with_capacity(2 * n + m_user);
            rhs.extend_from_slice(&upper);
            rhs.extend(lower.iter().map(|lo| -lo));
            if let Some((_, b)) = a_ineq.as_ref() {
                rhs.extend_from_slice(b);
            }
            Some((combined, rhs))
        } else {
            None
        };

        let mut start = initial.to_vec();
        for i in 0..n {
            start[i] = start[i].clamp(lower[i], upper[i]);
        }

        Ok((
            Self {
                problem,
                lower,
                upper,
                a_ineq,
                a_eq,
                ineq_combined,
                n_nonlin_ineq,
                n_nonlin_eq,
                project_callbacks: mode != ConstrainedMode::Slsqp,
            },
            start,
        ))
    }

    fn callback_point(&self, param: &[f64]) -> Array1<f64> {
        if self.project_callbacks {
            Array1::from_vec(
                param
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v.clamp(self.lower[i], self.upper[i]))
                    .collect(),
            )
        } else {
            Array1::from_vec(param.to_vec())
        }
    }

    fn effective_inequalities(&self) -> Option<(&basin::DenseMatrix, &Vec<f64>)> {
        if let Some(combined) = self.ineq_combined.as_ref() {
            Some((&combined.0, &combined.1))
        } else {
            self.a_ineq.as_ref().map(|(a, b)| (a, b))
        }
    }
}

impl<P: Problem> basin::CostFunction for ConstrainedProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationError;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, EvaluationError> {
        self.problem.objective(&self.callback_point(x))
    }
}

impl<P: Problem> basin::Gradient for ConstrainedProblem<'_, P> {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, EvaluationError> {
        let gradient = self.problem.gradient(&self.callback_point(x))?;
        if gradient.len() != x.len() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(EvaluationError::InvalidInput {
                reason: format!("Gradient must contain {} finite values.", x.len()),
            });
        }
        Ok(gradient.to_vec())
    }
}

impl<P: Problem> basin::LinearInequalityConstraints for ConstrainedProblem<'_, P> {
    type Matrix = basin::DenseMatrix;

    fn a(&self) -> &Self::Matrix {
        self.effective_inequalities().expect("linear inequalities must be present").0
    }

    fn b(&self) -> &Vec<f64> {
        self.effective_inequalities().expect("linear inequalities must be present").1
    }
}

impl<P: Problem> basin::LinearEqualityConstraints for ConstrainedProblem<'_, P> {
    type Matrix = basin::DenseMatrix;

    fn a(&self) -> &Self::Matrix {
        &self.a_eq.as_ref().expect("linear equalities must be present").0
    }

    fn b(&self) -> &Vec<f64> {
        &self.a_eq.as_ref().expect("linear equalities must be present").1
    }
}

impl<P: Problem> basin::NonlinearConstraints for ConstrainedProblem<'_, P> {
    type Matrix = basin::DenseMatrix;

    fn nonlinear_constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, EvaluationError> {
        let values = self.problem.constraints(&Array1::from_vec(x.clone()))?;
        if values.len() != self.n_nonlin_ineq {
            return Err(EvaluationError::ConstraintDimensionMismatch {
                expected: self.n_nonlin_ineq,
                actual: values.len(),
            });
        }
        // Basin uses c(x) <= 0; globalsearch uses g(x) >= 0 satisfied.
        Ok(values.iter().map(|v| -v).collect())
    }

    fn num_nonlinear_constraints(&self) -> usize {
        self.n_nonlin_ineq
    }

    fn num_nonlinear_equalities(&self) -> usize {
        self.n_nonlin_eq
    }

    fn nonlinear_equalities(&self, x: &Vec<f64>) -> Result<Option<Vec<f64>>, EvaluationError> {
        if self.n_nonlin_eq == 0 {
            return Ok(None);
        }
        let values = self.problem.nonlinear_equalities(&Array1::from_vec(x.clone()))?;
        if values.len() != self.n_nonlin_eq {
            return Err(EvaluationError::ConstraintDimensionMismatch {
                expected: self.n_nonlin_eq,
                actual: values.len(),
            });
        }
        Ok(Some(values.to_vec()))
    }

    fn inequalities(&self) -> Option<(&Self::Matrix, &Vec<f64>)> {
        self.effective_inequalities()
    }

    fn equalities(&self) -> Option<(&Self::Matrix, &Vec<f64>)> {
        self.a_eq.as_ref().map(|(a, b)| (a, b))
    }

    fn lower(&self) -> Option<&Vec<f64>> {
        Some(&self.lower)
    }

    fn upper(&self) -> Option<&Vec<f64>> {
        Some(&self.upper)
    }
}

impl<P: Problem> basin::ConstraintJacobian for ConstrainedProblem<'_, P> {
    fn constraint_jacobian(&self, x: &Vec<f64>) -> Result<Self::Matrix, EvaluationError> {
        let total_rows = self.n_nonlin_eq + self.n_nonlin_ineq;
        if total_rows == 0 {
            return Ok(basin::DenseMatrix::from_fn(0, x.len(), |_, _| 0.0));
        }
        let jacobian = self.problem.constraint_jacobian(&Array1::from_vec(x.clone()))?;
        if jacobian.dim() != (total_rows, x.len()) {
            return Err(EvaluationError::InvalidInput {
                reason: format!(
                    "Constraint Jacobian must have shape ({total_rows}, {}), got ({}, {}).",
                    x.len(),
                    jacobian.nrows(),
                    jacobian.ncols()
                ),
            });
        }
        // Equalities keep their sign; inequality rows are negated to match
        // Basin's c(x) <= 0 convention.
        Ok(basin::DenseMatrix::from_fn(total_rows, x.len(), |i, j| {
            let value = jacobian[[i, j]];
            if i < self.n_nonlin_eq { value } else { -value }
        }))
    }
}

fn finish<S: basin::State<Param = Vec<f64>, Float = f64>>(
    result: basin::OptimizationResult<S>,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if result.reason.is_failure() {
        return Err(LocalSolverError::RunFailed {
            solver_type: format!("{solver_type:?}"),
            reason: format!("solver terminated with {:?}", result.reason),
        });
    }
    if !result.best_cost().is_finite() || result.best_param().iter().any(|v| !v.is_finite()) {
        return Err(LocalSolverError::NoSolution {
            solver_type: format!("{solver_type:?}"),
            iterations: result.iter(),
        });
    }
    let evaluations = if track { result.cost_evals() } else { 0 };
    Ok((
        LocalSolution {
            point: Array1::from_vec(result.best_param().clone()),
            objective: result.best_cost(),
        },
        evaluations,
    ))
}

/// Finish for the outer-loop methods (Barrier, AugmentedLagrangian).
///
/// Their `best_*` readers track the inner surrogate (barrier/augmented
/// Lagrangian) across outer iterations with changing parameters, so the
/// minimum-surrogate point need not be the constrained solution. The latest
/// accepted state is the solution; re-evaluate the true objective there.
fn finish_outer_loop<P: Problem>(
    problem: &P,
    result: basin::OptimizationResult<basin::BasicState<Vec<f64>>>,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if result.reason.is_failure() {
        return Err(LocalSolverError::RunFailed {
            solver_type: format!("{solver_type:?}"),
            reason: format!("solver terminated with {:?}", result.reason),
        });
    }
    let point = Array1::from_vec(result.state.param().clone());
    if point.iter().any(|v| !v.is_finite()) {
        return Err(LocalSolverError::NoSolution {
            solver_type: format!("{solver_type:?}"),
            iterations: result.iter(),
        });
    }
    let objective = problem.objective(&point).map_err(|error| failed(solver_type, error))?;
    if !objective.is_finite() {
        return Err(LocalSolverError::NoSolution {
            solver_type: format!("{solver_type:?}"),
            iterations: result.iter(),
        });
    }
    let evaluations = if track { result.cost_evals() } else { 0 };
    Ok((LocalSolution { point, objective }, evaluations))
}

pub(super) fn solve_slsqp<P: Problem>(
    problem: &P,
    initial: Array1<f64>,
    config: &LocalSolverConfig,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if &config.solver_type() != solver_type {
        return Err(invalid(solver_type, "Solver type and configuration do not match."));
    }
    let LocalSolverConfig::SLSQP { max_iter, accuracy, max_subproblem_iter } = config else {
        return Err(invalid(solver_type, "Expected an SLSQP configuration."));
    };
    check_accuracy(solver_type, "accuracy", *accuracy)?;

    let (adapter, start) =
        ConstrainedProblem::new(problem, &initial, solver_type, ConstrainedMode::Slsqp)?;
    let fail = |error: EvaluationError| failed(solver_type, error);

    // SLSQP accepts every constraint block; the Jacobian is only consulted
    // when nonlinear constraints are present (handled inside the adapter).
    let mut solver = basin::Slsqp::new().with_absolute_accuracy_tolerance(*accuracy);
    if let Some(limit) = max_subproblem_iter {
        if *limit == 0 {
            return Err(invalid(solver_type, "`max_subproblem_iter` must be positive when set."));
        }
        solver = solver.with_max_subproblem_iterations(*limit);
    }
    let result = basin::Executor::from_start(adapter, solver, start)
        .max_iter(*max_iter)
        .run()
        .map_err(fail)?;
    finish(result, solver_type, track)
}

pub(super) fn solve_barrier<P: Problem>(
    problem: &P,
    initial: Array1<f64>,
    config: &LocalSolverConfig,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if &config.solver_type() != solver_type {
        return Err(invalid(solver_type, "Solver type and configuration do not match."));
    }
    let LocalSolverConfig::Barrier { max_iter, mu0, reduction, duality_gap_tol, inner_max_iter } =
        config
    else {
        return Err(invalid(solver_type, "Expected a Barrier configuration."));
    };
    check_positive(solver_type, "mu0", *mu0)?;
    check_strictly_above_one(solver_type, "reduction", *reduction)?;
    check_positive(solver_type, "duality_gap_tol", *duality_gap_tol)?;
    if *inner_max_iter == 0 {
        return Err(invalid(solver_type, "`inner_max_iter` must be positive."));
    }

    let (adapter, start) =
        ConstrainedProblem::new(problem, &initial, solver_type, ConstrainedMode::Barrier)?;
    let fail = |error: EvaluationError| failed(solver_type, error);

    if adapter.a_ineq.as_ref().is_none_or(|(a, _)| a.nrows() == 0) {
        return Err(invalid(
            solver_type,
            "Barrier requires `linear_inequalities` with at least one row (A x <= b).",
        ));
    }
    if adapter.a_eq.as_ref().is_some_and(|(a, _)| a.nrows() > 0) {
        return Err(invalid(
            solver_type,
            "Barrier handles only linear inequalities; move equalities to SLSQP or AugmentedLagrangian.",
        ));
    }
    if adapter.n_nonlin_ineq > 0 || adapter.n_nonlin_eq > 0 {
        return Err(invalid(
            solver_type,
            "Barrier handles only linear inequalities; move nonlinear constraints to SLSQP or COBYLA.",
        ));
    }

    let inner = basin::Bfgs::with_line_search(basin::Backtracking::new())
        .with_absolute_gradient_tolerance(Some(1e-8));
    let solver = basin::BarrierMethod::with_inner_solver(inner)
        .mu0(*mu0)
        .with_reduction(*reduction)
        .with_absolute_duality_gap_tolerance(*duality_gap_tol)
        .with_inner_max_iter(*inner_max_iter);
    let result = basin::Executor::from_start(adapter, solver, start)
        .max_iter(*max_iter)
        .run()
        .map_err(fail)?;
    finish_outer_loop(problem, result, solver_type, track)
}

pub(super) fn solve_augmented_lagrangian<P: Problem>(
    problem: &P,
    initial: Array1<f64>,
    config: &LocalSolverConfig,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if &config.solver_type() != solver_type {
        return Err(invalid(solver_type, "Solver type and configuration do not match."));
    }
    let LocalSolverConfig::AugmentedLagrangian {
        max_iter,
        rho0,
        rho_increase,
        feasibility_decrease,
        feasibility_tol,
        inner_max_iter,
    } = config
    else {
        return Err(invalid(solver_type, "Expected an AugmentedLagrangian configuration."));
    };
    check_positive(solver_type, "rho0", *rho0)?;
    check_strictly_above_one(solver_type, "rho_increase", *rho_increase)?;
    if !feasibility_decrease.is_finite() || !(0.0..1.0).contains(feasibility_decrease) {
        return Err(invalid(solver_type, "`feasibility_decrease` must lie in (0, 1)."));
    }
    check_positive(solver_type, "feasibility_tol", *feasibility_tol)?;
    if *inner_max_iter == 0 {
        return Err(invalid(solver_type, "`inner_max_iter` must be positive."));
    }

    let (adapter, start) = ConstrainedProblem::new(
        problem,
        &initial,
        solver_type,
        ConstrainedMode::AugmentedLagrangian,
    )?;
    let fail = |error: EvaluationError| failed(solver_type, error);

    if adapter.a_eq.as_ref().is_none_or(|(a, _)| a.nrows() == 0) {
        return Err(invalid(
            solver_type,
            "AugmentedLagrangian requires `linear_equalities` with at least one row (A x = b).",
        ));
    }
    if adapter.a_ineq.as_ref().is_some_and(|(a, _)| a.nrows() > 0) {
        return Err(invalid(
            solver_type,
            "AugmentedLagrangian handles only linear equalities; move inequalities to SLSQP or Barrier.",
        ));
    }
    if adapter.n_nonlin_ineq > 0 || adapter.n_nonlin_eq > 0 {
        return Err(invalid(
            solver_type,
            "AugmentedLagrangian handles only linear equalities; move nonlinear constraints to SLSQP or COBYLA.",
        ));
    }

    let inner = basin::Bfgs::with_line_search(basin::Backtracking::new())
        .with_absolute_gradient_tolerance(Some(1e-8));
    let solver = basin::AugmentedLagrangianMethod::with_inner_solver(inner)
        .rho0(*rho0)
        .with_rho_increase(*rho_increase)
        .with_feasibility_decrease(*feasibility_decrease)
        .with_absolute_feasibility_tolerance(*feasibility_tol)
        .with_inner_max_iter(*inner_max_iter);
    let result = basin::Executor::from_start(adapter, solver, start)
        .max_iter(*max_iter)
        .run()
        .map_err(fail)?;
    finish_outer_loop(problem, result, solver_type, track)
}

/// Post-solve feasibility gate shared by the constrained solvers.
///
/// Basin reports success on its own merit/dual measures; enforce the public
/// contract here so marginal points are surfaced as errors instead of
/// silently infeasible solutions.
pub(super) fn check_constrained_feasibility<P: Problem>(
    problem: &P,
    solution: &Array1<f64>,
    solver_type: &LocalSolverType,
) -> Result<(), LocalSolverError> {
    let n = solution.len();
    validate_linear_blocks(problem, n).map_err(|error| failed(solver_type, error))?;
    let violated = |reason: String| LocalSolverError::RunFailed {
        solver_type: format!("{solver_type:?}"),
        reason,
    };
    let ineq = problem.constraints(solution).map_err(|error| failed(solver_type, error))?;
    const INEQUALITY_FEASIBILITY_TOLERANCE: f64 = 1e-6;
    if ineq.iter().any(|&v| v < -INEQUALITY_FEASIBILITY_TOLERANCE) {
        return Err(violated(format!(
            "solution violates {} nonlinear inequality constraint(s)",
            ineq.iter().filter(|&&v| v < -INEQUALITY_FEASIBILITY_TOLERANCE).count()
        )));
    }
    let eq = problem.nonlinear_equalities(solution).map_err(|error| failed(solver_type, error))?;
    if eq.iter().any(|&v| v.abs() > INEQUALITY_FEASIBILITY_TOLERANCE) {
        return Err(violated("solution violates nonlinear equality constraint(s)".to_string()));
    }
    if let Some((a, b)) = problem.linear_inequalities() {
        let residual = a.dot(solution) - &b;
        if residual.iter().any(|&v| v > INEQUALITY_FEASIBILITY_TOLERANCE) {
            return Err(violated("solution violates linear inequality constraint(s)".to_string()));
        }
    }
    if let Some((a, b)) = problem.linear_equalities() {
        let residual = a.dot(solution) - &b;
        if residual.iter().any(|&v| v.abs() > INEQUALITY_FEASIBILITY_TOLERANCE) {
            return Err(violated("solution violates linear equality constraint(s)".to_string()));
        }
    }
    Ok(())
}
