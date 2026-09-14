//! Adapters for the optional basin local solvers.

use super::builders::{LocalSolverConfig, TrustRegionRadiusMethod};
use super::runner::LocalSolverError;
use crate::problem::Problem;
use crate::types::{EvaluationError, LocalSolution, LocalSolverType};
use basin::{BasicSimplexState, BasicState, DenseMatrix, LbfgsState};
use ndarray::Array1;

struct BasinProblem<'a, P> {
    problem: &'a P,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl<P: Problem> basin::CostFunction for BasinProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = EvaluationError;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, EvaluationError> {
        self.problem.objective(&Array1::from_vec(x.clone()))
    }
}

impl<P: Problem> basin::Gradient for BasinProblem<'_, P> {
    type Gradient = Vec<f64>;

    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, EvaluationError> {
        let gradient = self.problem.gradient(&Array1::from_vec(x.clone()))?;
        if gradient.len() != x.len() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(EvaluationError::InvalidInput {
                reason: format!("Gradient must contain {} finite values.", x.len()),
            });
        }
        Ok(gradient.to_vec())
    }
}

impl<P: Problem> basin::Hessian for BasinProblem<'_, P> {
    type Hessian = DenseMatrix;

    fn hessian(&self, x: &Vec<f64>) -> Result<DenseMatrix, EvaluationError> {
        let hessian = self.problem.hessian(&Array1::from_vec(x.clone()))?;
        if hessian.dim() != (x.len(), x.len()) || hessian.iter().any(|v| !v.is_finite()) {
            return Err(EvaluationError::InvalidInput {
                reason: format!("Hessian must be a finite {} by {} matrix.", x.len(), x.len()),
            });
        }
        // Index by coordinates because ndarray matrices may use nonstandard strides.
        Ok(DenseMatrix::from_fn(x.len(), x.len(), |i, j| hessian[[i, j]]))
    }
}

impl<P: Problem> basin::BoxConstraints for BasinProblem<'_, P> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

fn invalid(solver_type: &LocalSolverType, reason: impl Into<String>) -> LocalSolverError {
    LocalSolverError::InvalidBasinConfig {
        solver_type: format!("{solver_type:?}"),
        reason: reason.into(),
    }
}

fn positive(solver: &LocalSolverType, name: &str, value: f64) -> Result<(), LocalSolverError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(solver, format!("`{name}` must be finite and positive.")))
    }
}

fn tolerance(
    solver: &LocalSolverType,
    name: &str,
    value: Option<f64>,
) -> Result<(), LocalSolverError> {
    if value.is_none_or(|v| v.is_finite() && v >= 0.0) {
        Ok(())
    } else {
        Err(invalid(solver, format!("`{name}` must be finite and nonnegative, or None.")))
    }
}

fn finish<S: basin::State<Param = Vec<f64>, Float = f64>>(
    result: basin::OptimizationResult<S>,
    solver: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if result.reason.is_failure() {
        return Err(LocalSolverError::RunFailed {
            solver_type: format!("{solver:?}"),
            reason: format!("solver terminated with {:?}", result.reason),
        });
    }
    if !result.best_cost().is_finite() || result.best_param().iter().any(|v| !v.is_finite()) {
        return Err(LocalSolverError::NoSolution {
            solver_type: format!("{solver:?}"),
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

fn simplex<P>(
    problem: &BasinProblem<'_, P>,
    start: &[f64],
    delta: f64,
    bounded: bool,
) -> Vec<Vec<f64>> {
    let mut vertices = vec![start.to_vec()];
    for i in 0..start.len() {
        let mut vertex = start.to_vec();
        if bounded {
            // Choose the side with more room so a boundary start does not collapse the simplex.
            let up = problem.upper[i] - start[i];
            let down = start[i] - problem.lower[i];
            vertex[i] = if up >= down {
                (start[i] + delta.min(up)).min(problem.upper[i])
            } else {
                (start[i] - delta.min(down)).max(problem.lower[i])
            };
            if vertex[i] == start[i] {
                vertex[i] = if up >= down { start[i].next_up() } else { start[i].next_down() };
            }
        } else {
            vertex[i] += delta;
        }
        vertices.push(vertex);
    }
    vertices
}

pub(super) fn solve<P: Problem>(
    problem: &P,
    initial: Array1<f64>,
    config: &LocalSolverConfig,
    solver_type: &LocalSolverType,
    track: bool,
) -> Result<(LocalSolution, u64), LocalSolverError> {
    if &config.solver_type() != solver_type {
        return Err(invalid(solver_type, "Solver type and configuration do not match."));
    }
    let bounded = matches!(
        solver_type,
        LocalSolverType::BasinLBFGSB
            | LocalSolverType::BasinBoundedNelderMead
            | LocalSolverType::BasinBOBYQA
    );
    if initial.is_empty() || initial.iter().any(|v| !v.is_finite()) {
        return Err(invalid(solver_type, "The starting point must be nonempty and finite."));
    }
    let mut adapter = BasinProblem { problem, lower: Vec::new(), upper: Vec::new() };
    let mut start = initial.to_vec();
    if bounded {
        let bounds = problem.variable_bounds();
        if bounds.dim() != (start.len(), 2) {
            return Err(invalid(
                solver_type,
                format!("Bounds must have shape ({}, 2).", start.len()),
            ));
        }
        for i in 0..start.len() {
            let (lower, upper) = (bounds[[i, 0]], bounds[[i, 1]]);
            if !lower.is_finite()
                || !upper.is_finite()
                || lower >= upper
                || !(upper - lower).is_finite()
            {
                return Err(invalid(
                    solver_type,
                    "Bounds must be finite, with lower < upper and a finite width.",
                ));
            }
            adapter.lower.push(lower);
            adapter.upper.push(upper);
            start[i] = start[i].clamp(lower, upper);
        }
    }
    let failed = |error: EvaluationError| LocalSolverError::RunFailed {
        solver_type: format!("{solver_type:?}"),
        reason: error.to_string(),
    };
    if !problem.constraints(&Array1::from_vec(start.clone())).map_err(failed)?.is_empty() {
        return Err(invalid(
            solver_type,
            "Nonlinear constraints are unsupported by this solver. Use COBYLA instead.",
        ));
    }
    macro_rules! run {
        ($solver:expr, $state:expr, $max_iter:expr) => {{
            let result = basin::Executor::new(adapter, $solver, $state)
                .max_iter(*$max_iter)
                .run()
                .map_err(failed)?;
            finish(result, solver_type, track)
        }};
    }
    match config {
        LocalSolverConfig::BasinLBFGS {
            max_iter,
            tolerance_grad,
            tolerance_cost,
            history_size,
        } => {
            tolerance(solver_type, "tolerance_grad", *tolerance_grad)?;
            tolerance(solver_type, "tolerance_cost", *tolerance_cost)?;
            if *history_size == 0 {
                return Err(invalid(solver_type, "`history_size` must be positive."));
            }
            let solver = basin::Lbfgsb::new()
                .unbounded()
                .with_absolute_gradient_tolerance(*tolerance_grad)
                .with_absolute_cost_change_tolerance(*tolerance_cost);
            run!(solver, LbfgsState::new(start, *history_size), max_iter)
        }
        LocalSolverConfig::BasinLBFGSB {
            max_iter,
            tolerance_projected_grad,
            tolerance_cost,
            history_size,
        } => {
            tolerance(solver_type, "tolerance_projected_grad", *tolerance_projected_grad)?;
            tolerance(solver_type, "tolerance_cost", *tolerance_cost)?;
            if *history_size == 0 {
                return Err(invalid(solver_type, "`history_size` must be positive."));
            }
            let solver = basin::Lbfgsb::new()
                .with_absolute_projected_gradient_tolerance(*tolerance_projected_grad)
                .with_absolute_cost_change_tolerance(*tolerance_cost);
            run!(solver, LbfgsState::new(start, *history_size), max_iter)
        }
        LocalSolverConfig::BasinGradientDescent { max_iter, tolerance_grad, tolerance_cost } => {
            tolerance(solver_type, "tolerance_grad", *tolerance_grad)?;
            tolerance(solver_type, "tolerance_cost", *tolerance_cost)?;
            let solver = basin::GradientDescent::with_line_search(basin::MoreThuente::new())
                .with_absolute_gradient_tolerance(*tolerance_grad)
                .with_absolute_cost_change_tolerance(*tolerance_cost);
            run!(solver, BasicState::new(start), max_iter)
        }
        LocalSolverConfig::BasinTrustRegion {
            max_iter,
            tolerance_grad,
            trust_region_radius_method,
            radius,
            max_radius,
            eta,
        } => {
            tolerance(solver_type, "tolerance_grad", *tolerance_grad)?;
            positive(solver_type, "radius", *radius)?;
            positive(solver_type, "max_radius", *max_radius)?;
            if max_radius < radius || !eta.is_finite() || !(0.0..0.25).contains(eta) {
                return Err(invalid(
                    solver_type,
                    "Require max_radius >= radius and 0 <= eta < 0.25.",
                ));
            }
            macro_rules! trust_region {
                ($method:expr) => {{
                    let solver = basin::TrustRegion::with_subproblem($method)
                        .with_radius(*radius)
                        .with_max_radius(*max_radius)
                        .with_eta(*eta)
                        .with_absolute_gradient_tolerance(*tolerance_grad);
                    run!(solver, BasicState::new(start), max_iter)
                }};
            }
            match trust_region_radius_method {
                TrustRegionRadiusMethod::Cauchy => trust_region!(basin::CauchyPoint),
                TrustRegionRadiusMethod::Steihaug => trust_region!(basin::Steihaug::new()),
            }
        }
        LocalSolverConfig::BasinNelderMead {
            max_iter,
            simplex_delta,
            tolerance_simplex,
            tolerance_cost,
        }
        | LocalSolverConfig::BasinBoundedNelderMead {
            max_iter,
            simplex_delta,
            tolerance_simplex,
            tolerance_cost,
        } => {
            positive(solver_type, "simplex_delta", *simplex_delta)?;
            tolerance(solver_type, "tolerance_simplex", *tolerance_simplex)?;
            tolerance(solver_type, "tolerance_cost", *tolerance_cost)?;
            let vertices = simplex(&adapter, &start, *simplex_delta, bounded);
            if vertices.iter().flatten().any(|v| !v.is_finite())
                || vertices.iter().skip(1).any(|v| v == &start)
            {
                return Err(invalid(
                    solver_type,
                    "`simplex_delta` must produce distinct finite vertices.",
                ));
            }
            let state = BasicSimplexState::from_simplex(vertices);
            if bounded {
                let solver = basin::NelderMead::new()
                    .projected()
                    .with_absolute_simplex_size_tolerance(*tolerance_simplex)
                    .with_absolute_simplex_cost_tolerance(*tolerance_cost);
                run!(solver, state, max_iter)
            } else {
                let solver = basin::NelderMead::new()
                    .with_absolute_simplex_size_tolerance(*tolerance_simplex)
                    .with_absolute_simplex_cost_tolerance(*tolerance_cost);
                run!(solver, state, max_iter)
            }
        }
        LocalSolverConfig::BasinBOBYQA {
            max_iter,
            initial_radius,
            final_radius,
            interpolation_points,
        } => {
            positive(solver_type, "initial_radius", *initial_radius)?;
            positive(solver_type, "final_radius", *final_radius)?;
            if final_radius >= initial_radius {
                return Err(invalid(solver_type, "Require final_radius < initial_radius."));
            }
            let n = start.len();
            if interpolation_points.is_some_and(|m| m < 2 * n + 1 || m > (n + 1) * (n + 2) / 2) {
                return Err(invalid(
                    solver_type,
                    "`interpolation_points` must lie between 2n+1 and (n+1)(n+2)/2.",
                ));
            }
            let min_width = adapter
                .lower
                .iter()
                .zip(&adapter.upper)
                .map(|(lower, upper)| upper - lower)
                .fold(f64::INFINITY, f64::min);
            if *initial_radius > min_width / 2.0 {
                let adjusted_initial = min_width / 4.0;
                let adjusted_final = final_radius.min(0.1 * adjusted_initial);
                if adjusted_final <= 0.0 || adjusted_final >= adjusted_initial {
                    return Err(invalid(
                        solver_type,
                        "Bounds are too narrow to represent positive BOBYQA radii.",
                    ));
                }
            }
            let mut solver = basin::Bobyqa::new()
                .with_initial_radius(*initial_radius)
                .with_final_radius(*final_radius);
            if let Some(npt) = interpolation_points {
                solver = solver.with_npt(*npt);
            }
            run!(solver, basin::BobyqaState::new(start), max_iter)
        }
        _ => Err(invalid(solver_type, "Expected a basin solver configuration.")),
    }
}
