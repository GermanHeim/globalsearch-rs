//! Tests for the constrained local solvers: SLSQP, Barrier,
//! AugmentedLagrangian, and COBYLA's folded linear/equality blocks.

use globalsearch::local_solver::builders::*;
use globalsearch::local_solver::runner::{LocalSolver, LocalSolverError};
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, LocalSolverType};
use ndarray::{Array1, Array2, array};

// min (x-2)^2 + (y+1)^2 over [0,1]^2. Optimum: (1, 0), f = 2.
#[derive(Clone)]
struct BoundedQuadratic;

impl Problem for BoundedQuadratic {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[0.0, 1.0], [0.0, 1.0]]
    }
}

// min (x-2)^2 + (y+1)^2 s.t. x + y <= 0.5. Optimum: (1.75, -1.25), f = 0.125.
#[derive(Clone)]
struct NonlinearIneq {
    bounded: bool,
}

impl Problem for NonlinearIneq {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        if self.bounded { array![[0.0, 1.0], [0.0, 1.0]] } else { array![[-5.0, 5.0], [-5.0, 5.0]] }
    }
    fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![0.5 - x[0] - x[1]])
    }
    fn constraint_jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![[-1.0, -1.0]])
    }
}

// min x^2 + y^2 s.t. x + y = 1 (nonlinear equality). Optimum: (0.5, 0.5), f = 0.5.
#[derive(Clone)]
struct NonlinearEq;

impl Problem for NonlinearEq {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(x[0].powi(2) + x[1].powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * x[0], 2.0 * x[1]])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[-5.0, 5.0], [-5.0, 5.0]]
    }
    fn nonlinear_equalities(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![x[0] + x[1] - 1.0])
    }
    fn constraint_jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![[1.0, 1.0]])
    }
}

// min (x-2)^2 + (y-2)^2 s.t. x + y <= 1 (linear inequality).
// Optimum: (0.5, 0.5), f = 4.5.
#[derive(Clone)]
struct LinearIneq;

impl Problem for LinearIneq {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[-5.0, 5.0], [-5.0, 5.0]]
    }
    fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        Some((array![[1.0, 1.0]], array![1.0]))
    }
}

// min x^2 + y^2 s.t. x + y = 1 (linear equality). Optimum: (0.5, 0.5), f = 0.5.
#[derive(Clone)]
struct LinearEq;

impl Problem for LinearEq {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok(x[0].powi(2) + x[1].powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * x[0], 2.0 * x[1]])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[-5.0, 5.0], [-5.0, 5.0]]
    }
    fn linear_equalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        Some((array![[1.0, 1.0]], array![1.0]))
    }
}

#[test]
fn slsqp_solves_box_constrained_quadratic() {
    let config = SLSQPBuilder::default().build();
    let solution = LocalSolver::new(BoundedQuadratic, config.solver_type(), config)
        .solve(array![0.25, 0.75])
        .unwrap();
    assert!((solution.objective - 2.0).abs() < 1e-6, "{solution:?}");
    assert!((&solution.point - array![1.0, 0.0]).mapv(f64::abs).sum() < 1e-4);
}

#[test]
fn slsqp_solves_nonlinear_inequality() {
    let config = SLSQPBuilder::default().build();
    let problem = NonlinearIneq { bounded: false };
    let solution =
        LocalSolver::new(problem, config.solver_type(), config).solve(array![0.0, 0.0]).unwrap();
    assert!((solution.objective - 0.125).abs() < 1e-4, "{solution:?}");
    assert!((&solution.point - array![1.75, -1.25]).mapv(f64::abs).sum() < 1e-3);
}

#[test]
fn slsqp_solves_nonlinear_equality() {
    let config = SLSQPBuilder::default().build();
    let solution = LocalSolver::new(NonlinearEq, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap();
    assert!((solution.objective - 0.5).abs() < 1e-4, "{solution:?}");
    assert!((&solution.point - array![0.5, 0.5]).mapv(f64::abs).sum() < 1e-3);
}

#[test]
fn slsqp_solves_linear_blocks() {
    for (problem, expected_point, expected_obj) in
        [("ineq", array![0.5, 0.5], 4.5), ("eq", array![0.5, 0.5], 0.5)]
    {
        let config = SLSQPBuilder::default().build();
        let solution = match problem {
            "ineq" => LocalSolver::new(LinearIneq, config.solver_type(), config)
                .solve(array![0.0, 0.0])
                .unwrap(),
            _ => LocalSolver::new(LinearEq, config.solver_type(), config)
                .solve(array![0.0, 0.0])
                .unwrap(),
        };
        assert!((solution.objective - expected_obj).abs() < 1e-4, "{problem}: {solution:?}");
        assert!(
            (&solution.point - &expected_point).mapv(f64::abs).sum() < 1e-3,
            "{problem}: {solution:?}"
        );
    }
}

#[test]
fn barrier_solves_linear_inequality() {
    let config = BarrierBuilder::default().build();
    let solution =
        LocalSolver::new(LinearIneq, config.solver_type(), config).solve(array![0.0, 0.0]).unwrap();
    assert!((solution.objective - 4.5).abs() < 1e-3, "{solution:?}");
    assert!((&solution.point - array![0.5, 0.5]).mapv(f64::abs).sum() < 1e-2);
    // Box bounds folded into the barrier system: solution respects them.
    assert!(solution.point.iter().all(|&v| (-5.0..=5.0).contains(&v)));
}

#[test]
fn augmented_lagrangian_solves_linear_equality() {
    let config = AugmentedLagrangianBuilder::default().build();
    let solution =
        LocalSolver::new(LinearEq, config.solver_type(), config).solve(array![0.0, 0.0]).unwrap();
    assert!((solution.objective - 0.5).abs() < 1e-3, "{solution:?}");
    assert!((&solution.point - array![0.5, 0.5]).mapv(f64::abs).sum() < 1e-2);
}

#[test]
fn cobyla_folds_linear_and_equality_blocks() {
    use globalsearch::local_solver::builders::COBYLABuilder;
    let config = COBYLABuilder::default().max_iter(1000).build();
    let solution =
        LocalSolver::new(LinearIneq, config.solver_type(), config).solve(array![0.0, 0.0]).unwrap();
    assert!((solution.objective - 4.5).abs() < 0.2, "{solution:?}");

    let config = COBYLABuilder::default().max_iter(1000).build();
    let solution = LocalSolver::new(NonlinearEq, config.solver_type(), config)
        .solve(array![0.2, 0.2])
        .unwrap();
    assert!((solution.objective - 0.5).abs() < 0.2, "{solution:?}");
}

#[test]
fn classic_solvers_reject_linear_blocks() {
    let lbfgs = LBFGSBuilder::default().build();
    let error = LocalSolver::new(LinearIneq, lbfgs.solver_type(), lbfgs)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");
    assert!(error.to_string().contains("SLSQP"), "{error}");

    let lbfgs = LBFGSBuilder::default().build();
    let error =
        LocalSolver::new(LinearEq, lbfgs.solver_type(), lbfgs).solve(array![0.0, 0.0]).unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");

    // Nonlinear equalities are also rejected by the classic solvers.
    let trust = TrustRegionBuilder::default().build();
    let error = LocalSolver::new(NonlinearEq, trust.solver_type(), trust)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");
}

#[test]
fn barrier_and_alm_validate_their_blocks() {
    // Barrier without linear inequalities.
    let config = BarrierBuilder::default().build();
    let error = LocalSolver::new(BoundedQuadratic, config.solver_type(), config)
        .solve(array![0.5, 0.5])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");
    assert!(error.to_string().contains("linear_inequalities"), "{error}");

    // Barrier with equalities.
    let config = BarrierBuilder::default().build();
    let error = LocalSolver::new(LinearEq, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");

    // Barrier with nonlinear constraints.
    let config = BarrierBuilder::default().build();
    let error = LocalSolver::new(NonlinearIneq { bounded: false }, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");

    // ALM without linear equalities.
    let config = AugmentedLagrangianBuilder::default().build();
    let error = LocalSolver::new(LinearIneq, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");

    // Invalid hyperparameters are rejected before evaluation.
    let config = SLSQPBuilder::default().accuracy(-1.0).build();
    let error = LocalSolver::new(BoundedQuadratic, config.solver_type(), config)
        .solve(array![0.5, 0.5])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");

    let config = BarrierBuilder::default().reduction(1.0).build();
    let error = LocalSolver::new(LinearIneq, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{error}");
}

#[test]
fn slsqp_requires_jacobian_when_nonlinearly_constrained() {
    #[derive(Clone)]
    struct NoJacobian;
    impl Problem for NoJacobian {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(x[0].powi(2) + x[1].powi(2))
        }
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![2.0 * x[0], 2.0 * x[1]])
        }
        fn variable_bounds(&self) -> Array2<f64> {
            array![[-5.0, 5.0], [-5.0, 5.0]]
        }
        fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![1.0 - x[0] - x[1]])
        }
    }
    let config = SLSQPBuilder::default().build();
    let error = LocalSolver::new(NoJacobian, config.solver_type(), config)
        .solve(array![0.0, 0.0])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::RunFailed { .. }), "{error}");
    assert!(error.to_string().contains("Jacobian"), "{error}");
}

#[test]
fn solver_names_cover_new_solvers() {
    assert_eq!(LocalSolverType::from_string("SLSQP"), Ok(LocalSolverType::SLSQP));
    assert_eq!(LocalSolverType::from_string("barrier"), Ok(LocalSolverType::Barrier));
    assert_eq!(
        LocalSolverType::from_string("augmented_lagrangian"),
        Ok(LocalSolverType::AugmentedLagrangian)
    );
    for config in [
        SLSQPBuilder::default().build(),
        BarrierBuilder::default().build(),
        AugmentedLagrangianBuilder::default().build(),
    ] {
        let name = config.solver_type();
        assert_eq!(LocalSolverType::from_string(&format!("{name:?}")), Ok(name));
    }
}

#[test]
fn linear_equality_runs_through_global_search() {
    use globalsearch::{oqnlp::OQNLP, types::OQNLPParams};
    // Scatter projects samples onto A x = b, so linear equalities work globally.
    let params = OQNLPParams {
        iterations: 5,
        population_size: 20,
        wait_cycle: 3,
        local_solver_config: SLSQPBuilder::default().build(),
        ..Default::default()
    };
    let mut optimizer = OQNLP::new(LinearEq, params).unwrap();
    let solution = optimizer.run().unwrap();
    assert!((solution[0].objective - 0.5).abs() < 1e-4, "{solution:?}");
    assert!((solution[0].point[0] + solution[0].point[1] - 1.0).abs() < 1e-6, "{solution:?}");
}

#[test]
fn slsqp_runs_through_global_search() {
    use globalsearch::{oqnlp::OQNLP, types::OQNLPParams};
    let problem = NonlinearIneq { bounded: true };
    // Bounded box keeps scatter feasible; the inequality has volume.
    let params = OQNLPParams {
        iterations: 5,
        population_size: 20,
        wait_cycle: 3,
        local_solver_config: SLSQPBuilder::default().build(),
        ..Default::default()
    };
    let mut optimizer = OQNLP::new(problem, params).unwrap();
    let solution = optimizer.run().unwrap();
    // Bounded optimum of (x-2)^2 + (y+1)^2 over [0,1]^2 with x + y <= 0.5:
    // (0.75, -0.25) is outside the box; the box corner (0.5, 0) gives 2.5... the
    // feasible box corner (1, 0) violates x+y<=0.5; optimum is (0.5, 0) f=3.25.
    assert!(solution[0].objective < 3.26, "{solution:?}");
    assert!(solution[0].point[0] + solution[0].point[1] <= 0.5 + 1e-6, "{solution:?}");
}
