use globalsearch::local_solver::builders::*;
use globalsearch::local_solver::runner::{LocalSolver, LocalSolverError};
use globalsearch::problem::Problem;
use globalsearch::types::EvaluationError;
use ndarray::{Array1, Array2, array};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone)]
struct Quadratic {
    bounded: bool,
    constraints: bool,
    calls: Arc<AtomicU64>,
}

impl Problem for Quadratic {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.bounded {
            assert!(x.iter().all(|v| (0.0..=1.0).contains(v)), "{x:?}");
        }
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        if self.bounded {
            assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        Ok(array![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)])
    }
    fn hessian(&self, _: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![[2.0, 0.0], [0.0, 2.0]])
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[0.0, 1.0], [0.0, 1.0]]
    }
    fn constraints(&self, _: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(if self.constraints { array![1.0] } else { array![] })
    }
}

fn configs() -> Vec<(LocalSolverConfig, bool)> {
    vec![
        (LBFGSBuilder::default().build(), false),
        (GradientDescentBuilder::default().build(), false),
        (TrustRegionBuilder::default().build(), false),
        (TrustRegionBuilder::default().method(TrustRegionRadiusMethod::Cauchy).build(), false),
        (NelderMeadBuilder::default().build(), false),
        (LBFGSBBuilder::default().build(), true),
        (BoundedNelderMeadBuilder::default().build(), true),
        (BOBYQABuilder::default().build(), true),
    ]
}

#[test]
fn solves_quadratics_and_tracks_only_objective_calls() {
    for (config, bounded) in configs() {
        let calls = Arc::new(AtomicU64::new(0));
        let problem = Quadratic { bounded, constraints: false, calls: calls.clone() };
        let name = config.solver_type();
        let solver = LocalSolver::new(problem, name.clone(), config);
        let (solution, count) = solver.solve_with_tracking(array![0.25, 0.75], true).unwrap();
        let target = if bounded { array![1.0, 0.0] } else { array![2.0, -1.0] };
        assert!((&solution.point - &target).mapv(f64::abs).sum() < 1e-4, "{name:?}: {solution:?}");
        assert_eq!(count, calls.load(Ordering::Relaxed), "{name:?}");
        assert!(count > 0);
        assert_eq!(solver.solve_with_tracking(array![0.5, 0.5], false).unwrap().1, 0);
    }
}

#[test]
fn bounded_solvers_handle_boundary_and_infeasible_starts() {
    for (config, _) in configs().into_iter().filter(|(_, bounded)| *bounded) {
        for start in [array![0.0, 0.0], array![1.0, 1.0], array![-3.0, 4.0]] {
            let problem = Quadratic { bounded: true, constraints: false, calls: Arc::default() };
            let name = config.solver_type();
            let solution =
                LocalSolver::new(problem, name.clone(), config.clone()).solve(start).unwrap();
            assert!((solution.objective - 2.0).abs() < 1e-5, "{name:?}: {solution:?}");
        }
    }
}

#[test]
fn nonlinear_constraints_are_rejected_before_objective_evaluation() {
    for (config, bounded) in configs() {
        let calls = Arc::new(AtomicU64::new(0));
        let problem = Quadratic { bounded, constraints: true, calls: calls.clone() };
        let solver = LocalSolver::new(problem, config.solver_type(), config);
        let error = solver.solve(array![0.5, 0.5]).unwrap_err();
        assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
        assert!(error.to_string().contains("COBYLA"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn iteration_limit_returns_a_finite_initial_solution() {
    let config = LBFGSBuilder::default().max_iter(0).build();
    let problem = Quadratic { bounded: false, constraints: false, calls: Arc::default() };
    let solution =
        LocalSolver::new(problem, config.solver_type(), config).solve(array![0.5, 0.5]).unwrap();
    assert_eq!(solution.point, array![0.5, 0.5]);
    assert!(solution.objective.is_finite());
}

#[test]
fn invalid_configurations_return_errors() {
    let configs = [
        LBFGSBuilder::new().history_size(0).build(),
        LBFGSBuilder::new().tolerance_grad(-1.0).build(),
        LBFGSBuilder::new().tolerance_cost(f64::NAN).build(),
        LBFGSBBuilder::new().history_size(0).build(),
        LBFGSBBuilder::new().tolerance_projected_grad(f64::INFINITY).build(),
        GradientDescentBuilder::new().tolerance_grad(f64::NAN).build(),
        NelderMeadBuilder::new().simplex_delta(0.0).build(),
        BoundedNelderMeadBuilder::new().tolerance_simplex(-1.0).build(),
        TrustRegionBuilder::new().radius(-1.0).build(),
        TrustRegionBuilder::new().radius(2.0).max_radius(1.0).build(),
        TrustRegionBuilder::new().eta(0.25).build(),
        BOBYQABuilder::new().initial_radius(0.0).build(),
        BOBYQABuilder::new().final_radius(2.0).build(),
        BOBYQABuilder::new().interpolation_points(Some(4)).build(),
        BOBYQABuilder::new().interpolation_points(Some(7)).build(),
    ];
    for config in configs {
        let problem = Quadratic { bounded: false, constraints: false, calls: Arc::default() };
        let name = config.solver_type();
        let error =
            LocalSolver::new(problem, name.clone(), config).solve(array![0.5, 0.5]).unwrap_err();
        assert!(matches!(error, LocalSolverError::InvalidConfig { .. }), "{name:?}: {error}");
    }
}

#[test]
fn mismatched_solver_configuration_is_rejected_before_evaluation() {
    use globalsearch::types::LocalSolverType;

    let calls = Arc::new(AtomicU64::new(0));
    let problem = Quadratic { bounded: false, constraints: false, calls: calls.clone() };
    let config = LBFGSBuilder::new().build();
    let error = LocalSolver::new(problem, LocalSolverType::GradientDescent, config)
        .solve(array![0.5, 0.5])
        .unwrap_err();

    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
    assert!(error.to_string().contains("Solver type and configuration do not match"));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn nelder_mead_rejects_overflowing_or_collapsed_simplexes_before_evaluation() {
    for (start, delta) in [(f64::MAX, f64::MAX), (1e20, 0.1)] {
        let calls = Arc::new(AtomicU64::new(0));
        let problem = Quadratic { bounded: false, constraints: false, calls: calls.clone() };
        let config = NelderMeadBuilder::new().simplex_delta(delta).build();
        let error = LocalSolver::new(problem, config.solver_type(), config)
            .solve(array![start, start])
            .unwrap_err();

        assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
        assert!(error.to_string().contains("distinct finite vertices"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[derive(Clone)]
struct Fault(&'static str);

impl Problem for Fault {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        match self.0 {
            "objective" => Err(EvaluationError::ObjectiveFunctionEvaluationFailed {
                reason: "objective failure".into(),
            }),
            "infinite" => Ok(f64::INFINITY),
            _ => Ok(x.dot(x)),
        }
    }
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        match self.0 {
            "gradient" => Err(EvaluationError::GradientNotImplemented),
            "gradient_shape" => Ok(array![]),
            "overflow_gradient" => Ok(array![f64::MAX, f64::MAX]),
            _ => Ok(2.0 * x),
        }
    }
    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        match self.0 {
            "hessian" => Err(EvaluationError::HessianNotImplemented),
            "hessian_shape" => Ok(Array2::zeros((0, 0))),
            _ => Ok(2.0 * Array2::eye(x.len())),
        }
    }
    fn variable_bounds(&self) -> Array2<f64> {
        match self.0 {
            "bounds_shape" => array![[0.0, 1.0]],
            "bounds_nan" => array![[f64::NAN, 1.0], [0.0, 1.0]],
            "bounds_equal" => array![[1.0, 1.0], [0.0, 1.0]],
            "bounds_subnormal" => array![[0.0, f64::from_bits(1)], [0.0, 1.0]],
            _ => array![[-1.0, 1.0], [-1.0, 1.0]],
        }
    }
    fn constraints(&self, _: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        if self.0 == "constraints" {
            Err(EvaluationError::ConstraintEvaluationFailed {
                index: 0,
                reason: "constraint failure".into(),
            })
        } else {
            Ok(array![])
        }
    }
}

#[test]
fn callbacks_and_bad_derivative_shapes_propagate_without_panicking() {
    let cases = [
        ("objective", NelderMeadBuilder::new().build(), "objective failure"),
        ("constraints", LBFGSBBuilder::new().build(), "constraint failure"),
        ("gradient", LBFGSBuilder::new().build(), "Gradient not implemented"),
        ("gradient_shape", GradientDescentBuilder::new().build(), "Gradient must"),
        ("hessian", TrustRegionBuilder::new().build(), "Hessian not implemented"),
        ("hessian_shape", TrustRegionBuilder::new().build(), "Hessian must"),
    ];
    for (fault, config, message) in cases {
        let error = LocalSolver::new(Fault(fault), config.solver_type(), config)
            .solve(array![0.5, 0.5])
            .unwrap_err();
        assert!(matches!(error, LocalSolverError::RunFailed { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[test]
fn invalid_bounds_and_starts_return_errors() {
    for (config, _) in configs().into_iter().filter(|(_, bounded)| *bounded) {
        for fault in ["bounds_shape", "bounds_nan", "bounds_equal"] {
            let error = LocalSolver::new(Fault(fault), config.solver_type(), config.clone())
                .solve(array![0.5, 0.5])
                .unwrap_err();
            assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
        }
    }
    for (config, _) in configs() {
        for start in [array![], array![f64::NAN, 0.5], array![f64::INFINITY, 0.5]] {
            let error = LocalSolver::new(Fault(""), config.solver_type(), config.clone())
                .solve(start)
                .unwrap_err();
            assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
        }
    }
}

#[test]
fn infinite_objective_is_not_a_solution() {
    let config = NelderMeadBuilder::new().max_iter(0).build();
    let error = LocalSolver::new(Fault("infinite"), config.solver_type(), config)
        .solve(array![0.5, 0.5])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::NoSolution { .. }));
}

#[derive(Clone)]
struct NarrowBox;
impl Problem for NarrowBox {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        assert!((0.0..=1e-4).contains(&x[0]));
        Ok((x[0] - 4e-5).powi(2))
    }
    fn variable_bounds(&self) -> Array2<f64> {
        array![[0.0, 1e-4]]
    }
}

#[test]
fn bobyqa_adapts_to_a_narrow_one_dimensional_box() {
    let config = BOBYQABuilder::new().final_radius(1e-10).build();
    let solution =
        LocalSolver::new(NarrowBox, config.solver_type(), config).solve(array![0.0]).unwrap();
    assert!((solution.point[0] - 4e-5).abs() < 1e-8, "{solution:?}");
}

#[test]
fn bobyqa_rejects_bounds_that_underflow_its_radii() {
    let config = BOBYQABuilder::new().build();
    let error = LocalSolver::new(Fault("bounds_subnormal"), config.solver_type(), config)
        .solve(array![0.0, 0.5])
        .unwrap_err();

    assert!(matches!(error, LocalSolverError::InvalidConfig { .. }));
    assert!(error.to_string().contains("too narrow to represent positive BOBYQA radii"));
}

#[test]
fn bobyqa_accepts_both_interpolation_set_size_limits() {
    for interpolation_points in [5, 6] {
        let problem = Quadratic { bounded: true, constraints: false, calls: Arc::default() };
        let config = BOBYQABuilder::new()
            .initial_radius(0.25)
            .interpolation_points(Some(interpolation_points))
            .build();
        let solution = LocalSolver::new(problem, config.solver_type(), config)
            .solve(array![0.5, 0.5])
            .unwrap();

        assert!((solution.objective - 2.0).abs() < 1e-6, "{solution:?}");
        assert!((&solution.point - array![1.0, 0.0]).mapv(f64::abs).sum() < 1e-4);
    }
}

#[test]
fn solver_names_are_unambiguous() {
    use globalsearch::types::LocalSolverType;
    for (config, _) in configs() {
        let name = config.solver_type();
        let compact = format!("{name:?}");
        assert_eq!(LocalSolverType::from_string(&compact), Ok(name.clone()));
        assert_eq!(
            LocalSolverType::from_string(&compact.replace('_', "-").to_lowercase()),
            Ok(name.clone())
        );
        // Legacy `basin_` prefix is no longer accepted.
        let prefixed = format!("basin_{compact}");
        assert!(LocalSolverType::from_string(&prefixed).is_err());
    }
    assert!(LocalSolverType::from_string("basin_lbfgs").is_err());
    assert!(LocalSolverType::from_string("steepest_descent").is_err());
    assert!(LocalSolverType::from_string("newton_cg").is_err());
}

#[test]
fn each_solver_runs_through_global_search() {
    use globalsearch::{oqnlp::OQNLP, types::OQNLPParams};
    for (config, bounded) in configs() {
        for _parallel in [false, true] {
            let problem = Quadratic { bounded, constraints: false, calls: Arc::default() };
            let params = OQNLPParams {
                iterations: 10,
                population_size: 30,
                wait_cycle: 5,
                local_solver_config: config.clone(),
                ..Default::default()
            };
            let optimizer = OQNLP::new(problem, params).unwrap();
            #[cfg(feature = "rayon")]
            let optimizer = optimizer.parallel(_parallel);
            let mut optimizer = optimizer;
            let solution = optimizer.run().unwrap();
            assert!(
                (solution[0].objective - if bounded { 2.0 } else { 0.0 }).abs() < 1e-6,
                "{:?}",
                config.solver_type()
            );
        }
    }
}

#[cfg(feature = "checkpointing")]
#[test]
fn serialized_configs_preserve_behavior() {
    for (config, bounded) in configs() {
        let bytes = bincode::serde::encode_to_vec(&config, bincode::config::legacy()).unwrap();
        let (restored, consumed): (LocalSolverConfig, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(restored.solver_type(), config.solver_type());
        let problem = Quadratic { bounded, constraints: false, calls: Arc::default() };
        let solution = LocalSolver::new(problem, restored.solver_type(), restored)
            .solve(array![0.5, 0.5])
            .unwrap();
        assert!((solution.objective - if bounded { 2.0 } else { 0.0 }).abs() < 1e-6);
    }
}

#[cfg(feature = "checkpointing")]
#[test]
fn existing_cobyla_encoding_is_unchanged() {
    // COBYLA is the first variant, so its encoding is stable now that the
    // backend set is fixed.
    let mut bytes = 0_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&123_u64.to_le_bytes());
    for value in [0.5_f64, 1e-6, 1e-8, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    let config = COBYLABuilder::default().max_iter(123).build();
    assert_eq!(bincode::serde::encode_to_vec(&config, bincode::config::legacy()).unwrap(), bytes);
    let (restored, _): (LocalSolverConfig, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::legacy()).unwrap();
    assert_eq!(restored.solver_type(), config.solver_type());
}

#[cfg(feature = "checkpointing")]
#[test]
fn global_search_resumes_basin_checkpoints() {
    use globalsearch::{
        checkpoint::CheckpointManager,
        oqnlp::OQNLP,
        types::{CheckpointConfig, OQNLPParams},
    };
    struct Directory(std::path::PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!(
        "globalsearch-basin-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let directory = Directory(path);
    for (index, (config, bounded)) in configs().into_iter().enumerate() {
        let checkpoint = CheckpointConfig {
            checkpoint_dir: directory.0.clone(),
            checkpoint_name: format!("solver-{index}"),
            save_frequency: 1,
            auto_resume: true,
            ..Default::default()
        };
        let problem = Quadratic { bounded, constraints: false, calls: Arc::default() };
        let params = OQNLPParams {
            iterations: 10,
            population_size: 30,
            wait_cycle: 5,
            local_solver_config: config.clone(),
            ..Default::default()
        };
        let mut optimizer = OQNLP::new(problem.clone(), params.clone())
            .unwrap()
            .with_checkpointing(checkpoint.clone())
            .unwrap();
        let original = optimizer.run().unwrap();
        let saved =
            CheckpointManager::new(checkpoint.clone()).unwrap().load_latest_checkpoint().unwrap();
        assert_eq!(saved.params.local_solver_type(), config.solver_type());
        let mut resumed =
            OQNLP::new(problem, params).unwrap().with_checkpointing(checkpoint).unwrap();
        assert!(resumed.try_resume_from_checkpoint().unwrap());
        let result = resumed.run().unwrap();
        assert!((result[0].objective - original[0].objective).abs() < 1e-8);
    }
}

#[test]
fn trust_region_accepts_nonstandard_hessian_layout() {
    struct StridedHessian;
    impl Problem for StridedHessian {
        fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
            Ok(2.0 * x[0].powi(2) + 2.0 * x[0] * x[1] + 3.0 * x[1].powi(2))
        }
        fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![4.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 6.0 * x[1]])
        }
        fn hessian(&self, _: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
            let storage = array![[4.0, 0.0, 2.0], [0.0, 0.0, 0.0], [2.0, 0.0, 6.0]];
            Ok(storage.slice_move(ndarray::s![..;2, ..;2]))
        }
        fn variable_bounds(&self) -> Array2<f64> {
            array![[-5.0, 5.0], [-5.0, 5.0]]
        }
    }
    let config = TrustRegionBuilder::new().build();
    let solution = LocalSolver::new(StridedHessian, config.solver_type(), config)
        .solve(array![3.0, -2.0])
        .unwrap();
    assert!(solution.objective < 1e-10);
}

#[test]
fn a_positive_iteration_limit_is_respected() {
    let calls = Arc::new(AtomicU64::new(0));
    let config =
        GradientDescentBuilder::new().max_iter(1).tolerance_grad(None).tolerance_cost(None).build();
    let problem = Quadratic { bounded: false, constraints: false, calls: calls.clone() };
    let (solution, count) = LocalSolver::new(problem, config.solver_type(), config)
        .solve_with_tracking(array![10.0, 10.0], true)
        .unwrap();
    assert!(solution.objective < 185.0);
    assert_eq!(count, calls.load(Ordering::Relaxed));
    assert!(
        count <= 21,
        "One iteration should use at most the default 20 line-search evaluations plus initialization."
    );
}

#[test]
fn solver_failure_termination_is_reported() {
    let config = LBFGSBuilder::new().build();
    let error = LocalSolver::new(Fault("overflow_gradient"), config.solver_type(), config)
        .solve(array![0.5, 0.5])
        .unwrap_err();
    assert!(matches!(error, LocalSolverError::RunFailed { .. }));
    assert!(error.to_string().contains("terminated"));
}
