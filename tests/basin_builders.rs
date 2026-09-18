use globalsearch::local_solver::builders::*;

#[test]
fn constructors_use_documented_defaults() {
    let cases = [
        (
            [
                LBFGSBuilder::new().build(),
                LBFGSBuilder::default().build(),
                LocalSolverConfig::lbfgs().build(),
            ],
            "LBFGS { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None, history_size: 10 }",
        ),
        (
            [
                GradientDescentBuilder::new().build(),
                GradientDescentBuilder::default().build(),
                LocalSolverConfig::gradient_descent().build(),
            ],
            "GradientDescent { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None }",
        ),
        (
            [
                TrustRegionBuilder::new().build(),
                TrustRegionBuilder::default().build(),
                LocalSolverConfig::trust_region().build(),
            ],
            "TrustRegion { max_iter: 1000, tolerance_grad: Some(1e-6), trust_region_radius_method: Steihaug, radius: 1.0, max_radius: 100.0, eta: 0.125 }",
        ),
        (
            [
                NelderMeadBuilder::new().build(),
                NelderMeadBuilder::default().build(),
                LocalSolverConfig::nelder_mead().build(),
            ],
            "NelderMead { max_iter: 1000, simplex_delta: 0.1, tolerance_simplex: Some(1e-6), tolerance_cost: Some(1e-8) }",
        ),
        (
            [
                LBFGSBBuilder::new().build(),
                LBFGSBBuilder::default().build(),
                LocalSolverConfig::lbfgsb().build(),
            ],
            "LBFGSB { max_iter: 1000, tolerance_projected_grad: Some(1e-6), tolerance_cost: None, history_size: 10 }",
        ),
        (
            [
                BoundedNelderMeadBuilder::new().build(),
                BoundedNelderMeadBuilder::default().build(),
                LocalSolverConfig::bounded_nelder_mead().build(),
            ],
            "BoundedNelderMead { max_iter: 1000, simplex_delta: 0.1, tolerance_simplex: Some(1e-6), tolerance_cost: Some(1e-8) }",
        ),
        (
            [
                BOBYQABuilder::new().build(),
                BOBYQABuilder::default().build(),
                LocalSolverConfig::bobyqa().build(),
            ],
            "BOBYQA { max_iter: 1000, initial_radius: 1.0, final_radius: 1e-6, interpolation_points: None }",
        ),
        (
            [
                SLSQPBuilder::new().build(),
                SLSQPBuilder::default().build(),
                LocalSolverConfig::slsqp().build(),
            ],
            "SLSQP { max_iter: 1000, accuracy: Some(1e-6), max_subproblem_iter: None }",
        ),
        (
            [
                BarrierBuilder::new().build(),
                BarrierBuilder::default().build(),
                LocalSolverConfig::barrier().build(),
            ],
            "Barrier { max_iter: 100, mu0: 1.0, reduction: 10.0, duality_gap_tol: 1e-8, inner_max_iter: 50 }",
        ),
        (
            [
                AugmentedLagrangianBuilder::new().build(),
                AugmentedLagrangianBuilder::default().build(),
                LocalSolverConfig::augmented_lagrangian().build(),
            ],
            "AugmentedLagrangian { max_iter: 100, rho0: 10.0, rho_increase: 10.0, feasibility_decrease: 0.25, feasibility_tol: 1e-8, inner_max_iter: 50 }",
        ),
    ];

    for (configs, expected) in cases {
        for config in configs {
            assert_eq!(format!("{config:?}"), expected);
        }
    }
}

#[test]
fn debug_output_preserves_all_custom_settings() {
    let cases = [
        (
            LocalSolverConfig::lbfgs()
                .max_iter(42)
                .tolerance_grad(0.01)
                .tolerance_cost(Some(0.02))
                .history_size(7)
                .build(),
            "LBFGS { max_iter: 42, tolerance_grad: Some(0.01), tolerance_cost: Some(0.02), history_size: 7 }",
        ),
        (
            LocalSolverConfig::gradient_descent()
                .max_iter(43)
                .tolerance_grad(Some(0.03))
                .tolerance_cost(0.04)
                .build(),
            "GradientDescent { max_iter: 43, tolerance_grad: Some(0.03), tolerance_cost: Some(0.04) }",
        ),
        (
            LocalSolverConfig::trust_region()
                .max_iter(44)
                .tolerance_grad(0.05)
                .method(TrustRegionRadiusMethod::Cauchy)
                .radius(0.5)
                .max_radius(5.0)
                .eta(0.2)
                .build(),
            "TrustRegion { max_iter: 44, tolerance_grad: Some(0.05), trust_region_radius_method: Cauchy, radius: 0.5, max_radius: 5.0, eta: 0.2 }",
        ),
        (
            LocalSolverConfig::nelder_mead()
                .max_iter(45)
                .simplex_delta(0.2)
                .tolerance_simplex(0.06)
                .tolerance_cost(Some(0.07))
                .build(),
            "NelderMead { max_iter: 45, simplex_delta: 0.2, tolerance_simplex: Some(0.06), tolerance_cost: Some(0.07) }",
        ),
        (
            LocalSolverConfig::lbfgsb()
                .max_iter(46)
                .tolerance_projected_grad(Some(0.08))
                .tolerance_cost(0.09)
                .history_size(8)
                .build(),
            "LBFGSB { max_iter: 46, tolerance_projected_grad: Some(0.08), tolerance_cost: Some(0.09), history_size: 8 }",
        ),
        (
            LocalSolverConfig::bounded_nelder_mead()
                .max_iter(47)
                .simplex_delta(0.3)
                .tolerance_simplex(Some(0.11))
                .tolerance_cost(0.12)
                .build(),
            "BoundedNelderMead { max_iter: 47, simplex_delta: 0.3, tolerance_simplex: Some(0.11), tolerance_cost: Some(0.12) }",
        ),
        (
            LocalSolverConfig::bobyqa()
                .max_iter(48)
                .initial_radius(0.25)
                .final_radius(0.001)
                .interpolation_points(Some(6))
                .build(),
            "BOBYQA { max_iter: 48, initial_radius: 0.25, final_radius: 0.001, interpolation_points: Some(6) }",
        ),
        (
            LocalSolverConfig::slsqp().max_iter(49).accuracy(1e-7).max_subproblem_iter(11).build(),
            "SLSQP { max_iter: 49, accuracy: Some(1e-7), max_subproblem_iter: Some(11) }",
        ),
        (
            LocalSolverConfig::barrier()
                .max_iter(50)
                .mu0(0.5)
                .reduction(5.0)
                .duality_gap_tol(1e-7)
                .inner_max_iter(25)
                .build(),
            "Barrier { max_iter: 50, mu0: 0.5, reduction: 5.0, duality_gap_tol: 1e-7, inner_max_iter: 25 }",
        ),
        (
            LocalSolverConfig::augmented_lagrangian()
                .max_iter(51)
                .rho0(5.0)
                .rho_increase(5.0)
                .feasibility_decrease(0.5)
                .feasibility_tol(1e-7)
                .inner_max_iter(25)
                .build(),
            "AugmentedLagrangian { max_iter: 51, rho0: 5.0, rho_increase: 5.0, feasibility_decrease: 0.5, feasibility_tol: 1e-7, inner_max_iter: 25 }",
        ),
    ];

    for (config, expected) in cases {
        assert_eq!(format!("{config:?}"), expected);
    }
}

#[test]
fn optional_tolerances_can_be_disabled_or_set_to_zero() {
    for value in [None, Some(0.0)] {
        let configs = [
            LBFGSBuilder::new().tolerance_grad(value).tolerance_cost(value).build(),
            GradientDescentBuilder::new().tolerance_grad(value).tolerance_cost(value).build(),
            TrustRegionBuilder::new().tolerance_grad(value).build(),
            LBFGSBBuilder::new().tolerance_projected_grad(value).tolerance_cost(value).build(),
            NelderMeadBuilder::new().tolerance_simplex(value).tolerance_cost(value).build(),
            BoundedNelderMeadBuilder::new().tolerance_simplex(value).tolerance_cost(value).build(),
        ];
        for config in configs {
            match config {
                LocalSolverConfig::LBFGS { tolerance_grad, tolerance_cost, .. }
                | LocalSolverConfig::GradientDescent { tolerance_grad, tolerance_cost, .. } => {
                    assert_eq!(tolerance_grad, value);
                    assert_eq!(tolerance_cost, value);
                }
                LocalSolverConfig::TrustRegion { tolerance_grad, .. } => {
                    assert_eq!(tolerance_grad, value);
                }
                LocalSolverConfig::LBFGSB { tolerance_projected_grad, tolerance_cost, .. } => {
                    assert_eq!(tolerance_projected_grad, value);
                    assert_eq!(tolerance_cost, value);
                }
                LocalSolverConfig::NelderMead { tolerance_simplex, tolerance_cost, .. }
                | LocalSolverConfig::BoundedNelderMead {
                    tolerance_simplex, tolerance_cost, ..
                } => {
                    assert_eq!(tolerance_simplex, value);
                    assert_eq!(tolerance_cost, value);
                }
                _ => panic!("Unexpected solver: {config:?}"),
            }
        }
    }
}
