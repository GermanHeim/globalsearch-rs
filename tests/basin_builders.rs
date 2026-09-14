#![cfg(feature = "basin")]

use globalsearch::local_solver::builders::*;

#[test]
fn constructors_use_documented_defaults() {
    let cases = [
        (
            [
                BasinLBFGSBuilder::new().build(),
                BasinLBFGSBuilder::default().build(),
                LocalSolverConfig::basin_lbfgs().build(),
            ],
            "BasinLBFGS { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None, history_size: 10 }",
        ),
        (
            [
                BasinGradientDescentBuilder::new().build(),
                BasinGradientDescentBuilder::default().build(),
                LocalSolverConfig::basin_gradient_descent().build(),
            ],
            "BasinGradientDescent { max_iter: 1000, tolerance_grad: Some(1e-6), tolerance_cost: None }",
        ),
        (
            [
                BasinTrustRegionBuilder::new().build(),
                BasinTrustRegionBuilder::default().build(),
                LocalSolverConfig::basin_trust_region().build(),
            ],
            "BasinTrustRegion { max_iter: 1000, tolerance_grad: Some(1e-6), trust_region_radius_method: Steihaug, radius: 1.0, max_radius: 100.0, eta: 0.125 }",
        ),
        (
            [
                BasinNelderMeadBuilder::new().build(),
                BasinNelderMeadBuilder::default().build(),
                LocalSolverConfig::basin_nelder_mead().build(),
            ],
            "BasinNelderMead { max_iter: 1000, simplex_delta: 0.1, tolerance_simplex: Some(1e-6), tolerance_cost: Some(1e-8) }",
        ),
        (
            [
                BasinLBFGSBBuilder::new().build(),
                BasinLBFGSBBuilder::default().build(),
                LocalSolverConfig::basin_lbfgsb().build(),
            ],
            "BasinLBFGSB { max_iter: 1000, tolerance_projected_grad: Some(1e-6), tolerance_cost: None, history_size: 10 }",
        ),
        (
            [
                BasinBoundedNelderMeadBuilder::new().build(),
                BasinBoundedNelderMeadBuilder::default().build(),
                LocalSolverConfig::basin_bounded_nelder_mead().build(),
            ],
            "BasinBoundedNelderMead { max_iter: 1000, simplex_delta: 0.1, tolerance_simplex: Some(1e-6), tolerance_cost: Some(1e-8) }",
        ),
        (
            [
                BasinBOBYQABuilder::new().build(),
                BasinBOBYQABuilder::default().build(),
                LocalSolverConfig::basin_bobyqa().build(),
            ],
            "BasinBOBYQA { max_iter: 1000, initial_radius: 1.0, final_radius: 1e-6, interpolation_points: None }",
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
            LocalSolverConfig::basin_lbfgs()
                .max_iter(42)
                .tolerance_grad(0.01)
                .tolerance_cost(Some(0.02))
                .history_size(7)
                .build(),
            "BasinLBFGS { max_iter: 42, tolerance_grad: Some(0.01), tolerance_cost: Some(0.02), history_size: 7 }",
        ),
        (
            LocalSolverConfig::basin_gradient_descent()
                .max_iter(43)
                .tolerance_grad(Some(0.03))
                .tolerance_cost(0.04)
                .build(),
            "BasinGradientDescent { max_iter: 43, tolerance_grad: Some(0.03), tolerance_cost: Some(0.04) }",
        ),
        (
            LocalSolverConfig::basin_trust_region()
                .max_iter(44)
                .tolerance_grad(0.05)
                .method(TrustRegionRadiusMethod::Cauchy)
                .radius(0.5)
                .max_radius(5.0)
                .eta(0.2)
                .build(),
            "BasinTrustRegion { max_iter: 44, tolerance_grad: Some(0.05), trust_region_radius_method: Cauchy, radius: 0.5, max_radius: 5.0, eta: 0.2 }",
        ),
        (
            LocalSolverConfig::basin_nelder_mead()
                .max_iter(45)
                .simplex_delta(0.2)
                .tolerance_simplex(0.06)
                .tolerance_cost(Some(0.07))
                .build(),
            "BasinNelderMead { max_iter: 45, simplex_delta: 0.2, tolerance_simplex: Some(0.06), tolerance_cost: Some(0.07) }",
        ),
        (
            LocalSolverConfig::basin_lbfgsb()
                .max_iter(46)
                .tolerance_projected_grad(Some(0.08))
                .tolerance_cost(0.09)
                .history_size(8)
                .build(),
            "BasinLBFGSB { max_iter: 46, tolerance_projected_grad: Some(0.08), tolerance_cost: Some(0.09), history_size: 8 }",
        ),
        (
            LocalSolverConfig::basin_bounded_nelder_mead()
                .max_iter(47)
                .simplex_delta(0.3)
                .tolerance_simplex(Some(0.11))
                .tolerance_cost(0.12)
                .build(),
            "BasinBoundedNelderMead { max_iter: 47, simplex_delta: 0.3, tolerance_simplex: Some(0.11), tolerance_cost: Some(0.12) }",
        ),
        (
            LocalSolverConfig::basin_bobyqa()
                .max_iter(48)
                .initial_radius(0.25)
                .final_radius(0.001)
                .interpolation_points(Some(6))
                .build(),
            "BasinBOBYQA { max_iter: 48, initial_radius: 0.25, final_radius: 0.001, interpolation_points: Some(6) }",
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
            BasinLBFGSBuilder::new().tolerance_grad(value).tolerance_cost(value).build(),
            BasinGradientDescentBuilder::new().tolerance_grad(value).tolerance_cost(value).build(),
            BasinTrustRegionBuilder::new().tolerance_grad(value).build(),
            BasinLBFGSBBuilder::new().tolerance_projected_grad(value).tolerance_cost(value).build(),
            BasinNelderMeadBuilder::new().tolerance_simplex(value).tolerance_cost(value).build(),
            BasinBoundedNelderMeadBuilder::new()
                .tolerance_simplex(value)
                .tolerance_cost(value)
                .build(),
        ];
        for config in configs {
            match config {
                LocalSolverConfig::BasinLBFGS { tolerance_grad, tolerance_cost, .. }
                | LocalSolverConfig::BasinGradientDescent {
                    tolerance_grad, tolerance_cost, ..
                } => {
                    assert_eq!(tolerance_grad, value);
                    assert_eq!(tolerance_cost, value);
                }
                LocalSolverConfig::BasinTrustRegion { tolerance_grad, .. } => {
                    assert_eq!(tolerance_grad, value);
                }
                LocalSolverConfig::BasinLBFGSB {
                    tolerance_projected_grad, tolerance_cost, ..
                } => {
                    assert_eq!(tolerance_projected_grad, value);
                    assert_eq!(tolerance_cost, value);
                }
                LocalSolverConfig::BasinNelderMead {
                    tolerance_simplex, tolerance_cost, ..
                }
                | LocalSolverConfig::BasinBoundedNelderMead {
                    tolerance_simplex,
                    tolerance_cost,
                    ..
                } => {
                    assert_eq!(tolerance_simplex, value);
                    assert_eq!(tolerance_cost, value);
                }
                _ => panic!("Unexpected solver: {config:?}"),
            }
        }
    }
}
