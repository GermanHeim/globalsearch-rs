/// Constrained Optimization Example
///
/// This example demonstrates how to solve optimization problems with constraints
/// using the COBYLA solver, which natively supports general nonlinear constraints.
///
/// Problem: Minimize f(x,y) = (x-1)² + (y-1)² subject to:
/// - x + y ≤ 1.5 (linear constraint)
/// - x² + y² ≥ 0.5 (nonlinear constraint)
/// - 0 ≤ x ≤ 2, 0 ≤ y ≤ 2 (box constraints)
///
/// This is a circle-packing style problem where we want to minimize distance
/// from (1,1) while staying inside a feasible region.
///
/// ## Key Concepts:
/// - Constraint functions must return ≥ 0 when satisfied, < 0 when violated
/// - COBYLA is the only solver that supports general constraints
/// - Box constraints are handled separately via variable_bounds()
/// - Penalty methods can be used as alternatives for other solvers
use globalsearch::problem::Problem;
use globalsearch::{
    local_solver::builders::COBYLABuilder,
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{array, Array1, Array2};

/// Constrained optimization problem
#[derive(Debug, Clone)]
pub struct ConstrainedProblem;

impl Problem for ConstrainedProblem {
    /// Objective: minimize distance from (1, 1)
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        if x.len() != 2 {
            return Err(EvaluationError::InvalidInput(
                "Problem requires exactly 2 variables".to_string(),
            ));
        }

        let x_val = x[0];
        let y_val = x[1];

        // Minimize squared distance from (1, 1)
        Ok((x_val - 1.0).powi(2) + (y_val - 1.0).powi(2))
    }

    /// Box constraints: 0 ≤ x ≤ 2, 0 ≤ y ≤ 2
    fn variable_bounds(&self) -> Array2<f64> {
        array![
            [0.0, 2.0], // x bounds
            [0.0, 2.0]  // y bounds
        ]
    }

    /// General constraints (only supported by COBYLA)
    /// Return value ≥ 0 means constraint is satisfied
    /// Return value < 0 means constraint is violated
    fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
        vec![
            // Constraint 1: x + y ≤ 1.5  ⇒  1.5 - x - y ≥ 0
            |x: &[f64], _: &mut ()| 1.5 - x[0] - x[1],
            // Constraint 2: x² + y² ≥ 0.5  ⇒  x² + y² - 0.5 ≥ 0
            |x: &[f64], _: &mut ()| x[0] * x[0] + x[1] * x[1] - 0.5,
        ]
    }
}

/// Helper function to check constraint violations
fn evaluate_constraints(point: &Array1<f64>) -> (Vec<f64>, bool) {
    let x = point[0];
    let y = point[1];

    let constraint_values = vec![
        1.5 - x - y,         // Linear constraint
        x * x + y * y - 0.5, // Nonlinear constraint
    ];

    let all_satisfied = constraint_values.iter().all(|&c| c >= -1e-6);

    (constraint_values, all_satisfied)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Constrained Optimization Example");
    println!("================================");
    println!("Problem: Minimize f(x,y) = (x-1)² + (y-1)²");
    println!("Subject to:");
    println!("  - x + y ≤ 1.5 (linear constraint)");
    println!("  - x² + y² ≥ 0.5 (nonlinear constraint)");
    println!("  - 0 ≤ x ≤ 2, 0 ≤ y ≤ 2 (box constraints)");
    println!();

    let problem = ConstrainedProblem;

    // COBYLA configuration
    let cobyla_config = COBYLABuilder::default()
        .max_iter(400)
        .initial_step_size(0.1)
        .ftol_rel(1e-8)
        .ftol_abs(1e-10)
        .build();

    let params = OQNLPParams {
        iterations: 15,
        population_size: 150,
        wait_cycle: 5,
        threshold_factor: 0.1,
        distance_factor: 0.8,
        local_solver_type: LocalSolverType::COBYLA,
        local_solver_config: cobyla_config,
        seed: 0,
    };

    println!("Optimization Configuration:");
    println!("- Using COBYLA solver (supports constraints)");
    println!("- Population size: {}", params.population_size);
    println!("- Stage two iterations: {}", params.iterations);
    println!();

    let mut oqnlp: OQNLP<ConstrainedProblem> = OQNLP::new(problem.clone(), params)?.verbose();

    println!("Starting constrained optimization...");
    let solution_set: SolutionSet = oqnlp.run()?;

    // Display results with constraint information
    println!(
        "{}",
        solution_set.display_with_constraints(&problem, Some(&["x + y ≤ 1.5", "x² + y² ≥ 0.5"]))
    );

    // Detailed analysis
    if let Some(best) = solution_set.best_solution() {
        let x_opt = best.point[0];
        let y_opt = best.point[1];
        let f_opt = best.objective;

        println!();
        println!("Detailed Solution Analysis:");
        println!("==========================");
        println!("Optimal point: ({:.6}, {:.6})", x_opt, y_opt);
        println!("Objective value: {:.8}", f_opt);

        // Check constraints manually
        let (constraint_values, all_satisfied) = evaluate_constraints(&best.point);

        println!();
        println!("Constraint Verification:");
        println!(
            "- Linear constraint (x + y ≤ 1.5): {:.6} {}",
            constraint_values[0],
            if constraint_values[0] >= -1e-6 { "✓" } else { "✗" }
        );
        println!(
            "- Nonlinear constraint (x² + y² ≥ 0.5): {:.6} {}",
            constraint_values[1],
            if constraint_values[1] >= -1e-6 { "✓" } else { "✗" }
        );

        println!();
        if all_satisfied {
            println!("All constraints satisfied!");

            // Analyze the solution quality
            let distance_from_unconstrained =
                ((x_opt - 1.0).powi(2) + (y_opt - 1.0).powi(2)).sqrt();
            println!(
                "Distance from unconstrained optimum (1,1): {:.6}",
                distance_from_unconstrained
            );

            // Check if we're at a constraint boundary (active constraints)
            println!();
            println!("Active Constraints (within tolerance 1e-3):");
            if constraint_values[0].abs() < 1e-3 {
                println!("- Linear constraint x + y ≤ 1.5 is ACTIVE");
            }
            if constraint_values[1].abs() < 1e-3 {
                println!("- Nonlinear constraint x² + y² ≥ 0.5 is ACTIVE");
            }

            // Theoretical analysis
            println!();
            println!("Theoretical Analysis:");
            println!("- Unconstrained optimum: (1, 1) with f = 0");
            println!("- Point (1, 1) violates x + y ≤ 1.5 since 1 + 1 = 2 > 1.5");
            println!("- Optimal solution should be on the constraint boundary x + y = 1.5");
            println!("- Among points on x + y = 1.5, closest to (1,1) is (0.75, 0.75)");

            let theoretical_opt = (0.75_f64, 0.75_f64);
            let theoretical_f =
                (theoretical_opt.0 - 1.0).powi(2) + (theoretical_opt.1 - 1.0).powi(2);
            let error = (f_opt - theoretical_f).abs();

            println!(
                "- Theoretical optimum: ({:.3}, {:.3}) with f = {:.6}",
                theoretical_opt.0, theoretical_opt.1, theoretical_f
            );
            println!("- Error from theoretical optimum: {:.8}", error);

            if error < 1e-3 {
                println!("Found theoretical optimum");
            } else if error < 1e-2 {
                println!("Very good approximation to theoretical optimum");
            }
        } else {
            println!("Some constraints are violated!");
            println!("This may indicate insufficient iterations or numerical issues.");
        }
    }

    Ok(())
}
