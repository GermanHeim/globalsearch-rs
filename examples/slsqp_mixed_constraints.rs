/// SLSQP Mixed-Constraints Example
///
/// This example demonstrates the gradient-based SLSQP local solver on a
/// problem combining every constraint block it supports: box bounds, a
/// linear inequality (`A x <= b`), and a nonlinear inequality (`g(x) >= 0`).
///
/// Problem: Minimize f(x,y) = (x-2)² + (y+1)² subject to:
/// - x + y ≤ 0.5 (linear inequality)
/// - x² + y² ≥ 0.5 (nonlinear inequality, inactive at the optimum)
/// - -5 ≤ x ≤ 5, -5 ≤ y ≤ 5 (box constraints)
///
/// The unconstrained optimum (2, -1) violates x + y ≤ 0.5, so the constrained
/// optimum lies on that boundary at (1.75, -1.25) with f = 0.125.
use globalsearch::problem::Problem;
use globalsearch::{
    local_solver::builders::SLSQPBuilder,
    oqnlp::OQNLP,
    types::{EvaluationError, OQNLPParams, SolutionSet},
};
use ndarray::{Array1, Array2, array};

/// Mixed-constraint quadratic problem.
#[derive(Debug, Clone)]
pub struct MixedConstraints;

impl Problem for MixedConstraints {
    /// Objective: minimize distance from (2, -1).
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }

    /// Analytic gradient required by SLSQP.
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![2.0 * (x[0] - 2.0), 2.0 * (x[1] + 1.0)])
    }

    /// Box constraints: -5 ≤ x ≤ 5, -5 ≤ y ≤ 5.
    fn variable_bounds(&self) -> Array2<f64> {
        array![
            [-5.0, 5.0], // x bounds
            [-5.0, 5.0]  // y bounds
        ]
    }

    /// Linear inequality x + y ≤ 0.5 as (A, b).
    fn linear_inequalities(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        Some((array![[1.0, 1.0]], array![0.5]))
    }

    /// Nonlinear inequality x² + y² ≥ 0.5 (satisfied means ≥ 0).
    fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![x[0] * x[0] + x[1] * x[1] - 0.5])
    }

    /// Jacobian of the nonlinear blocks: one inequality row.
    fn constraint_jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![[2.0 * x[0], 2.0 * x[1]]])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SLSQP Mixed-Constraints Example");
    println!("===============================");
    println!("Problem: Minimize f(x,y) = (x-2)² + (y+1)²");
    println!("Subject to:");
    println!("  - x + y ≤ 0.5 (linear inequality)");
    println!("  - x² + y² ≥ 0.5 (nonlinear inequality)");
    println!("  - -5 ≤ x ≤ 5, -5 ≤ y ≤ 5 (box constraints)");
    println!();

    let problem = MixedConstraints;

    let slsqp_config = SLSQPBuilder::default().max_iter(1000).accuracy(Some(1e-8)).build();

    let params = OQNLPParams {
        iterations: 10,
        population_size: 100,
        wait_cycle: 5,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        local_solver_config: slsqp_config,
        seed: 0,
    };

    println!("Optimization Configuration:");
    println!("- Using SLSQP solver (gradient-based, constrained)");
    println!("- Population size: {}", params.population_size);
    println!("- Stage two iterations: {}", params.iterations);
    println!();

    let mut oqnlp: OQNLP<MixedConstraints> = OQNLP::new(problem.clone(), params)?.verbose();

    println!("Starting mixed-constraint optimization...");
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("{}", solution_set.display_with_constraints(&problem, Some(&["x² + y² ≥ 0.5"]))?);

    if let Some(best) = solution_set.best_solution() {
        let (x_opt, y_opt, f_opt) = (best.point[0], best.point[1], best.objective);
        println!();
        println!("Detailed Solution Analysis:");
        println!("==========================");
        println!("Optimal point: ({x_opt:.6}, {y_opt:.6})");
        println!("Objective value: {f_opt:.8}");
        println!("Linear residual (x + y - 0.5 ≤ 0): {:.8}", x_opt + y_opt - 0.5);

        let theoretical = (1.75_f64, -1.25_f64);
        let theoretical_f = (theoretical.0 - 2.0).powi(2) + (theoretical.1 + 1.0).powi(2);
        let error = (f_opt - theoretical_f).abs();
        println!(
            "Theoretical optimum: ({:.3}, {:.3}) with f = {theoretical_f:.6}",
            theoretical.0, theoretical.1
        );
        println!("Error from theoretical optimum: {error:.8}");

        if error < 1e-3 {
            println!("Found theoretical optimum");
        } else if error < 1e-2 {
            println!("Very good approximation to theoretical optimum");
        } else {
            println!("Result is off; consider more iterations or population.");
        }
    }

    Ok(())
}
