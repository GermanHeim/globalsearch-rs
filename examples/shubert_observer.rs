/// Example demonstrating Observer usage with the Shubert function
///
/// The Shubert function is a complex multimodal function with many global minima.
/// This example shows how to use observers to track the optimization progress.
///
/// Function definition: f(x₁, x₂) = [Σᵢ₌₁⁵ i·cos((i+1)x₁ + i)] · [Σᵢ₌₁⁵ i·cos((i+1)x₂ + i)]
///
/// Properties:
/// - Bounds: [-10, 10] for each dimension
/// - Global minimum: f(x*) = -186.7309
/// - Has 18 global minima
/// - Highly multimodal with many local minima (760 in total)
///
/// This example demonstrates:
/// - Solving a challenging 2D multimodal problem
/// - Using observers to track both Stage 1 (scatter search) and Stage 2 (local refinement)
/// - Finding multiple global minima
/// - Real-time progress monitoring
///
/// Please note that we use the definition of the Shubert function as given in the reference by Molga, M., & Smutnicki, C.
/// X. Wang, S. -s. Wang and L. Xiao solved a modified version of the Shubert function that has two additional terms.
///
/// References:
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 36-38. Retrieved October 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
///
/// X. Wang, S. -s. Wang and L. Xiao, "Solving Shubert Function Optimization Problem by Using Thermodynamics Evolutionary Algorithm," 2010 International Conference on Biomedical Engineering and Computer Science, Wuhan, China, 2010, pp. 1-4, doi: 10.1109/ICBECS.2010.5462350.
use globalsearch::local_solver::builders::COBYLABuilder;
use globalsearch::observers::Observer;
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, LocalSolverType, OQNLPParams};
use ndarray::{Array1, Array2, array};

/// Shubert function implementation
///
/// A highly multimodal 2D function with 18 global minima at f(x*) = -186.7309.
/// The function has 760 local minima in total, making it very challenging.
#[derive(Debug, Clone)]
pub struct Shubert;

impl Problem for Shubert {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let x1 = x[0];
        let x2 = x[1];

        // First sum: Σᵢ₌₁⁵ i·cos((i+1)x₁ + i)
        let sum1: f64 = (1..=5)
            .map(|i| {
                let i_f64 = i as f64;
                i_f64 * ((i_f64 + 1.0) * x1 + i_f64).cos()
            })
            .sum();

        // Second sum: Σᵢ₌₁⁵ i·cos((i+1)x₂ + i)
        let sum2: f64 = (1..=5)
            .map(|i| {
                let i_f64 = i as f64;
                i_f64 * ((i_f64 + 1.0) * x2 + i_f64).cos()
            })
            .sum();

        Ok(sum1 * sum2)
    }

    fn variable_bounds(&self) -> Array2<f64> {
        // Standard bounds for Shubert function: [-10, 10] for both dimensions
        array![[-10.0, 10.0], [-10.0, 10.0]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Shubert Function Optimization with Observer\n");

    println!("Global minimum: f(x*) = -186.7309");
    println!("Number of global minima: 18");
    println!("Bounds: [-10.0, 10.0] for each dimension\n");

    let problem = Shubert;

    // Optimization parameters
    let params = OQNLPParams {
        iterations: 5000,
        wait_cycle: 25,
        threshold_factor: 0.3,
        distance_factor: 0.8,
        population_size: 13000,
        local_solver_type: LocalSolverType::COBYLA,
        local_solver_config: COBYLABuilder::default().max_iter(75).initial_step_size(1.0).build(),
        seed: 0,
    };

    // Create observer with callbacks for both Stage 1 and Stage 2 tracking
    let observer = Observer::new()
        .with_stage1_tracking()
        .with_stage2_tracking()
        .with_timing()
        .with_default_callback();

    println!("Running optimization...\n");
    let mut oqnlp = OQNLP::new(problem.clone(), params)?.add_observer(observer);

    let solution_set = oqnlp.run()?;

    println!("\nOptimization Complete");
    println!("{}", solution_set);

    Ok(())
}
