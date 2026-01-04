/// Basic Tutorial - Getting Started with GlobalSearch-rs
///
/// This example provides a simple introduction to using the globalsearch-rs library
/// for global optimization. We'll optimize the Rosenbrock function, a classic
/// optimization benchmark with a global minimum at (1, 1).
///
/// The Rosenbrock function is defined as:
/// f(x, y) = (a - x)² + b(y - x²)²
///
/// With a = 1 and b = 100, the function has a global minimum at (1, 1) with f(1, 1) = 0.
/// This function is challenging because it has a narrow curved valley leading to the minimum.
///
/// This tutorial covers:
/// 1. Implementing the Problem trait
/// 2. Setting up basic optimization parameters  
/// 3. Running the optimization
/// 4. Interpreting results
///
/// # References
/// H. H. Rosenbrock, An Automatic Method for Finding the Greatest or Least Value of a Function,
/// The Computer Journal, Volume 3, Issue 3, 1960, Pages 175–184, https://doi.org/10.1093/comjnl/3.3.175
use globalsearch::problem::Problem;
use globalsearch::{
    local_solver::builders::COBYLABuilder,
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{Array1, Array2, array};

/// The Rosenbrock function
#[derive(Debug, Clone)]
pub struct Rosenbrock {
    a: f64,
    b: f64,
}

impl Rosenbrock {
    /// Create a new Rosenbrock function with parameters a and b
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Create the standard Rosenbrock function with a=1, b=100
    pub fn standard() -> Self {
        Self::new(1.0, 100.0)
    }
}

// Step 1: Implement the Problem trait
impl Problem for Rosenbrock {
    /// The objective function: f(x, y) = (a - x)² + b(y - x²)²
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Validate input dimensions
        if x.len() != 2 {
            return Err(EvaluationError::InvalidInput {
                reason: "Rosenbrock function requires exactly 2 variables".to_string(),
            });
        }

        let x_val = x[0];
        let y_val = x[1];

        let term1 = (self.a - x_val).powi(2);
        let term2 = self.b * (y_val - x_val.powi(2)).powi(2);

        Ok(term1 + term2)
    }

    /// Define the search bounds: typically [-5, 5] for both variables
    fn variable_bounds(&self) -> Array2<f64> {
        array![
            [-5.0, 10.0], // x bounds
            [-5.0, 10.0]  // y bounds
        ]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimizing the Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²");
    println!("Global minimum: f(1, 1) = 0");
    println!();

    // Step 1: Create the problem instance
    let problem = Rosenbrock::standard();

    // Step 2: Configure local optimization parameters
    let local_solver_config = COBYLABuilder::default()
        .max_iter(500) // Maximum iterations for COBYLA
        .initial_step_size(0.1) // Initial step size for COBYLA
        .ftol_rel(1e-10) // Relative tolerance
        .ftol_abs(1e-12) // Absolute tolerance
        .build();

    // Step 3: Configure optimization parameters
    let params = OQNLPParams {
        iterations: 750,        // Number of stage two iterations
        population_size: 1500,  // Size of the scatter search population
        wait_cycle: 15,         // Wait before updating search parameters
        threshold_factor: 0.65, // Merit filter sensitivity
        distance_factor: 0.10,  // Minimum distance between solutions
        local_solver_type: LocalSolverType::COBYLA,
        local_solver_config, // Pass the COBYLA configuration
        seed: 0,             // Random seed for reproducibility
    };

    println!("Optimization Settings:");
    println!("- Stage two iterations: {}", params.iterations);
    println!("- Population size: {}", params.population_size);
    println!("- Local solver: {:?}", params.local_solver_type);
    println!("- Random seed: {}", params.seed);
    println!();

    // Step 4: Create and run the optimizer
    println!("Starting optimization...");
    let mut oqnlp: OQNLP<Rosenbrock> = OQNLP::new(problem, params)?;
    let solution_set: SolutionSet = oqnlp.run()?;

    // Step 5: Analyze the results
    println!("Optimization completed!");
    println!("{}", solution_set);

    if let Some(best) = solution_set.best_solution() {
        println!("Detailed Analysis:");
        println!("=================");

        let x_opt = best.point[0];
        let y_opt = best.point[1];
        let f_opt = best.objective;

        println!("Distance from global minimum (1, 1):");
        let distance = ((x_opt - 1.0).powi(2) + (y_opt - 1.0).powi(2)).sqrt();
        println!("  Euclidean distance: {:.6}", distance);
        println!("  Error in objective: {:.8}", f_opt);
        println!();

        // Evaluate how good the solution is
        if f_opt < 1e-4 {
            println!("Found global minimum with high precision");
        } else if f_opt < 1e-2 {
            println!("Close to the global minimum");
        } else if f_opt < 1.0 {
            println!("Good solution, near the global minimum");
        } else {
            println!("Solution found, but may be a local minimum");
        }
    } else {
        println!("No solutions found");
    }

    Ok(())
}
