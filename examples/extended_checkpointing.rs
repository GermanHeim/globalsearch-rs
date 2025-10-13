/// Extended example demonstrating checkpoint continuation with more iterations
///
/// This example shows how to:
/// 1. Run optimization with checkpointing enabled
/// 2. Resume from checkpoint with increased iteration count
/// 3. Handle multiple checkpoint continuation cycles
use ndarray::{array, Array1, Array2};

#[cfg(feature = "checkpointing")]
use globalsearch::{
    local_solver::builders::{TrustRegionBuilder, TrustRegionRadiusMethod},
    oqnlp::OQNLP,
    problem::Problem,
    types::{CheckpointConfig, EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};

#[cfg(feature = "checkpointing")]
use std::path::PathBuf;

#[cfg(not(feature = "checkpointing"))]
use globalsearch::{problem::Problem, types::EvaluationError};

#[derive(Debug, Clone)]
pub struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
            + x[0] * x[1]
            + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2))
    }

    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        Ok(array![
            (8.0 - 8.4 * x[0].powi(2) + 2.0 * x[0].powi(4)) * x[0] + x[1],
            x[0] + (-8.0 + 16.0 * x[1].powi(2)) * x[1]
        ])
    }

    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        Ok(array![
            [
                (4.0 * x[0].powi(2) - 4.2) * x[0].powi(2)
                    + 4.0 * (4.0 / 3.0 * x[0].powi(3) - 4.2 * x[0]) * x[0]
                    + 2.0 * (x[0].powi(4) / 3.0 - 2.1 * x[0].powi(2) + 4.0),
                1.0
            ],
            [1.0, 40.0 * x[1].powi(2) + 2.0 * (4.0 * x[1].powi(2) - 4.0)]
        ])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

#[cfg(feature = "checkpointing")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OQNLP Extended Checkpointing Example");
    println!("====================================");

    let problem = SixHumpCamel;

    // Configure checkpointing
    let checkpoint_config = CheckpointConfig {
        checkpoint_dir: PathBuf::from("./extended_checkpoints"),
        checkpoint_name: "sixhump_extended".to_string(),
        save_frequency: 5,
        keep_all: true,     // Keep all checkpoints to see progression
        auto_resume: false, // We'll manually control resumption
    };

    println!("Running optimization in stages with increasing iterations...");
    println!();

    // Stage 1: Initial run with 100 iterations
    println!("=== Stage 1: Initial optimization (100 iterations) ===");
    run_optimization_stage(&problem, 10, 500, &checkpoint_config, false)?;

    // Stage 2: Continue with 200 total iterations (100 more)
    println!("\n=== Stage 2: Continue optimization (200 total iterations) ===");
    run_optimization_stage(&problem, 200, 500, &checkpoint_config, true)?;

    // Stage 3: Continue with 500 total iterations (300 more)
    println!("\n=== Stage 3: Extended optimization (500 total iterations) ===");
    run_optimization_stage(&problem, 500, 800, &checkpoint_config, true)?;

    // Stage 4: Final push with 1000 total iterations (500 more)
    println!("\n=== Stage 4: Final optimization (1000 total iterations) ===");
    run_optimization_stage(&problem, 1000, 1200, &checkpoint_config, true)?;

    println!("\n=== Optimization Complete ===");
    println!("Check ./extended_checkpoints/ to see all saved checkpoints");

    Ok(())
}

#[cfg(feature = "checkpointing")]
fn run_optimization_stage(
    problem: &SixHumpCamel,
    iterations: usize,
    population_size: usize,
    checkpoint_config: &CheckpointConfig,
    resume_from_checkpoint: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure OQNLP parameters for this stage
    let params = OQNLPParams {
        local_solver_type: LocalSolverType::TrustRegion,
        local_solver_config: TrustRegionBuilder::default()
            .method(TrustRegionRadiusMethod::Steihaug)
            .build(),
        iterations,
        population_size,
        ..OQNLPParams::default()
    };

    // Create OQNLP instance
    let mut oqnlp = OQNLP::new(problem.clone(), params.clone())?
        .with_checkpointing(checkpoint_config.clone())?
        .verbose();

    // Handle resumption
    let resumed = if resume_from_checkpoint {
        println!("Attempting to resume from checkpoint...");
        oqnlp.resume_with_modified_params(params)?
    } else {
        println!("Starting fresh optimization...");
        false
    };

    if resumed {
        println!("Resumed from checkpoint");
    } else {
        println!("Starting new optimization");
    }

    // Run optimization
    let start_time = std::time::Instant::now();
    let sol_set: SolutionSet = oqnlp.run()?;
    let elapsed = start_time.elapsed();

    // Display results
    println!("Completed in {:.2} seconds", elapsed.as_secs_f64());
    println!("Best solution found:");
    if let Some(best) = sol_set.best_solution() {
        println!("  Objective: {:.8e}", best.objective);
        println!("  Parameters: {:.6e}", best.point);
    }
    println!("Total solutions in set: {}", sol_set.len());

    Ok(())
}

#[cfg(not(feature = "checkpointing"))]
pub fn main() {
    println!("This example requires the 'checkpointing' feature to be enabled.");
    println!("Run with: cargo run --example checkpointing --features checkpointing");
}
