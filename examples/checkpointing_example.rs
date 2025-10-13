/// Example demonstrating checkpointing functionality in globalsearch-rs
///
/// This example shows how to:
/// 1. Configure checkpointing for an OQNLP optimization
/// 2. Resume optimization from a saved checkpoint
/// 3. Modify parameters during resumption
/// 4. Handle long-running optimizations with periodic saves
use ndarray::{array, Array1, Array2};

#[cfg(feature = "checkpointing")]
use globalsearch::{
    local_solver::builders::NelderMeadBuilder,
    oqnlp::OQNLP,
    problem::Problem,
    types::{CheckpointConfig, EvaluationError, LocalSolverType, OQNLPParams},
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
    println!("OQNLP Checkpointing Example with Parameter Modification");
    println!("======================================================");

    let problem = SixHumpCamel;

    // Configure OQNLP parameters for initial optimization
    let params = OQNLPParams {
        local_solver_type: LocalSolverType::NelderMead,
        local_solver_config: NelderMeadBuilder::default().build(),
        iterations: 15, // Very few iterations to find only one solution initially
        population_size: 200, // Smaller population size
        wait_cycle: 10, // Ensure wait_cycle < iterations
        ..OQNLPParams::default()
    };

    // Configure checkpointing
    let checkpoint_config = CheckpointConfig {
        checkpoint_dir: PathBuf::from("./oqnlp_checkpoints"),
        checkpoint_name: "sixhump_camel".to_string(),
        save_frequency: 10, // Save every 10 iterations (less than iterations)
        keep_all: false,    // Keep all checkpoint files
        auto_resume: true,  // Automatically resume if checkpoint exists
    };

    println!("Checkpoint configuration:");
    println!("  Directory: {}", checkpoint_config.checkpoint_dir.display());
    println!("  Save frequency: every {} iterations", checkpoint_config.save_frequency);
    println!("  Auto-resume: {}", checkpoint_config.auto_resume);
    println!();

    // First run: optimize with initial parameters
    println!("Starting initial optimization with {} iterations...", params.iterations);
    let mut oqnlp = OQNLP::new(problem.clone(), params.clone())?
        .with_checkpointing(checkpoint_config)?
        .verbose();

    let result = oqnlp.run()?;
    println!("Initial optimization completed!");
    println!("{}", result);

    // Continue optimization with more iterations to find additional solutions
    let extended_params = OQNLPParams {
        iterations: 80,        // More iterations to find the second solution
        population_size: 1000, // Increase population size to allow more iterations
        ..params
    };

    println!(
        "\nContinuing optimization with {} iterations and expanded population to {} to find additional solutions...",
        extended_params.iterations, extended_params.population_size
    );

    // Create a new OQNLP instance and try to resume with modified parameters
    let mut continued_oqnlp = OQNLP::new(problem, extended_params.clone())?
        .with_checkpointing(CheckpointConfig {
            checkpoint_dir: PathBuf::from("./oqnlp_checkpoints"),
            checkpoint_name: "sixhump_camel".to_string(),
            save_frequency: 20,
            keep_all: false,
            auto_resume: false, // Disable auto_resume since we'll manually resume with modified params
        })?
        .verbose();

    // Try to resume from existing checkpoint with modified parameters
    if continued_oqnlp.resume_with_modified_params(extended_params)? {
        println!("Successfully resumed from checkpoint with extended parameters!");

        let final_result = continued_oqnlp.run()?;
        println!("Extended optimization completed!");
        println!("{}", final_result)
    } else {
        println!("No checkpoint found to continue from!");
    }

    println!();
    println!("This example demonstrates:");
    println!("1. Running an initial optimization with checkpointing");
    println!("2. Loading a checkpoint and continuing with modified parameters");
    println!("3. Comparing results between the initial and extended runs");

    Ok(())
}

#[cfg(not(feature = "checkpointing"))]
fn main() {
    println!("This example requires the 'checkpointing' feature to be enabled.");
    println!("Run with: cargo run --example checkpointing_example --features checkpointing");
}
