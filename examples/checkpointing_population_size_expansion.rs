/// Example demonstrating population size expansion in checkpointing
///
/// This example shows how the resume_with_modified_params function handles:
/// 1. Expanding the reference set when population_size is increased
/// 2. Warning when population_size is decreased but continuing with original set
#[cfg(feature = "checkpointing")]
use ndarray::{Array1, Array2, array};

#[cfg(feature = "checkpointing")]
use globalsearch::{
    local_solver::builders::LBFGSBuilder,
    oqnlp::OQNLP,
    problem::Problem,
    types::{CheckpointConfig, EvaluationError, LocalSolverType, OQNLPParams},
};

#[cfg(feature = "checkpointing")]
use std::path::PathBuf;

#[cfg(not(feature = "checkpointing"))]
fn main() {
    println!("This example requires the 'checkpointing' feature to be enabled.");
    println!("Run with: cargo run --example population_size_expansion --features checkpointing");
}

#[cfg(feature = "checkpointing")]
#[derive(Debug, Clone)]
pub struct SixHumpCamel;

#[cfg(feature = "checkpointing")]
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

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

#[cfg(feature = "checkpointing")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = SixHumpCamel;

    // Clean up any existing checkpoints
    let checkpoint_dir = PathBuf::from("./checkpoints_population_test");
    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
    }

    println!("=== Example for Population Size Expansion ===\n");

    // Stage 1: Run with smaller population size
    let initial_params = OQNLPParams {
        iterations: 50,
        population_size: 100, // Start with smaller population
        wait_cycle: 10,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default().build(),
        seed: 42,
    };

    let checkpoint_config = CheckpointConfig {
        checkpoint_dir: checkpoint_dir.clone(),
        checkpoint_name: "population_example".to_string(),
        save_frequency: 25,
        keep_all: false,
        auto_resume: false,
    };

    println!("Stage 1: Running with population_size = {}", initial_params.population_size);

    let mut oqnlp = OQNLP::new(problem.clone(), initial_params)?
        .with_checkpointing(checkpoint_config.clone())?
        .verbose();

    let solution_set1 = oqnlp.run()?;
    println!("Stage 1 completed. Found {} solutions.", solution_set1.len());
    println!("Best objective: {:.8}\n", solution_set1.best_solution().unwrap().objective);

    // Stage 2: Resume with expanded population size
    let expanded_params = OQNLPParams {
        iterations: 100,      // More iterations
        population_size: 200, // Expand population
        wait_cycle: 10,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default().build(),
        seed: 42,
    };

    println!(
        "Stage 2: Resuming with expanded population_size = {}",
        expanded_params.population_size
    );

    let mut oqnlp2 = OQNLP::new(problem.clone(), expanded_params.clone())?
        .with_checkpointing(checkpoint_config.clone())?
        .verbose();

    // This should expand the reference set from 100 to 200 points
    let resumed = oqnlp2.resume_with_modified_params(expanded_params)?;
    if resumed {
        println!("Successfully resumed from checkpoint with expanded population!");
        let solution_set2 = oqnlp2.run()?;
        println!("Stage 2 completed. Found {} solutions.", solution_set2.len());
        println!("Best objective: {:.8}\n", solution_set2.best_solution().unwrap().objective);
    } else {
        println!("Failed to resume from checkpoint.");
    }

    // Stage 3: Test with reduced population size (should show warning)
    let reduced_params = OQNLPParams {
        iterations: 40,      // Keep iterations <= population_size
        population_size: 50, // Reduce population (should warn)
        wait_cycle: 10,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default().build(),
        seed: 42,
    };

    println!(
        "Stage 3: Resuming with reduced population_size = {} (should show warning)",
        reduced_params.population_size
    );

    let mut oqnlp3 = OQNLP::new(problem, reduced_params.clone())?
        .with_checkpointing(checkpoint_config)?
        .verbose();

    // This should issue a warning but continue with the original reference set
    let resumed2 = oqnlp3.resume_with_modified_params(reduced_params)?;
    if resumed2 {
        println!("Successfully resumed from checkpoint despite smaller population!");
        let solution_set3 = oqnlp3.run()?;
        println!("Stage 3 completed. Found {} solutions.", solution_set3.len());
        println!("Best objective: {:.8}", solution_set3.best_solution().unwrap().objective);
    } else {
        println!("Failed to resume from checkpoint.");
    }

    // Clean up
    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
    }

    println!("\n=== Population Size Expansion Example Completed ===");
    Ok(())
}
