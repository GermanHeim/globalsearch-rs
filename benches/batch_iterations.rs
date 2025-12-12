//! # Benchmarking batch_iterations with different batch sizes
//! and comparing sequential vs parallel execution.
//!
//! This benchmark evaluates the performance impact of the `batch_iterations`
//! feature in the OQNLP algorithm using the Six-Hump Camel function and a more
//! computationally expensive problem.
//!
//! ## Batch Sizes Tested
//! - 1 (sequential, no parallelism)
//! - 4 (moderate parallelism)
//! - 8 (high parallelism)
//! - 16 (very high parallelism)
//!
//! Run the benchmark with:
//! ```bash
//! cargo bench --features rayon --bench batch_iterations
//! ```
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use globalsearch::{
    oqnlp::OQNLP,
    problem::Problem,
    types::{EvaluationError, OQNLPParams},
};
use ndarray::{Array1, Array2, array};
use std::hint::black_box;

#[cfg(not(feature = "rayon"))]
compile_error!(
    "This benchmark requires the 'rayon' feature. Run with: cargo bench --features rayon"
);

// Six-Hump Camel problem for benchmarking
#[derive(Clone)]
struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let x1 = x[0];
        let x2 = x[1];
        let term1 = (4.0 - 2.1 * x1.powi(2) + x1.powi(4) / 3.0) * x1.powi(2);
        let term2 = x1 * x2;
        let term3 = (-4.0 + 4.0 * x2.powi(2)) * x2.powi(2);
        Ok(term1 + term2 + term3)
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

// A more computationally expensive problem for testing parallel benefits
#[derive(Clone)]
struct ExpensiveProblem;

impl Problem for ExpensiveProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Add computational overhead to make parallelization beneficial
        let mut result = 0.0;
        for i in 0..x.len() {
            for j in 0..100 {
                // Add computational overhead
                result += (x[i] * j as f64).sin().powi(2) + (x[i] * j as f64).cos().powi(2);
            }
        }
        Ok(result + x.iter().map(|&xi| xi.powi(2)).sum::<f64>())
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![
            [-5.0 + 1.0, 5.0 + 1.0],
            [-5.0 + 1.0, 5.0 + 1.0],
            [-5.0 + 1.0, 5.0 + 1.0],
            [-5.0 + 1.0, 5.0 + 1.0]
        ]
    }
}

fn benchmark_batch_iterations_sixhump(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_iterations_sixhump");
    group.measurement_time(std::time::Duration::from_secs(120));

    let problem = SixHumpCamel;
    let base_params = OQNLPParams { iterations: 2500, population_size: 5000, ..Default::default() };

    // Test different batch sizes
    let batch_sizes = vec![1, 4, 8, 16];

    for &batch_size in &batch_sizes {
        let batch_type = if batch_size == 1 { "sequential" } else { "parallel" };

        group.bench_with_input(BenchmarkId::new(batch_type, batch_size), &batch_size, |b, _| {
            b.iter(|| {
                #[cfg(feature = "rayon")]
                {
                    let mut oqnlp = OQNLP::new(problem.clone(), base_params.clone())
                        .unwrap()
                        .batch_iterations(batch_size);

                    black_box(oqnlp.run().unwrap())
                }
                #[cfg(not(feature = "rayon"))]
                {
                    let mut oqnlp = OQNLP::new(problem.clone(), base_params.clone()).unwrap();
                    black_box(oqnlp.run().unwrap())
                }
            })
        });
    }

    group.finish();
}

fn benchmark_batch_iterations_expensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_iterations_expensive");
    group.measurement_time(std::time::Duration::from_secs(120));

    let problem = ExpensiveProblem;
    let base_params = OQNLPParams { iterations: 2500, population_size: 5000, ..Default::default() };

    // Test different batch sizes for expensive problem
    let batch_sizes = vec![1, 4, 8, 16];

    for &batch_size in &batch_sizes {
        let batch_type = if batch_size == 1 { "sequential" } else { "parallel" };

        group.bench_with_input(BenchmarkId::new(batch_type, batch_size), &batch_size, |b, _| {
            b.iter(|| {
                #[cfg(feature = "rayon")]
                {
                    let mut oqnlp = OQNLP::new(problem.clone(), base_params.clone())
                        .unwrap()
                        .batch_iterations(batch_size);

                    black_box(oqnlp.run().unwrap())
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_batch_iterations_sixhump, benchmark_batch_iterations_expensive,);
criterion_main!(benches);
