//! # Parallel vs Sequential Performance Benchmark
//!
//! This benchmark compares the performance of ScatterSearch algorithms
//! with parallel processing enabled vs disabled using Rayon.
//!
//! ## Benchmark Overview
//!
//! The benchmark evaluates performance characteristics across different scenarios:
//!
//! - **ScatterSearch Algorithm**: Core scatter search with parallel control
//! - **Different Population Sizes**: Small, medium, and large populations
//! - **Parallel vs Sequential**: Direct comparison of execution modes
//!
//! Run the benchmark with:
//! ```bash
//! cargo bench --features rayon --bench parallel_performance
//! ```
#[cfg(not(feature = "rayon"))]
compile_error!(
    "This benchmark requires the 'rayon' feature. Run with: cargo bench --features rayon"
);

use criterion::{criterion_group, criterion_main, Criterion};
use globalsearch::problem::Problem;
use globalsearch::scatter_search::ScatterSearch;
use globalsearch::types::{EvaluationError, OQNLPParams};
use ndarray::{array, Array1, Array2};
use std::hint::black_box;

// Test problems for benchmarking
#[derive(Clone)]
struct SixHumpCamel;

impl Problem for SixHumpCamel {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Ok((4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
            + x[0] * x[1]
            + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2))
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-3.0, 3.0], [-2.0, 2.0]]
    }
}

#[derive(Clone)]
struct DummyProblem {
    dims: usize,
}

impl DummyProblem {
    fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Problem for DummyProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Minimal computation - just sum of squares
        Ok(x.iter().map(|&xi| xi * xi).sum())
    }

    fn variable_bounds(&self) -> Array2<f64> {
        Array2::from_shape_vec((self.dims, 2), vec![-1.0, 1.0].repeat(self.dims)).unwrap()
    }
}

// Benchmark Functions

/// Benchmark ScatterSearch performance with parallel vs sequential execution
fn bench_scatter_search_parallel_vs_sequential(c: &mut Criterion) {
    let params = OQNLPParams { population_size: 5000, ..OQNLPParams::default() };

    c.bench_function("scatter_search_parallel", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(true); // Enable parallel processing
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    c.bench_function("scatter_search_sequential", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(false); // Disable parallel processing
            let result = ss.run().unwrap();
            black_box(result);
        });
    });
}

/// Benchmark parallel overhead at small scales
fn bench_parallel_overhead_small_scale(c: &mut Criterion) {
    let params = OQNLPParams {
        population_size: 100, // Small population where overhead should dominate
        ..OQNLPParams::default()
    };

    c.bench_function("small_scale_parallel", |b| {
        b.iter(|| {
            let problem = DummyProblem::new(2);
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(true);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    c.bench_function("small_scale_sequential", |b| {
        b.iter(|| {
            let problem = DummyProblem::new(2);
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(false);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });
}

/// Benchmark parallel scaling with large populations
fn bench_parallel_scaling_large_population(c: &mut Criterion) {
    let params = OQNLPParams {
        population_size: 15000, // Large population to maximize parallel benefits
        ..OQNLPParams::default()
    };

    c.bench_function("large_scale_parallel", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(true);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    c.bench_function("large_scale_sequential", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(false);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });
}

/// Benchmark with different computational loads
fn bench_computational_load_comparison(c: &mut Criterion) {
    let params = OQNLPParams { population_size: 2000, ..OQNLPParams::default() };

    // SixHump Camel (moderate computation)
    c.bench_function("sixhump_parallel", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(true);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    c.bench_function("sixhump_sequential", |b| {
        b.iter(|| {
            let problem = SixHumpCamel;
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(false);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    // Dummy Problem (minimal computation)
    c.bench_function("dummy_parallel", |b| {
        b.iter(|| {
            let problem = DummyProblem::new(2);
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(true);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });

    c.bench_function("dummy_sequential", |b| {
        b.iter(|| {
            let problem = DummyProblem::new(2);
            let ss = ScatterSearch::new(problem, params.clone()).unwrap().parallel(false);
            let result = ss.run().unwrap();
            black_box(result);
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .sample_size(15)
        .measurement_time(std::time::Duration::from_secs(60));
    targets =
        bench_scatter_search_parallel_vs_sequential,
        bench_parallel_overhead_small_scale,
        bench_parallel_scaling_large_population,
        bench_computational_load_comparison
}

criterion_main!(benches);
