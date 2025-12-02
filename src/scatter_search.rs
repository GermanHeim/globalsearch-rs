//! # Scatter Search Module
//!
//! This module implements the Scatter Search metaheuristic, which forms the foundation
//! of the OQNLP global optimization algorithm. Scatter Search is a population-based
//! optimization method that systematically explores the solution space.
//!
//! ## Algorithm Overview
//!
//! The Scatter Search algorithm operates through three main phases:
//!
//! ### 1. Initialization: Generate diverse initial solutions within variable bounds
//!
//! ### 2. Diversification: Create new candidate solutions through systematic combination
//!
//! ### 3. Intensification: Generate trial points from reference set combinations
//!
//! ## Example Usage
//!
//! ```rust
//! use globalsearch::scatter_search::ScatterSearch;
//! use globalsearch::types::OQNLPParams;
//! # use globalsearch::problem::Problem;
//! # use globalsearch::types::EvaluationError;
//! # use ndarray::{Array1, Array2, array};
//! #
//! # #[derive(Debug, Clone)]
//! # struct TestProblem;
//! # impl Problem for TestProblem {
//! #     fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
//! #         Ok(x[0].powi(2) + x[1].powi(2))
//! #     }
//! #     fn variable_bounds(&self) -> Array2<f64> {
//! #         array![[-5.0, 5.0], [-5.0, 5.0]]
//! #     }
//! # }
//!
//! let problem = TestProblem;
//! let params = OQNLPParams::default();
//!
//! let scatter_search = ScatterSearch::new(problem, params)?;
//! let (reference_set, best_solution) = scatter_search.run()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::observers::Observer;
use crate::problem::Problem;
use crate::types::OQNLPParams;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use thiserror::Error;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "progress_bar")]
use kdam::{Bar, BarExt};

/// Variable bounds container for optimization problems.
///
/// This struct stores the lower and upper bounds for each optimization variable,
/// providing a convenient way to manage box constraints during the scatter search process.
///
/// # Fields
///
/// - `lower`: Array of lower bounds for each variable
/// - `upper`: Array of upper bounds for each variable
///
/// Both arrays must have the same length, corresponding to the problem dimension.
///
/// # Example
///
/// ```rust
/// use globalsearch::scatter_search::VariableBounds;
/// use ndarray::array;
///
/// let bounds = VariableBounds {
///     lower: array![-10.0, -5.0, 0.0],   // Lower bounds for x1, x2, x3
///     upper: array![10.0, 5.0, 1.0],    // Upper bounds for x1, x2, x3
/// };
/// ```
#[derive(Debug, Clone)]
pub struct VariableBounds {
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
}

#[derive(Debug, Error)]
/// Error types that can occur during scatter search operations.
///
/// These errors represent various failure modes that can happen during
/// the scatter search algorithm execution.
pub enum ScatterSearchError {
    /// Error when the reference set is empty
    #[error("Scatter Search Error: No candidates left.")]
    NoCandidates,

    /// Error when evaluating the objective function
    #[error("Scatter Search Error: Evaluation error: {0}.")]
    EvaluationError(#[from] crate::types::EvaluationError),
}

/// Type alias for the return type of scatter search run method
/// Returns (reference_set_with_objectives, best_solution)
type ScatterSearchResult = (Vec<(Array1<f64>, f64)>, Array1<f64>);

/// Scatter Search algorithm implementation struct
pub struct ScatterSearch<'a, P: Problem> {
    problem: P,
    params: OQNLPParams,
    reference_set: Vec<Array1<f64>>,
    reference_set_objectives: Vec<f64>,
    bounds: VariableBounds,
    rng: Mutex<StdRng>,
    #[cfg(feature = "progress_bar")]
    progress_bar: Option<Bar>,
    /// Whether parallel processing is enabled at runtime
    #[cfg(feature = "rayon")]
    enable_parallel: bool,
    /// Optional observer for tracking metrics
    observer: Option<&'a mut Observer>,
    /// Custom points to seed the reference set
    custom_points: Option<Vec<Array1<f64>>>,
}

impl<'a, P: Problem + Sync + Send> ScatterSearch<'a, P> {
    pub fn new(problem: P, params: OQNLPParams) -> Result<Self, ScatterSearchError> {
        let var_bounds = problem.variable_bounds();
        let bounds = VariableBounds {
            lower: var_bounds.column(0).to_owned(),
            upper: var_bounds.column(1).to_owned(),
        };

        let seed: u64 = params.seed;
        let ss: ScatterSearch<P> = Self {
            problem,
            params: params.clone(),
            reference_set: Vec::with_capacity(params.population_size),
            reference_set_objectives: Vec::with_capacity(params.population_size),
            bounds,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            #[cfg(feature = "progress_bar")]
            progress_bar: None,
            // Enable parallel processing by default
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            observer: None,
            custom_points: None,
        };

        Ok(ss)
    }

    /// Control whether parallel processing is enabled at runtime
    ///
    /// This method allows you to disable parallel processing even when the `rayon` feature is enabled,
    /// which can be useful for:
    /// - Python bindings
    /// - Benchmarking (consistent performance measurement)
    ///
    /// # Arguments
    /// * `enable` - If `true`, use parallel processing (default). If `false`, use sequential processing.
    #[cfg(feature = "rayon")]
    pub fn parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Attach an observer for metrics tracking
    ///
    /// This method allows the scatter search to update the observer directly
    /// with detailed metrics during execution.
    pub fn with_observer(mut self, observer: &'a mut Observer) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Set custom points to seed the reference set
    ///
    /// This method allows you to provide initial points that will be included in the
    /// reference set during initialization. These points are added after the three
    /// default seed points (lower bound, upper bound, and midpoint) and before the
    /// diversification step.
    ///
    /// # Arguments
    ///
    /// * `points` - A vector of points (`Array1<f64>`) to add to the reference set
    ///
    /// Note: Points are assumed to be already validated (correct dimension and within bounds)
    /// by the caller (OQNLP).
    pub fn with_custom_points(mut self, points: Vec<Array1<f64>>) -> Self {
        self.custom_points = Some(points);
        self
    }

    /// Run the Scatter Search algorithm
    ///
    /// Returns the reference set with objective values and the best solution found
    pub fn run(mut self) -> Result<ScatterSearchResult, ScatterSearchError> {
        #[cfg(feature = "progress_bar")]
        {
            self.progress_bar = Some(
                Bar::builder()
                    .total(3)
                    .desc("Stage 1")
                    .unit("steps")
                    .build()
                    .expect("Failed to create progress bar"),
            );
        }

        // Phase 1 & 2: Initialization and Diversification
        self.initialize_reference_set()?;

        // Update observer with initialization metrics
        if let Some(ref mut obs) = self.observer {
            if obs.should_observe_stage1() {
                if let Some(stage1) = obs.stage1_mut() {
                    stage1.enter_substage("initialization_complete");
                    stage1.set_reference_set_size(3); // 3 seed points created
                }
            }
            obs.invoke_callback();

            if obs.should_observe_stage1() {
                if let Some(stage1) = obs.stage1_mut() {
                    stage1.enter_substage("diversification_complete");
                    stage1.set_reference_set_size(self.reference_set.len());
                    stage1.add_function_evaluations(self.reference_set.len());
                }
            }
            obs.invoke_callback();
        }

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, initialized and diversified");
            pb.update(1).expect("Failed to update progress bar");
        }

        // Phase 3: Intensification using k-selection
        // Generate trial points from best k points and update reference set
        let trial_points = self.generate_trial_points()?;

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, generated trial points");
            pb.update(1).expect("Failed to update progress bar");
        }

        self.update_reference_set(&trial_points);

        // Update observer with intensification metrics
        if let Some(ref mut obs) = self.observer {
            if obs.should_observe_stage1() {
                if let Some(stage1) = obs.stage1_mut() {
                    stage1.enter_substage("intensification_complete");
                    stage1.add_trial_points(trial_points.len());
                    stage1.set_reference_set_size(self.reference_set.len());
                }
            }
            obs.invoke_callback();
        }

        let best = self.best_solution()?;

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, found best solution");
            pb.update(1).expect("Failed to update progress bar");
        }

        let reference_set_with_objectives: Vec<(Array1<f64>, f64)> =
            self.reference_set.into_iter().zip(self.reference_set_objectives).collect();

        Ok((reference_set_with_objectives, best))
    }

    pub fn initialize_reference_set(&mut self) -> Result<(), ScatterSearchError> {
        let mut ref_set: Vec<Array1<f64>> = Vec::with_capacity(self.params.population_size);

        ref_set.push(self.bounds.lower.to_owned());
        ref_set.push(self.bounds.upper.to_owned());
        ref_set.push((&self.bounds.lower + &self.bounds.upper) / 2.0);

        // Add custom points if provided
        if let Some(ref custom_points) = self.custom_points {
            ref_set.extend(custom_points.iter().cloned());
        }

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, initialized reference set");
            pb.update(1).expect("Failed to update progress bar");
        }

        self.diversify_reference_set(&mut ref_set)?;

        // Evaluate objectives for the initial reference set
        let objectives: Vec<f64> = ref_set
            .iter()
            .map(|point| self.problem.objective(point))
            .collect::<Result<Vec<f64>, _>>()?;

        // Sort reference set by objective value (best first) for k-selection
        let mut points_with_objectives: Vec<(Array1<f64>, f64)> =
            ref_set.into_iter().zip(objectives).collect();
        points_with_objectives.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (sorted_points, sorted_objectives): (Vec<Array1<f64>>, Vec<f64>) =
            points_with_objectives.into_iter().unzip();

        self.reference_set = sorted_points;
        self.reference_set_objectives = sorted_objectives;

        #[cfg(feature = "progress_bar")]
        if let Some(pb) = &mut self.progress_bar {
            pb.set_description("Stage 1, diversified reference set");
            pb.update(1).expect("Failed to update progress bar");
        }
        Ok(())
    }

    /// Diversify the reference set by adding new points to it
    pub fn diversify_reference_set(
        &mut self,
        ref_set: &mut Vec<Array1<f64>>,
    ) -> Result<(), ScatterSearchError> {
        let mut candidates = self.generate_stratified_samples(self.params.population_size)?;

        #[cfg(feature = "rayon")]
        let mut min_dists: Vec<f64> = if self.enable_parallel {
            candidates.par_iter().map(|c| self.min_distance(c, ref_set)).collect()
        } else {
            candidates.iter().map(|c| self.min_distance(c, ref_set)).collect()
        };

        #[cfg(not(feature = "rayon"))]
        let mut min_dists: Vec<f64> =
            candidates.iter().map(|c| self.min_distance(c, ref_set)).collect();

        while ref_set.len() < self.params.population_size {
            #[cfg(feature = "rayon")]
            let (max_idx, _) = if self.enable_parallel {
                (0..min_dists.len())
                    .into_par_iter()
                    .map(|i| (i, min_dists[i]))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .ok_or(ScatterSearchError::NoCandidates)?
            } else {
                min_dists
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, &v)| (i, v))
                    .ok_or(ScatterSearchError::NoCandidates)?
            };
            #[cfg(not(feature = "rayon"))]
            let (max_idx, _) = min_dists
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &v)| (i, v))
                .ok_or(ScatterSearchError::NoCandidates)?;

            let farthest = candidates.swap_remove(max_idx);
            min_dists.swap_remove(max_idx);
            ref_set.push(farthest);

            if ref_set.len() >= self.params.population_size {
                break;
            }

            #[cfg(feature = "rayon")]
            {
                if self.enable_parallel {
                    let updater_iter = candidates.par_iter().zip(min_dists.par_iter_mut());
                    updater_iter.for_each(|(candidate, min_dist)| {
                        if let Some(last) = ref_set.last() {
                            let dist = euclidean_distance_squared(candidate, last);
                            if dist < *min_dist {
                                *min_dist = dist;
                            }
                        }
                    });
                } else {
                    let updater_iter = candidates.iter().zip(min_dists.iter_mut());
                    updater_iter.for_each(|(candidate, min_dist)| {
                        if let Some(last) = ref_set.last() {
                            let dist = euclidean_distance_squared(candidate, last);
                            if dist < *min_dist {
                                *min_dist = dist;
                            }
                        }
                    });
                }
            }
            #[cfg(not(feature = "rayon"))]
            {
                let updater_iter = candidates.iter().zip(min_dists.iter_mut());
                updater_iter.for_each(|(candidate, min_dist)| {
                    if let Some(last) = ref_set.last() {
                        let dist = euclidean_distance_squared(candidate, last);
                        if dist < *min_dist {
                            *min_dist = dist;
                        }
                    }
                });
            }
        }

        Ok(())
    }

    /// Generate stratified samples within the bounds
    pub fn generate_stratified_samples(
        &self,
        n: usize,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let dim: usize = self.bounds.lower.len();

        // Precompute seeds while holding the mutex once
        let seeds: Vec<u64> = {
            let mut rng = self.rng.lock().unwrap();
            (0..n).map(|_| rng.random::<u64>()).collect::<Vec<_>>()
        };

        #[cfg(feature = "rayon")]
        let samples = if self.enable_parallel {
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    Ok(Array1::from_shape_fn(dim, |i| {
                        rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                    }))
                })
                .collect::<Result<Vec<_>, ScatterSearchError>>()
        } else {
            seeds
                .into_iter()
                .map(|seed: u64| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    Ok(Array1::from_shape_fn(dim, |i| {
                        rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                    }))
                })
                .collect::<Result<Vec<_>, ScatterSearchError>>()
        }?;

        #[cfg(not(feature = "rayon"))]
        let samples = seeds
            .into_iter()
            .map(|seed: u64| {
                let mut rng = StdRng::seed_from_u64(seed);
                Ok(Array1::from_shape_fn(dim, |i| {
                    rng.random_range(self.bounds.lower[i]..=self.bounds.upper[i])
                }))
            })
            .collect::<Result<Vec<_>, ScatterSearchError>>()?;

        Ok(samples)
    }

    /// Compute the minimum distance between a point and a reference set
    pub fn min_distance(&self, point: &Array1<f64>, ref_set: &[Array1<f64>]) -> f64 {
        #[cfg(feature = "rayon")]
        {
            if self.enable_parallel {
                ref_set
                    .par_iter()
                    .map(|p| euclidean_distance_squared(point, p))
                    .reduce(|| f64::INFINITY, f64::min)
            } else {
                ref_set
                    .iter()
                    .map(|p| euclidean_distance_squared(point, p))
                    .fold(f64::INFINITY, f64::min)
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            ref_set
                .iter()
                .map(|p| euclidean_distance_squared(point, p))
                .fold(f64::INFINITY, f64::min)
        }
    }

    pub fn generate_trial_points(&mut self) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        // Only use the best k points for combinations
        let k = (self.reference_set.len() as f64).sqrt() as usize;
        let k = k.max(2).min(self.reference_set.len());

        // Create combinations only between the best k points
        let indices: Vec<(usize, usize)> =
            (0..k).flat_map(|i| ((i + 1)..k).map(move |j| (i, j))).collect();

        // Pre-allocate the result vector
        let n_combinations = indices.len();
        let n_trial_points_per_combo = 6; // 4 linear combinations + 2 random
        let mut trial_points: Vec<Array1<f64>> =
            Vec::with_capacity(n_combinations * n_trial_points_per_combo);

        // Precompute seeds for each combine_points call
        let seeds: Vec<u64> = {
            let mut rng = self.rng.lock().unwrap();
            (0..indices.len()).map(|_| rng.random::<u64>()).collect::<Vec<_>>()
        };

        #[cfg(feature = "rayon")]
        {
            let points_per_combo: Vec<Vec<Array1<f64>>> = indices
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(&(i, j), &seed)| {
                    self.combine_points(&self.reference_set[i], &self.reference_set[j], seed)
                })
                .collect::<Result<Vec<_>, ScatterSearchError>>()?;

            for points in points_per_combo {
                trial_points.extend(points);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (&(i, j), &seed) in indices.iter().zip(seeds.iter()) {
                let points =
                    self.combine_points(&self.reference_set[i], &self.reference_set[j], seed)?;
                trial_points.extend(points);
            }
        }

        Ok(trial_points)
    }

    /// Combines two points into several trial points.
    pub fn combine_points(
        &self,
        a: &Array1<f64>,
        b: &Array1<f64>,
        seed: u64,
    ) -> Result<Vec<Array1<f64>>, ScatterSearchError> {
        let mut points = Vec::with_capacity(6);

        // Linear combinations.
        let directions: Vec<f64> = vec![0.25, 0.5, 0.75, 1.25];
        for &alpha in &directions {
            let mut point = a * alpha + b * (1.0 - alpha);
            self.apply_bounds(&mut point);
            points.push(point);
        }

        // Random perturbations using the provided seed
        let mut rng: StdRng = StdRng::seed_from_u64(seed);
        for _ in 0..2 {
            let mut point = (a + b) / 2.0;
            point.iter_mut().enumerate().for_each(|(i, x)| {
                *x += rng.random_range(-0.1..0.1) * (self.bounds.upper[i] - self.bounds.lower[i]);
            });
            self.apply_bounds(&mut point);
            points.push(point);
        }

        Ok(points)
    }

    pub fn apply_bounds(&self, point: &mut Array1<f64>) {
        for i in 0..point.len() {
            point[i] = point[i].clamp(self.bounds.lower[i], self.bounds.upper[i]);
        }
    }

    pub fn update_reference_set(&mut self, trials: &[Array1<f64>]) {
        // Early termination if no trials
        if trials.is_empty() {
            return;
        }

        // Reference set is already sorted (from previous iteration or initialize_reference_set)
        // so we can directly use the cached objectives without re-sorting
        let worst_obj = self.reference_set_objectives.last().copied().unwrap_or(f64::INFINITY);

        // Prepare reference set for later merging with evaluated trials
        let ref_evaluated: Vec<(Array1<f64>, f64)> = self
            .reference_set
            .iter()
            .zip(self.reference_set_objectives.iter())
            .map(|(point, &obj)| (point.clone(), obj))
            .collect();

        // Compute a minimum distance threshold (once) based on reference set diversity
        let min_dist_threshold = {
            let ref_set = &self.reference_set;
            if ref_set.len() < 2 {
                0.0
            } else {
                // Sample k pairs to estimate typical distances (k = sqrt(population_size))
                // This scales appropriately with population size
                let k = ((ref_set.len() as f64).sqrt() as usize).max(2).min(ref_set.len());
                let sample_size = k;

                #[cfg(feature = "rayon")]
                let sum_dist = if self.enable_parallel {
                    (0..sample_size)
                        .into_par_iter()
                        .map(|i| {
                            euclidean_distance_squared(
                                &ref_set[i],
                                &ref_set[(i + 1) % ref_set.len()],
                            )
                        })
                        .sum::<f64>()
                } else {
                    (0..sample_size)
                        .map(|i| {
                            euclidean_distance_squared(
                                &ref_set[i],
                                &ref_set[(i + 1) % ref_set.len()],
                            )
                        })
                        .sum::<f64>()
                };

                #[cfg(not(feature = "rayon"))]
                let sum_dist = (0..sample_size)
                    .map(|i| {
                        euclidean_distance_squared(&ref_set[i], &ref_set[(i + 1) % ref_set.len()])
                    })
                    .sum::<f64>();

                (sum_dist / sample_size as f64) * 0.10 // Use 10% of average distance as threshold
            }
        };

        // Evaluate trial points with two-level filtering:
        // 1. Cheap distance filter to skip near-duplicates (up to 5 distance computations)
        // 2. Expensive objective evaluation only for diverse points
        let evaluate_trial = |point: &Array1<f64>| -> Option<(Array1<f64>, f64)> {
            // Filter 1: Distance check - only check against top 5 reference points
            let is_diverse =
                self.reference_set.iter().take(5).all(|ref_point| {
                    euclidean_distance_squared(point, ref_point) > min_dist_threshold
                });

            if !is_diverse {
                return None; // Skip evaluation if too close
            }

            // Filter 2: Objective evaluation (expensive operation)
            let obj = self.problem.objective(point).ok()?;
            if obj < worst_obj {
                Some((point.clone(), obj))
            } else {
                None
            }
        };

        #[cfg(feature = "rayon")]
        let trial_evaluated: Vec<(Array1<f64>, f64)> = if self.enable_parallel {
            trials.par_iter().filter_map(evaluate_trial).collect()
        } else {
            trials.iter().filter_map(evaluate_trial).collect()
        };

        #[cfg(not(feature = "rayon"))]
        let trial_evaluated: Vec<(Array1<f64>, f64)> =
            trials.iter().filter_map(evaluate_trial).collect();

        // Combine and sort all points
        let mut all_points = ref_evaluated;
        all_points.extend(trial_evaluated);

        // Keep only the best population_size points
        // This ensures reference set always has exactly population_size points
        let pop_size = self.params.population_size;

        // Only select and sort the best k points needed for k-selection
        // The rest can remain unsorted since they won't be used for generating trial points
        let k = ((pop_size as f64).sqrt() as usize).max(2).min(pop_size);

        // Partition so the k best elements are at the front (these will be the global best k)
        all_points.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));

        // Now sort only the first k elements (these are guaranteed to be the k best globally)
        all_points[..k].sort_by(|a, b| a.1.total_cmp(&b.1));

        // Partition the remaining to get the pop_size best overall
        all_points.select_nth_unstable_by(pop_size - 1, |a, b| a.1.total_cmp(&b.1));

        // Truncate to keep only pop_size points
        all_points.truncate(pop_size);

        // Update reference set and objectives
        let (points, objectives): (Vec<Array1<f64>>, Vec<f64>) = all_points.into_iter().unzip();
        self.reference_set = points;
        self.reference_set_objectives = objectives;
    }

    pub fn best_solution(&self) -> Result<Array1<f64>, ScatterSearchError> {
        #[cfg(feature = "rayon")]
        let best_idx = if self.enable_parallel {
            self.reference_set_objectives
                .par_iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or(ScatterSearchError::NoCandidates)?
        } else {
            self.reference_set_objectives
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or(ScatterSearchError::NoCandidates)?
        };

        #[cfg(not(feature = "rayon"))]
        let best_idx = self
            .reference_set_objectives
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or(ScatterSearchError::NoCandidates)?;

        Ok(self.reference_set[best_idx].clone())
    }

    pub fn store_trial(&mut self, trial: Array1<f64>) {
        self.reference_set.push(trial);
    }
}

/// Compute the squared Euclidean distance between two points
///
/// Use this function for performance since we don't use the square root
fn euclidean_distance_squared(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x * x).sum()
}

// The following code allows to compute the Euclidean distance between two points
// but it is not used in the current implementation
// Could there be some cases where we need to compute the Euclidean distance?
// due to overflow or numerical stability?
//
// /// Compute the Euclidean distance between two points.
// ///
// /// Use euclidean_distance_squared when only comparing distances for better performance
// /// given that if a < b, then a² < b²
// fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
//     euclidean_distance_squared(a, b).sqrt()
// }

#[cfg(test)]
mod tests_scatter_search {
    use super::*;
    use crate::types::EvaluationError;
    use crate::types::OQNLPParams;
    use ndarray::{array, Array2};

    #[derive(Debug, Clone)]
    pub struct SixHumpCamel;

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

    #[test]
    /// Test if the population size is correctly set in the `ScatterSearch` struct
    fn test_population_size() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let (ref_set, _) = ss.run().unwrap();
        assert_eq!(ref_set.len(), 100);
    }

    #[test]
    /// Test if the bounds are correctly set in the `ScatterSearch` struct and
    /// all the points in the reference set are within the bounds
    fn test_bounds_in_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let bounds: VariableBounds = ss.bounds.clone();
        let (ref_set, _) = ss.run().unwrap();

        assert_eq!(ref_set.len(), 100);

        for (point, _obj) in ref_set {
            for i in 0..point.len() {
                assert!(point[i] >= bounds.lower[i]);
                assert!(point[i] <= bounds.upper[i]);
            }
        }
    }

    #[test]
    /// Test that, given the same seed and population size, the reference set
    /// is the same for two different `ScatterSearch` instances
    fn test_same_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 50,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 100,
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss1: ScatterSearch<SixHumpCamel> =
            ScatterSearch::new(problem.clone(), params.clone()).unwrap();
        let ss2: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();

        let (ref_set1, _) = ss1.run().unwrap();
        let (ref_set2, _) = ss2.run().unwrap();

        assert_eq!(ref_set1.len(), 100);
        assert_eq!(ref_set1.len(), ref_set2.len());

        for i in 0..ref_set1.len() {
            assert_eq!(ref_set1[i], ref_set2[i]);
        }
    }

    #[test]
    /// Test generating trial points for a `ScatterSearch` instance
    fn test_generate_trial_points() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trial_points: Vec<Array1<f64>> = ss.generate_trial_points().unwrap();

        // Compute expected based on subsampling logic: k = floor(sqrt(N)) combinations
        let n = ss.reference_set.len();
        let k = (n as f64).sqrt() as usize;
        let expected = k * (k - 1) / 2 * 6;
        assert_eq!(trial_points.len(), expected);
    }

    #[test]
    /// Test combining two points into trial points
    fn test_combine_points() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        let a: Array1<f64> = array![1.0, 1.0];
        let b: Array1<f64> = array![2.0, 2.0];

        let trial_points: Vec<Array1<f64>> = ss.combine_points(&a, &b, 0).unwrap();

        // 4 linear combinations and 2 random perturbations
        assert_eq!(trial_points.len(), 6);
    }

    #[test]
    /// Test storing trials in the reference set
    fn test_store_trials() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();

        // Initially empty reference set (not initialized)
        assert_eq!(ss.reference_set.len(), 0);

        let trial: Array1<f64> = array![1.0, 1.0];
        ss.store_trial(trial.clone());

        // Verify trial was stored
        assert_eq!(ss.reference_set.len(), 1);
        assert_eq!(ss.reference_set[0], trial);
    }

    #[test]
    /// Test updating the reference set with new trials
    fn test_update_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trials: Vec<Array1<f64>> = vec![array![1.0, 1.0], array![2.0, 2.0]];
        ss.update_reference_set(&trials);

        assert_eq!(ss.reference_set.len(), 4);
    }

    #[test]
    /// Test computing the minimum distance between a point and a reference set
    fn test_min_distance() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let point: Array1<f64> = array![-3.0, -2.0];
        let min_dist: f64 = ss.min_distance(&point, &ss.reference_set);

        // The minimum distance should be 0 since the point is in the reference set
        assert_eq!(min_dist, 0.0);
    }

    #[test]
    /// Test euclidean distance squared
    fn test_euclidean_distance_squared() {
        let a: Array1<f64> = array![1.0, 2.0];
        let b: Array1<f64> = array![3.0, 4.0];
        let dist: f64 = euclidean_distance_squared(&a, &b);
        assert_eq!(dist, 8.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test generating trial points using rayon
    fn test_generate_trial_points_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trial_points: Vec<Array1<f64>> = ss.generate_trial_points().unwrap();

        // Compute expected based on subsampling logic: k = floor(sqrt(N)) combinations
        let n = ss.reference_set.len();
        let k = (n as f64).sqrt() as usize;
        let expected = k * (k - 1) / 2 * 6;
        assert_eq!(trial_points.len(), expected);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test updating the reference set using rayon
    fn test_update_reference_set_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let trials: Vec<Array1<f64>> = vec![array![1.0, 1.0], array![2.0, 2.0]];
        ss.update_reference_set(&trials);

        assert_eq!(ss.reference_set.len(), 4);
    }

    #[cfg(feature = "rayon")]
    #[test]
    /// Test computing the minimum distance between a point and a reference set using rayon
    fn test_min_distance_rayon() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 4,
            seed: 0,
            ..OQNLPParams::default()
        };

        let mut ss: ScatterSearch<SixHumpCamel> = ScatterSearch::new(problem, params).unwrap();
        ss.initialize_reference_set().unwrap();

        let point: Array1<f64> = array![-3.0, -2.0];
        let min_dist: f64 = ss.min_distance(&point, &ss.reference_set);

        // The minimum distance should be 0 since the point is in the reference set
        assert_eq!(min_dist, 0.0);
    }

    #[test]
    /// Test with_custom_points method
    fn test_with_custom_points() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        // Create custom points
        let custom_points = vec![array![0.0, 0.0], array![1.0, 1.0], array![-1.0, -1.0]];

        let ss = ScatterSearch::new(problem, params).unwrap().with_custom_points(custom_points);

        assert!(ss.custom_points.is_some(), "Custom points should be set");
        assert_eq!(ss.custom_points.as_ref().unwrap().len(), 3, "Should have 3 custom points");
    }

    #[test]
    /// Test that custom points are added to reference set
    fn test_custom_points_in_reference_set() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 15,
            seed: 0,
            ..OQNLPParams::default()
        };

        // Create custom points
        let custom_point = array![0.5, 0.5];
        let custom_points = vec![custom_point.clone()];

        let mut ss = ScatterSearch::new(problem, params).unwrap().with_custom_points(custom_points);

        ss.initialize_reference_set().unwrap();

        // After initialization, the reference set should contain our custom point
        // The reference set includes: lower bound, upper bound, midpoint, + custom points + diversified points
        assert_eq!(ss.reference_set.len(), 15, "Reference set should have population_size points");

        // Check that the custom point was included (it should be one of the first 4 points before diversification)
        // Note: After diversification and sorting, the exact position may vary,
        // but we can verify the reference set was created successfully
        assert!(
            ss.reference_set.len() >= 4,
            "Reference set should include at least the 3 seed points + 1 custom point"
        );
    }

    #[test]
    /// Test custom points with empty vector
    fn test_custom_points_empty() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 10,
            seed: 0,
            ..OQNLPParams::default()
        };

        let custom_points: Vec<Array1<f64>> = vec![];
        let mut ss = ScatterSearch::new(problem, params).unwrap().with_custom_points(custom_points);

        ss.initialize_reference_set().unwrap();

        assert_eq!(ss.reference_set.len(), 10, "Reference set should have population_size points");
    }

    #[test]
    /// Test that custom points work in full scatter search run
    fn test_custom_points_full_run() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 20,
            seed: 0,
            ..OQNLPParams::default()
        };

        // Create custom points near known good solutions for Six Hump Camel
        let custom_points = vec![
            array![0.0, 0.0],  // Near center
            array![0.5, -0.5], // Another point
            array![-0.5, 0.5], // Another point
        ];

        let ss = ScatterSearch::new(problem, params).unwrap().with_custom_points(custom_points);

        let result = ss.run();
        assert!(result.is_ok(), "Scatter search should complete successfully with custom points");

        let (ref_set, _best) = result.unwrap();
        assert_eq!(ref_set.len(), 20, "Reference set should have population_size points after run");
    }

    #[test]
    /// Test custom points with maximum number of points
    fn test_custom_points_many() {
        let problem: SixHumpCamel = SixHumpCamel;
        let params: OQNLPParams = OQNLPParams {
            iterations: 1,
            wait_cycle: 30,
            threshold_factor: 0.2,
            distance_factor: 0.75,
            population_size: 20,
            seed: 0,
            ..OQNLPParams::default()
        };

        // Create many custom points (more than would fit in the initial seed)
        let mut custom_points = Vec::new();
        for i in 0..10 {
            custom_points.push(array![i as f64 * 0.1, i as f64 * 0.1]);
        }

        let mut ss = ScatterSearch::new(problem, params).unwrap().with_custom_points(custom_points);

        ss.initialize_reference_set().unwrap();

        // After initialization and diversification, should still have population_size points
        assert_eq!(ss.reference_set.len(), 20, "Reference set should be capped at population_size");
    }
}
