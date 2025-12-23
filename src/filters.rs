//! # Solution Filtering Module
//!
//! This module implements filtering mechanisms used in the OQNLP algorithm to maintain
//! solution quality and diversity throughout the optimization process.
//!
//! ## Filtering Strategies
//!
//! The OQNLP algorithm employs two complementary filtering mechanisms:
//!
//! ### Merit Filter ([`MeritFilter`])
//! Controls solution acceptance based on objective function value quality:
//! - Maintains a dynamic threshold for solution acceptance
//! - Rejects solutions with objective values above the threshold
//! - Threshold adapts based on search progress and parameter settings
//! - Ensures only improving or competitive solutions are retained
//!
//! ### Distance Filter ([`DistanceFilter`])
//! Enforces minimum separation between solutions to maintain diversity:
//! - Prevents clustering of solutions in small regions
//! - Uses Euclidean distance as the separation metric
//! - Configurable minimum distance via `distance_factor` parameter
//! - Essential for maintaining exploration capability
//!
//! ## Usage in OQNLP
//!
//! Both filters work together during the optimization process:
//! 1. **Merit Filtering**: Solutions must pass objective value threshold
//! 2. **Distance Filtering**: Remaining solutions must maintain minimum separation
//! 3. **Reference Set Update**: Filtered solutions update the population
//!
//! ## Example
//!
//! ```rust
//! use globalsearch::filters::{MeritFilter, DistanceFilter};
//! use globalsearch::types::{FilterParams, LocalSolution};
//! use ndarray::array;
//!
//! // Create a merit filter
//! let mut merit_filter = MeritFilter::new();
//! merit_filter.update_threshold(-10.0);
//!
//! // Create a distance filter
//! let filter_params = FilterParams {
//!     distance_factor: 0.5,
//!     wait_cycle: 10,
//!     threshold_factor: 0.1,
//! };
//! let mut distance_filter = DistanceFilter::new(filter_params)?;
//!
//! // Check if a solution passes both filters
//! let candidate_value = -12.0;
//! let candidate_point = array![1.0, 2.0];
//!
//! if merit_filter.check(candidate_value) && distance_filter.check(&candidate_point) {
//!     println!("Solution accepted!");
//!     distance_filter.add_solution(LocalSolution {
//!         point: candidate_point,
//!         objective: candidate_value,
//!     });
//! }
//! # Ok::<(), globalsearch::filters::FiltersErrors>(())
//! ```

use crate::types::{FilterParams, LocalSolution};
use ndarray::Array1;
use thiserror::Error;

#[derive(Debug, Error)]
/// Filters errors
pub enum FiltersErrors {
    /// Distance factor must be positive or equal to zero
    #[error("Distance factor must be positive or equal to zero, got {0}.")]
    NegativeDistanceFactor(f64),
}

/// Quality-based filter for controlling solution acceptance in optimization.
///
/// The `MeritFilter` implements a dynamic threshold mechanism that determines
/// whether candidate solutions should be accepted based on their objective
/// function values. This filter is crucial for maintaining solution quality
/// and preventing the acceptance of poor solutions.
///
/// ## Mechanism
///
/// - **Threshold Management**: Maintains a dynamic acceptance threshold
/// - **Adaptive Behavior**: Threshold can be updated based on search progress
/// - **Quality Control**: Only solutions with objective values ≤ threshold pass
///
/// ## Example
///
/// ```rust
/// use globalsearch::filters::MeritFilter;
///
/// let mut filter = MeritFilter::new();
/// assert!(filter.check(1000.0)); // Initially accepts everything
///
/// // Set threshold based on best solution found
/// filter.update_threshold(-5.0);
/// assert!(filter.check(-6.0));   // Better solution - accepted
/// assert!(!filter.check(-4.0));  // Worse solution - rejected
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
pub struct MeritFilter {
    pub threshold: f64,
}

impl Default for MeritFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl MeritFilter {
    /// Create a new MeritFilter
    pub fn new() -> Self {
        Self { threshold: f64::INFINITY }
    }

    pub fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Check if the given value is below the threshold
    pub fn check(&self, value: f64) -> bool {
        value <= self.threshold
    }
}

/// Diversity-preserving filter that enforces minimum separation between solutions.
///
/// The `DistanceFilter` maintains population diversity by rejecting candidate
/// solutions that are too close to existing solutions. This prevents the
/// optimization algorithm from clustering solutions in small regions of the
/// search space.
///
/// ## Diversity Mechanism
///
/// - **Distance Calculation**: Uses squared Euclidean distance for efficiency
/// - **Minimum Separation**: Enforced via configurable `distance_factor`
/// - **Solution Storage**: Maintains internal collection of accepted solutions
/// - **Rejection Criterion**: New solutions must be far enough from all existing ones
///
/// ## Distance Calculation
///
/// For a candidate point **x** and existing solution **s**, the acceptance condition is:
///
/// ```text
/// ||x - s||² > distance_factor²
/// ```
///
/// This must hold for ALL existing solutions for the candidate to be accepted.
///
/// ## Example
///
/// ```rust
/// use globalsearch::filters::DistanceFilter;
/// use globalsearch::types::{FilterParams, LocalSolution};
/// use ndarray::array;
///
/// let params = FilterParams {
///     distance_factor: 0.5,
///     wait_cycle: 10,
///     threshold_factor: 0.1,
/// };
///
/// let mut filter = DistanceFilter::new(params)?;
///
/// // Add first solution
/// let solution1 = LocalSolution {
///     point: array![0.0, 0.0],
///     objective: -1.0,
/// };
/// filter.add_solution(solution1);
///
/// // Check if new points maintain sufficient distance
/// assert!(!filter.check(&array![0.1, 0.1])); // Too close - rejected
/// assert!(filter.check(&array![1.0, 1.0]));  // Far enough - accepted
/// # Ok::<(), globalsearch::filters::FiltersErrors>(())
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "checkpointing", derive(serde::Serialize, serde::Deserialize))]
pub struct DistanceFilter {
    solutions: Vec<LocalSolution>, // TODO: Change to ndarray?
    params: FilterParams,
}

impl DistanceFilter {
    /// # Create a new DistanceFilter with the given parameters
    ///
    /// Create a new DistanceFilter with the given parameters and an empty solution set
    /// to store the solutions.
    ///
    /// ## Errors
    ///
    /// Returns an error if the distance factor is negative
    pub fn new(params: FilterParams) -> Result<Self, FiltersErrors> {
        if params.distance_factor < 0.0 {
            return Err(FiltersErrors::NegativeDistanceFactor(params.distance_factor));
        }

        Ok(Self {
            solutions: Vec::new(), // Use ndarray?
            params,
        })
    }

    /// Add a solution to DistanceFilter
    pub fn add_solution(&mut self, solution: LocalSolution) {
        self.solutions.push(solution);
    }

    /// Check if the given point is far enough from all solutions in DistanceFilter
    pub fn check(&self, point: &Array1<f64>) -> bool {
        self.solutions.iter().all(|s| {
            euclidean_distance_squared(point, &s.point)
                > self.params.distance_factor * self.params.distance_factor
        })
    }

    /// Get the current solutions stored in the filter
    #[cfg(feature = "checkpointing")]
    pub fn get_solutions(&self) -> &Vec<LocalSolution> {
        &self.solutions
    }

    /// Restore solutions from a checkpoint
    #[cfg(feature = "checkpointing")]
    pub fn set_solutions(&mut self, solutions: Vec<LocalSolution>) {
        self.solutions = solutions;
    }
}

/// Compute the squared Euclidean distance between two points
#[inline]
fn euclidean_distance_squared(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff = a - b;
    diff.dot(&diff)
}

#[cfg(test)]
mod test_filters {
    use super::*;
    use ndarray::array;

    #[test]
    /// Test the invalid distance factor for the Distance Filter
    fn test_filter_params_invalid_distance_factor() {
        let params: FilterParams = FilterParams {
            distance_factor: -0.5, // Distance Factor should be greater or equal to 0.0
            wait_cycle: 10,
            threshold_factor: 0.1,
        };

        let df: Result<DistanceFilter, FiltersErrors> = DistanceFilter::new(params);

        assert!(matches!(df, Err(FiltersErrors::NegativeDistanceFactor(-0.5))));
    }

    #[test]
    /// Test updating MeritFilter threshold
    fn test_merit_filter_update_threshold() {
        let mut filter = MeritFilter::new();
        filter.update_threshold(10.0);
        assert_eq!(filter.threshold, 10.0);
    }

    #[test]
    /// Test valid construction of DistanceFilter
    fn test_distance_filter_valid() {
        let params = FilterParams { distance_factor: 1.0, wait_cycle: 5, threshold_factor: 0.2 };

        let filter = DistanceFilter::new(params).unwrap();
        assert_eq!(filter.params.distance_factor, 1.0);
        assert_eq!(filter.solutions.len(), 0);
    }

    #[test]
    /// Test adding solutions to DistanceFilter
    fn test_distance_filter_add_solution() {
        let params = FilterParams { distance_factor: 1.0, wait_cycle: 5, threshold_factor: 0.2 };

        let mut filter = DistanceFilter::new(params).unwrap();
        let solution = LocalSolution { point: array![1.0, 2.0, 3.0], objective: 5.0 };

        filter.add_solution(solution);
        assert_eq!(filter.solutions.len(), 1);
        assert_eq!(filter.solutions[0].objective, 5.0);
    }

    #[test]
    /// Test distance check
    fn test_distance_filter_check() {
        let params = FilterParams { distance_factor: 2.0, wait_cycle: 5, threshold_factor: 0.2 };

        let mut filter = DistanceFilter::new(params).unwrap();

        filter.add_solution(LocalSolution { point: array![0.0, 0.0, 0.0], objective: 5.0 });

        // Point is at distance 1.73... from origin, which is less than distance_factor=2.0
        assert!(!filter.check(&array![1.0, 1.0, 1.0]));

        // Point is at distance 5.2 from origin, which is greater than distance_factor=2.0
        assert!(filter.check(&array![3.0, 4.0, 3.0]));
    }
}
