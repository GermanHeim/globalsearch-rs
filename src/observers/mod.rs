//! # Observers Module
//!
//! The observers module provides comprehensive monitoring and tracking capabilities
//! for the OQNLP global optimization algorithm. Observers allow you to monitor the
//! algorithm's progress in real-time, collect detailed metrics about each stage of
//! the optimization process, and implement custom logging or visualization.
//!
//! ## Overview
//!
//! The OQNLP algorithm operates in two main stages:
//!
//! 1. **Stage 1 (Scatter Search)**: Explores the parameter space using scatter search
//!    metaheuristics to identify promising regions and build an initial reference set.
//! 2. **Stage 2 (Iterative Refinement)**: Performs local optimization from multiple
//!    starting points, iteratively improving the solution set through merit filtering
//!    and distance-based selection.
//!
//! Observers track key metrics for each stage, providing insights into algorithm
//! behavior, convergence patterns, and computational efficiency.
//!
//! ## Key Features
//!
//! - **Real-time Monitoring**: Track algorithm progress with customizable callbacks
//! - **Detailed Metrics**: Comprehensive statistics for both optimization stages
//! - **Flexible Configuration**: Choose which stages and metrics to monitor
//! - **Performance Tracking**: Monitor function evaluations, timing, and convergence
//! - **Custom Callbacks**: Implement custom logging, visualization, or early stopping
//!
//! ## Architecture
//!
//! The observer system consists of three main components:
//!
//! - [`Observer`]: Main coordinator that manages tracking configuration and callbacks
//! - [`Stage1State`]: Tracks metrics during scatter search and reference set construction
//! - [`Stage2State`]: Tracks metrics during iterative local refinement
//!
//! ## Example Usage
//!
//! ```rust
//! use globalsearch::observers::Observer;
//!
//! // Create an observer with tracking for both stages
//! let observer = Observer::new()
//!     .with_stage1_tracking()
//!     .with_stage2_tracking()
//!     .with_timing()
//!     .with_default_callback();
//!
//! // Use with OQNLP optimizer (see OQNLP documentation for details)
//! // let mut optimizer = OQNLP::new(problem, params).unwrap().add_observer(observer);
//! // let solutions = optimizer.run();
//!
//! // After optimization, access observer metrics
//! // if let Some(observer) = optimizer.observer() {
//! //     if let Some(stage1) = observer.stage1_final() {
//! //         println!("Stage 1 completed with {} evaluations", stage1.function_evaluations());
//! //     }
//! //     if let Some(stage2) = observer.stage2() {
//! //         println!("Stage 2 found {} solutions", stage2.solution_set_size());
//! //     }
//! // }
//! ```

use std::sync::Arc;
use std::time::Instant;

mod stage1;
mod stage2;

pub use stage1::Stage1State;
pub use stage2::Stage2State;

/// Observer mode determines which stages to track
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverMode {
    /// Only track Stage 1 (reference set construction)
    Stage1Only,
    /// Only track Stage 2 (iterative improvement)
    Stage2Only,
    /// Track both stages
    Both,
}

/// Callback function type for observer updates
///
/// The callback receives a mutable reference to the Observer, allowing access to
/// all tracked metrics and modification of internal state during optimization.
pub type ObserverCallback = Arc<dyn Fn(&mut Observer) + Send + Sync>;

/// Previous Stage 2 state for change detection
#[derive(Debug, Clone, PartialEq)]
struct PreviousStage2State {
    best_objective: f64,
    solution_set_size: usize,
    threshold_value: f64,
    local_solver_calls: usize,
    improved_local_calls: usize,
    function_evaluations: usize,
    unchanged_cycles: usize,
}

impl PreviousStage2State {
    fn from_stage2(stage2: &Stage2State) -> Self {
        Self {
            best_objective: stage2.best_objective(),
            solution_set_size: stage2.solution_set_size(),
            threshold_value: stage2.threshold_value(),
            local_solver_calls: stage2.local_solver_calls(),
            improved_local_calls: stage2.improved_local_calls(),
            function_evaluations: stage2.function_evaluations(),
            unchanged_cycles: stage2.unchanged_cycles(),
        }
    }

    fn has_changed(&self, stage2: &Stage2State) -> bool {
        self.best_objective != stage2.best_objective()
            || self.solution_set_size != stage2.solution_set_size()
            || self.threshold_value != stage2.threshold_value()
            || self.local_solver_calls != stage2.local_solver_calls()
            || self.improved_local_calls != stage2.improved_local_calls()
            || self.function_evaluations != stage2.function_evaluations()
    }
}

/// Main observer struct that tracks algorithm state
///
/// The observer can be configured to track different metrics during
/// Stage 1 (reference set construction) and Stage 2 (iterative improvement).
/// It supports real-time monitoring through callbacks and provides detailed
/// statistics about algorithm performance and convergence.
///
/// # Configuration Options
///
/// Observers are configured using the builder pattern:
///
/// ```rust
/// use globalsearch::observers::Observer;
///
/// // Basic observer with no tracking
/// let observer = Observer::new();
///
/// // Track both stages with default logging
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_stage2_tracking()
///     .with_default_callback();
///
/// // Custom configuration
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_timing()
///     .with_callback(|obs| {
///         // Custom callback logic
///     });
/// ```
///
/// # Configuration Options
///
/// Observers are configured using the builder pattern:
///
/// ```rust
/// use globalsearch::observers::Observer;
///
/// // Basic observer with no tracking
/// let observer = Observer::new();
///
/// // Track both stages with default logging
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_stage2_tracking()
///     .with_default_callback();
///
/// // Custom configuration
/// let observer = Observer::new()
///     .with_stage1_tracking()
///     .with_timing()
///     .with_callback(|obs| {
///         // Custom callback logic
///     });
/// ```
///
/// # Observer Modes
///
/// The observer can operate in different modes to control which stages are tracked:
///
/// - [`ObserverMode::Both`]: Track both Stage 1 and Stage 2 (default)
/// - [`ObserverMode::Stage1Only`]: Track only Stage 1 scatter search
/// - [`ObserverMode::Stage2Only`]: Track only Stage 2 local refinement
///
/// # Callback System
///
/// Callbacks allow real-time monitoring of the optimization process. They receive
/// a reference to the observer and can access all tracked metrics. Callbacks can be:
///
/// - **Default callbacks**: Pre-built logging functions for common use cases
/// - **Custom callbacks**: User-defined functions for specialized monitoring
/// - **Frequency-controlled**: Callbacks can be invoked every N iterations
///
/// # Timing Information
///
/// When timing is enabled with `with_timing()`, the observer tracks:
///
/// - Total time spent in each stage
/// - Time spent in sub-phases within Stage 1
/// - Cumulative timing information accessible via `stage1_final()` and `stage2()`
///
/// # Accessing Metrics
///
/// Metrics can be accessed in two ways:
///
/// 1. **During optimization**: Via callbacks that receive the observer reference
/// 2. **After optimization**: Via the observer stored in the OQNLP instance
///
/// ```rust
/// use globalsearch::observers::Observer;
///
/// // During optimization (in callback)
/// let observer = Observer::new()
///     .with_stage2_tracking()
///     .with_callback(|obs| {
///         if let Some(stage2) = obs.stage2() {
///             println!("Current best: {}", stage2.best_objective());
///         }
///     });
/// ```
pub struct Observer {
    /// Observer mode determines which stages to track
    mode: ObserverMode,

    /// Stage 1 tracking state (None if not tracking Stage 1)
    stage1: Option<Stage1State>,

    /// Stage 2 tracking state (None if not tracking Stage 2)
    stage2: Option<Stage2State>,

    /// Whether to track timing information for stages
    track_timing: bool,

    /// Start time for the overall optimization (used for elapsed time calculations)
    start_time: Option<Instant>,

    /// Optional callback function invoked during optimization
    callback: Option<ObserverCallback>,

    /// Frequency of callback invocation (every N iterations in Stage 2)
    callback_frequency: usize,

    /// Flag to track if Stage 1 has completed (prevents repeated logging)
    stage1_completed: bool,

    /// Flag to track if Stage 2 has started (prevents premature logging)
    stage2_started: bool,

    /// Previous Stage 2 state for change detection in callbacks (using RwLock for thread-safe interior mutability)
    previous_stage2_state: Option<std::sync::RwLock<PreviousStage2State>>,

    /// Whether to filter Stage 2 callback messages to only show changes
    filter_stage2_changes: bool,
}

// Manual Debug implementation since ObserverCallback doesn't implement Debug
impl std::fmt::Debug for Observer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observer")
            .field("mode", &self.mode)
            .field("stage1", &self.stage1)
            .field("stage2", &self.stage2)
            .field("track_timing", &self.track_timing)
            .field("start_time", &self.start_time)
            .field("callback", &self.callback.as_ref().map(|_| "Some(...)"))
            .field("callback_frequency", &self.callback_frequency)
            .field("stage1_completed", &self.stage1_completed)
            .field("stage2_started", &self.stage2_started)
            .finish()
    }
}

// Manual Clone implementation since ObserverCallback is now Arc (clonable)
impl Clone for Observer {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            stage1: self.stage1.clone(),
            stage2: self.stage2.clone(),
            track_timing: self.track_timing,
            start_time: self.start_time,
            callback: self.callback.clone(),
            callback_frequency: self.callback_frequency,
            stage1_completed: self.stage1_completed,
            stage2_started: self.stage2_started,
            previous_stage2_state: self
                .previous_stage2_state
                .as_ref()
                .map(|cell| std::sync::RwLock::new(cell.read().unwrap().clone())),
            filter_stage2_changes: self.filter_stage2_changes,
        }
    }
}

impl Observer {
    /// Create a new observer with no tracking enabled
    ///
    /// Returns a minimal observer that tracks nothing by default.
    /// Use the builder methods to enable specific tracking features.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new();
    /// // No tracking enabled - use builder methods to configure
    /// ```
    pub fn new() -> Self {
        Self {
            mode: ObserverMode::Both,
            stage1: None,
            stage2: None,
            track_timing: false,
            start_time: None,
            callback: None,
            callback_frequency: 1,
            stage1_completed: false,
            stage2_started: false,
            previous_stage2_state: None,
            filter_stage2_changes: false,
        }
    }

    /// Enable Stage 1 tracking
    ///
    /// Enables tracking of scatter search metrics including:
    /// - Reference set size and composition
    /// - Best objective values found
    /// - Function evaluation counts
    /// - Trial point generation statistics
    /// - Sub-stage progression (initialization, diversification, intensification)
    ///
    /// Stage 1 tracking is required for `stage1()` and `stage1_final()` to return data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking();
    /// ```
    pub fn with_stage1_tracking(mut self) -> Self {
        self.stage1 = Some(Stage1State::new());
        self
    }

    /// Enable Stage 2 tracking
    ///
    /// Enables tracking of iterative refinement metrics including:
    /// - Current iteration number
    /// - Solution set size and composition
    /// - Best objective values
    /// - Local solver call statistics
    /// - Function evaluation counts
    /// - Threshold values and merit filtering
    /// - Convergence metrics (unchanged cycles)
    ///
    /// Stage 2 tracking is required for `stage2()` to return data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking();
    /// ```
    pub fn with_stage2_tracking(mut self) -> Self {
        self.stage2 = Some(Stage2State::new());
        self
    }

    /// Enable timing tracking for stages
    ///
    /// When enabled, tracks elapsed time for:
    /// - Total Stage 1 duration
    /// - Total Stage 2 duration
    /// - Sub-stage timing within Stage 1
    ///
    /// Timing data is accessible via the `total_time()` methods on
    /// [`Stage1State`] and [`Stage2State`].
    ///
    /// # Performance Impact
    ///
    /// Timing has minimal performance impact but requires system clock access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_timing();
    ///
    /// // Later, access timing data
    /// if let Some(stage1) = observer.stage1_final() {
    ///     if let Some(time) = stage1.total_time() {
    ///         println!("Stage 1 took {:.3} seconds", time);
    ///     }
    /// }
    /// ```
    pub fn with_timing(mut self) -> Self {
        self.track_timing = true;
        self
    }

    /// Set observer mode
    ///
    /// Controls which stages of the optimization algorithm are monitored.
    /// This allows fine-grained control over tracking scope and performance.
    ///
    /// # Arguments
    ///
    /// * `mode` - The observer mode determining which stages to track
    ///
    /// # Performance Considerations
    ///
    /// Using [`ObserverMode::Stage1Only`] or [`ObserverMode::Stage2Only`] can
    /// reduce memory usage and callback overhead when only specific stage
    /// information is needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::{Observer, ObserverMode};
    ///
    /// // Track only Stage 2 for performance monitoring
    /// let observer = Observer::new()
    ///     .with_mode(ObserverMode::Stage2Only)
    ///     .with_stage2_tracking()
    ///     .with_default_callback();
    ///
    /// // Track both stages (default behavior)
    /// let observer = Observer::new()
    ///     .with_mode(ObserverMode::Both)
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking();
    /// ```
    pub fn with_mode(mut self, mode: ObserverMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set a callback function to be called during optimization
    ///
    /// The callback receives a reference to the Observer, allowing access to
    /// all tracked metrics in real-time during optimization. Callbacks are invoked
    /// at key points during the algorithm execution.
    ///
    /// # Callback Timing
    ///
    /// - **Stage 1**: Called after major substages (initialization, diversification,
    ///   intensification, scatter search completion, local optimization completion)
    /// - **Stage 2**: Called according to the callback frequency (default: every iteration)
    ///
    /// # Arguments
    ///
    /// * `callback` - Function to call during optimization
    ///
    /// # Thread Safety
    ///
    /// Callbacks must be thread-safe (`Send + Sync`) as they may be called from
    /// parallel execution contexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         if let Some(stage2) = obs.stage2() {
    ///             println!("Iteration {}: Best = {:.6}",
    ///                 stage2.current_iteration(),
    ///                 stage2.best_objective());
    ///         }
    ///     });
    /// ```
    ///
    /// # Advanced Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         // Log Stage 1 progress
    ///         if let Some(stage1) = obs.stage1() {
    ///             println!("Stage 1: {} evaluations, best = {:.6}",
    ///                 stage1.function_evaluations(),
    ///                 stage1.best_objective());
    ///         }
    ///
    ///         // Log Stage 2 progress
    ///         if let Some(stage2) = obs.stage2() {
    ///             println!("Stage 2: Iteration {}, {} solutions",
    ///                 stage2.current_iteration(),
    ///                 stage2.solution_set_size());
    ///         }
    ///     });
    /// ```
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&mut Observer) + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
        self
    }

    /// Set the frequency for callback invocation
    ///
    /// Controls how often the callback is invoked during Stage 2. For example,
    /// a frequency of 10 means the callback is called every 10 iterations.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Number of iterations between callback calls
    ///
    /// # Default Behavior
    ///
    /// - Default frequency is 1 (callback called every iteration)
    /// - If no callback has been set with `with_callback()`, this method will
    ///   automatically use the default callback
    ///
    /// # Performance Considerations
    ///
    /// Lower frequencies reduce callback overhead but provide less detailed monitoring.
    /// Higher frequencies provide more detailed progress information but may impact
    /// performance for very fast optimization problems.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// // This will automatically use the default callback
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback_frequency(10); // Logs every 10 iterations with default callback
    ///
    /// // Custom callback with custom frequency
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_callback(|obs| {
    ///         // Custom logging logic
    ///     })
    ///     .with_callback_frequency(25); // Custom callback every 25 iterations
    /// ```
    pub fn with_callback_frequency(mut self, frequency: usize) -> Self {
        self.callback_frequency = frequency;
        // If no callback has been set, use the default one
        if self.callback.is_none() {
            self = self.with_default_callback();
        }
        self
    }

    /// Enable filtering of Stage 2 callback messages to only show unique updates
    ///
    /// When enabled, Stage 2 callback messages will only be printed when
    /// there is an actual change in the optimization state (other than just
    /// the iteration number). This reduces log verbosity by filtering out
    /// identical consecutive messages.
    ///
    /// # Changes that trigger printing:
    /// - Best objective value changes
    /// - Solution set size changes
    /// - Threshold value changes
    /// - Local solver call counts change
    /// - Function evaluation counts change
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_default_callback()
    ///     .unique_updates(); // Only print when state changes
    /// ```
    pub fn unique_updates(mut self) -> Self {
        self.filter_stage2_changes = true;
        self
    }

    /// Use a default console logging callback for Stage 1 and Stage 2
    ///
    /// This is a convenience method that provides sensible default logging
    /// for both stages of the optimization. The default callback prints progress
    /// information to stderr (using `eprintln!`).
    ///
    /// # Stage 1 Logging
    ///
    /// Logs major substages:
    /// - Scatter search start
    /// - Initialization completion
    /// - Diversification completion
    /// - Intensification completion
    /// - Scatter search completion
    /// - Local optimization completion
    ///
    /// # Stage 2 Logging
    ///
    /// Logs iteration progress according to callback frequency:
    /// - Current iteration number
    /// - Best objective value found
    /// - Current solution set size
    /// - Merit filter threshold value
    /// - Local solver call counts
    /// - Function evaluation counts
    ///
    /// # Output Format
    ///
    /// The default callback prints progress information to stderr:
    ///
    /// ```text
    /// [Stage 1] Scatter Search Complete | Best: 1.234567
    /// [Stage 2] Iter 50 | Best: 0.123456 | Solutions: 8 | Threshold: 0.500000 | Local Calls: 25 | Fn Evals: 1250
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_default_callback();
    /// ```
    ///
    /// # Controlling Frequency
    ///
    /// Use `with_callback_frequency()` to control how often Stage 2 updates are printed:
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage2_tracking()
    ///     .with_default_callback()
    ///     .with_callback_frequency(10); // Print every 10 iterations
    /// ```
    pub fn with_default_callback(self) -> Self {
        // Helper function to format array coordinates cleanly
        fn format_coords(arr: &ndarray::Array1<f64>) -> String {
            let values: Vec<String> = arr.iter().map(|v| format!("{:.6}", v)).collect();
            format!("[{}]", values.join(", "))
        }

        self.with_callback(|obs| {
            // Stage 1 updates
            if let Some(stage1) = obs.stage1() {
                let substage = stage1.current_substage();
                let message = if substage == "scatter_search_running" {
                    "[Stage 1] Starting Scatter Search...".to_string()
                } else if substage == "initialization_complete" {
                    format!(
                        "[Stage 1] Initialization Complete | Initial Points: {}",
                        stage1.function_evaluations()
                    )
                } else if substage == "diversification_complete" {
                    format!(
                        "[Stage 1] Diversification Complete | Ref. Set Size: {}",
                        stage1.reference_set_size()
                    )
                } else if substage == "intensification_complete" {
                    format!(
                        "[Stage 1] Intensification Complete | Trial Points Generated: {} | Accepted: {}",
                        stage1.trial_points_generated(),
                        stage1.reference_set_size()
                    )
                } else if substage == "scatter_search_complete" {
                    if let Some(point) = stage1.best_point() {
                        format!(
                            "[Stage 1] Scatter Search Complete | Best: {:.6} at {}",
                            stage1.best_objective(),
                            format_coords(point)
                        )
                    } else {
                        format!(
                            "[Stage 1] Scatter Search Complete | Best: {:.6}",
                            stage1.best_objective()
                        )
                    }
                } else if substage == "local_optimization_complete" {
                    if let Some(point) = stage1.best_point() {
                        format!(
                            "[Stage 1] Local Optimization Complete | Best: {:.6} at {} | Total Fn Evals: {}",
                            stage1.best_objective(),
                            format_coords(point),
                            stage1.function_evaluations()
                        )
                    } else {
                        format!(
                            "[Stage 1] Local Optimization Complete | Best: {:.6} | Total Fn Evals: {}",
                            stage1.best_objective(),
                            stage1.function_evaluations()
                        )
                    }
                } else {
                    return; // No message for other substages
                };

                // Print directly for real-time output in both sequential and parallel modes
                eprintln!("{}", message);
            }
            // Stage 2 updates (only when started)
            if let Some(stage2) = obs.stage2() {
                if stage2.current_iteration() > 0 {
                    // Extract all stage2 data first to avoid borrowing conflicts
                    let current_iter = stage2.current_iteration();
                    let best_obj = stage2.best_objective();
                    let last_added_coords = stage2.last_added_point().map(format_coords);
                    let sol_size = stage2.solution_set_size();
                    let threshold = stage2.threshold_value();
                    let local_calls = stage2.local_solver_calls();
                    let fn_evals = stage2.function_evaluations();

                    // Check if we should print this iteration
                    let should_print = if obs.filter_stage2_changes {
                        // Use RwLock for thread-safe interior mutability to avoid borrowing conflicts
                        let prev_state = obs.previous_stage2_state.as_ref().map(|cell| cell.read().unwrap().clone());

                        // Check if state changed
                        let has_changed = prev_state.as_ref().is_none_or(|prev| prev.has_changed(stage2));

                        // Update the previous state for next comparison
                        let current_state = PreviousStage2State::from_stage2(stage2);
                        obs.previous_stage2_state = Some(std::sync::RwLock::new(current_state));

                        has_changed
                    } else {
                        true // Always print if filtering is disabled
                    };

                    if should_print {
                        let message = if let Some(coords) = last_added_coords {
                            format!(
                                "[Stage 2] Iter {} | Best: {:.6} at {} | Solutions: {} | Threshold: {:.6} | Local Calls: {} | Fn Evals: {}",
                                current_iter, best_obj, coords, sol_size, threshold, local_calls, fn_evals
                            )
                        } else {
                            format!(
                                "[Stage 2] Iter {} | Best: {:.6} | Solutions: {} | Threshold: {:.6} | Local Calls: {} | Fn Evals: {}",
                                current_iter, best_obj, sol_size, threshold, local_calls, fn_evals
                            )
                        };

                        // Print directly for real-time output in both sequential and parallel modes
                        eprintln!("{}", message);
                    }
                }
            }
        })
    }

    /// Use a default console logging callback for Stage 1 only
    ///
    /// This prints updates during scatter search and local optimization in Stage 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage1_tracking()
    ///     .with_stage1_callback();
    /// ```
    pub fn with_stage1_callback(self) -> Self {
        // Helper function to format array coordinates cleanly
        fn format_coords(arr: &ndarray::Array1<f64>) -> String {
            let values: Vec<String> = arr.iter().map(|v| format!("{:.6}", v)).collect();
            format!("[{}]", values.join(", "))
        }

        self.with_callback(|obs| {
            if let Some(stage1) = obs.stage1() {
                let substage = stage1.current_substage();
                if substage == "scatter_search_running" {
                    eprintln!("[Stage 1] Starting Scatter Search...");
                } else if substage == "initialization_complete" {
                    eprintln!(
                        "[Stage 1] Initialization Complete | Initial Points: {}",
                        stage1.function_evaluations()
                    );
                } else if substage == "diversification_complete" {
                    eprintln!(
                        "[Stage 1] Diversification Complete | Ref. Set Size: {}",
                        stage1.reference_set_size()
                    );
                } else if substage == "intensification_complete" {
                    eprintln!(
                        "[Stage 1] Intensification Complete | Trial Points Generated: {} | Accepted: {}",
                        stage1.trial_points_generated(),
                        stage1.reference_set_size()
                    );
                } else if substage == "scatter_search_complete" {
                    if let Some(point) = stage1.best_point() {
                        eprintln!(
                            "[Stage 1] Scatter Search Complete | Best: {:.6} at {}",
                            stage1.best_objective(),
                            format_coords(point)
                        );
                    } else {
                        eprintln!(
                            "[Stage 1] Scatter Search Complete | Best: {:.6}",
                            stage1.best_objective()
                        );
                    }
                } else if substage == "local_optimization_complete" {
                    if let Some(point) = stage1.best_point() {
                        eprintln!(
                            "[Stage 1] Local Optimization Complete | Best: {:.6} at {} | TotalFnEvals: {}",
                            stage1.best_objective(),
                            format_coords(point),
                            stage1.function_evaluations()
                        );
                    } else {
                        eprintln!(
                            "[Stage 1] Local Optimization Complete | Best: {:.6} | TotalFnEvals: {}",
                            stage1.best_objective(),
                            stage1.function_evaluations()
                        );
                    }
                }
                // Don't print for "stage1_complete" - it's just an internal marker
            }
        })
    }

    /// Use a default console logging callback for Stage 2 only
    ///
    /// This prints iteration progress during Stage 2. Use `with_callback_frequency()`
    /// to control how often updates are printed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new()
    ///     .with_stage2_tracking()
    ///     .with_stage2_callback()
    ///     .with_callback_frequency(10); // Print every 10 iterations
    /// ```
    pub fn with_stage2_callback(self) -> Self {
        // Helper function to format array coordinates cleanly
        fn format_coords(arr: &ndarray::Array1<f64>) -> String {
            let values: Vec<String> = arr.iter().map(|v| format!("{:.6}", v)).collect();
            format!("[{}]", values.join(", "))
        }

        self.with_callback(|obs| {
            if let Some(stage2) = obs.stage2() {
                if stage2.current_iteration() > 0 {
                    if let Some(point) = stage2.last_added_point() {
                        eprintln!(
                            "[Stage 2] Iter {} | Best: {:.6} at {} | Solutions: {} | Threshold: {:.6} | Local Calls: {} | Fn Evals: {}",
                            stage2.current_iteration(),
                            stage2.best_objective(),
                            format_coords(point),
                            stage2.solution_set_size(),
                            stage2.threshold_value(),
                            stage2.local_solver_calls(),
                            stage2.function_evaluations()
                        );
                    } else {
                        eprintln!(
                            "[Stage 2] Iter {} | Best: {:.6} | Solutions: {} | Threshold: {:.6} | Local Calls: {} | Fn Evals: {}",
                            stage2.current_iteration(),
                            stage2.best_objective(),
                            stage2.solution_set_size(),
                            stage2.threshold_value(),
                            stage2.local_solver_calls(),
                            stage2.function_evaluations()
                        );
                    }
                }
            }
        })
    }

    /// Start timing
    ///
    /// Records the current time as the start time for the optimization.
    /// This is called internally when optimization begins.
    pub(crate) fn start_timer(&mut self) {
        if self.track_timing {
            self.start_time = Some(Instant::now());
        }
    }

    /// Get elapsed time in seconds
    ///
    /// Returns the time elapsed since `start_timer()` was called.
    /// Returns `None` if timing is not enabled or timer hasn't started.
    pub fn elapsed_time(&self) -> Option<f64> {
        self.start_time.map(|start| start.elapsed().as_secs_f64())
    }

    /// Check if Stage 1 should be observed
    ///
    /// Returns true if Stage 1 tracking is enabled and the observer mode
    /// allows Stage 1 observation (Stage1Only or Both modes).
    pub fn should_observe_stage1(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage1Only | ObserverMode::Both) && self.stage1.is_some()
    }

    /// Check if Stage 2 should be observed
    ///
    /// Returns true if Stage 2 tracking is enabled and the observer mode
    /// allows Stage 2 observation (Stage2Only or Both modes).
    pub fn should_observe_stage2(&self) -> bool {
        matches!(self.mode, ObserverMode::Stage2Only | ObserverMode::Both) && self.stage2.is_some()
    }

    /// Get Stage 1 state reference
    ///
    /// Returns the current Stage 1 state if Stage 1 tracking is enabled and
    /// Stage 1 is still active. Returns `None` after Stage 1 completes to
    /// prevent repeated callback invocations.
    ///
    /// For final Stage 1 statistics after completion, use `stage1_final()`.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage1State)` if Stage 1 is active and tracking is enabled
    /// - `None` if Stage 1 has completed or tracking is disabled
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new().with_stage1_tracking();
    ///
    /// // In a callback during Stage 1
    /// if let Some(stage1) = observer.stage1() {
    ///     println!("Current best: {}", stage1.best_objective());
    ///     println!("Reference set size: {}", stage1.reference_set_size());
    /// }
    /// ```
    pub fn stage1(&self) -> Option<&Stage1State> {
        // Don't return Stage 1 state after it's completed to prevent repeated logging
        if self.stage1_completed { None } else { self.stage1.as_ref() }
    }

    /// Get Stage 1 state reference even after completion (for final statistics)
    ///
    /// Returns the final Stage 1 state regardless of whether Stage 1 is still
    /// active. This method should be used for accessing final statistics after
    /// optimization completes.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage1State)` if Stage 1 tracking was enabled
    /// - `None` if Stage 1 tracking was not enabled
    ///
    /// # Difference from `stage1()`
    ///
    /// - `stage1()` returns `None` after Stage 1 completes (to prevent repeated callbacks)
    /// - `stage1_final()` always returns the final state when available
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// // After optimization completes
    /// let observer = Observer::new().with_stage1_tracking();
    /// // ... run optimization ...
    /// if let Some(stage1) = observer.stage1_final() {
    ///     println!("Stage 1 Summary:");
    ///     println!("  Total function evaluations: {}", stage1.function_evaluations());
    ///     println!("  Trial points generated: {}", stage1.trial_points_generated());
    ///     println!("  Final reference set size: {}", stage1.reference_set_size());
    ///     if let Some(time) = stage1.total_time() {
    ///         println!("  Total time: {:.3}s", time);
    ///     }
    /// }
    /// ```
    pub fn stage1_final(&self) -> Option<&Stage1State> {
        self.stage1.as_ref()
    }

    /// Get mutable Stage 1 state reference
    ///
    /// Used internally by the OQNLP algorithm to update Stage 1 metrics.
    /// Returns None if Stage 1 tracking is not enabled.
    pub(crate) fn stage1_mut(&mut self) -> Option<&mut Stage1State> {
        self.stage1.as_mut()
    }

    /// Mark Stage 1 as completed (prevents further Stage 1 callback invocations)
    ///
    /// Called internally when Stage 1 finishes. This prevents the observer
    /// from returning Stage 1 state in subsequent `stage1()` calls,
    /// avoiding repeated callback invocations for completed stages.
    pub(crate) fn mark_stage1_complete(&mut self) {
        self.stage1_completed = true;
    }

    /// Get Stage 2 state reference
    ///
    /// Returns the current Stage 2 state if Stage 2 tracking is enabled and
    /// Stage 2 has started. Returns `None` before Stage 2 begins to prevent
    /// premature callback invocations.
    ///
    /// # Returns
    ///
    /// - `Some(&Stage2State)` if Stage 2 is active and tracking is enabled
    /// - `None` if Stage 2 hasn't started yet or tracking is disabled
    ///
    /// # Example
    ///
    /// ```rust
    /// use globalsearch::observers::Observer;
    ///
    /// let observer = Observer::new().with_stage2_tracking();
    ///
    /// // In a callback during Stage 2
    /// if let Some(stage2) = observer.stage2() {
    ///     println!("Iteration: {}", stage2.current_iteration());
    ///     println!("Best objective: {}", stage2.best_objective());
    ///     println!("Solution set size: {}", stage2.solution_set_size());
    /// }
    /// ```
    pub fn stage2(&self) -> Option<&Stage2State> {
        // Don't return Stage 2 state until it has started to prevent premature logging
        if self.stage2_started { self.stage2.as_ref() } else { None }
    }

    /// Get mutable Stage 2 state reference
    ///
    /// Used internally by the OQNLP algorithm to update Stage 2 metrics.
    /// Returns None if Stage 2 tracking is not enabled.
    pub(crate) fn stage2_mut(&mut self) -> Option<&mut Stage2State> {
        self.stage2.as_mut()
    }

    /// Mark Stage 2 as started (allows Stage 2 callback invocations)
    ///
    /// Called internally when Stage 2 begins. This allows the observer
    /// to return Stage 2 state in subsequent `stage2()` calls,
    /// enabling callback invocations for active Stage 2 operation.
    pub(crate) fn mark_stage2_started(&mut self) {
        self.stage2_started = true;
    }

    /// Check if timing is enabled
    ///
    /// Returns true if the observer is configured to track timing information.
    pub fn is_timing_enabled(&self) -> bool {
        self.track_timing
    }

    /// Invoke the callback if one is set
    ///
    /// Called internally by the OQNLP algorithm at appropriate points during
    /// optimization. The callback receives a reference to this observer,
    /// allowing access to all current metrics.
    pub(crate) fn invoke_callback(&mut self) {
        if let Some(callback) = &self.callback {
            let callback = Arc::clone(callback);
            callback(self);
        }
    }

    /// Check if callback should be invoked for the current iteration
    ///
    /// Determines whether the callback should be called based on the current
    /// iteration number and the configured callback frequency.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current iteration number in Stage 2
    ///
    /// # Returns
    ///
    /// True if a callback is configured and the iteration is a multiple of
    /// the callback frequency.
    pub(crate) fn should_invoke_callback(&self, iteration: usize) -> bool {
        self.callback.is_some() && (iteration % self.callback_frequency == 0)
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests_observers {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    /// Test Observer creation and default configuration
    fn test_observer_creation() {
        let observer = Observer::new();
        assert!(!observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());
        assert!(!observer.is_timing_enabled());
    }

    #[test]
    /// Test Observer with Stage 1 tracking enabled
    fn test_observer_with_stage1() {
        let observer = Observer::new().with_stage1_tracking();
        assert!(observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());
    }

    #[test]
    /// Test Observer with Stage 2 tracking enabled
    fn test_observer_with_stage2() {
        let observer = Observer::new().with_stage2_tracking();
        assert!(!observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    /// Test Observer with both Stage 1 and Stage 2 tracking enabled
    fn test_observer_with_both_stages() {
        let observer = Observer::new().with_stage1_tracking().with_stage2_tracking();
        assert!(observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    /// Test Observer with timing functionality enabled
    fn test_observer_with_timing() {
        let observer = Observer::new().with_timing();
        assert!(observer.is_timing_enabled());
    }

    #[test]
    /// Test Observer mode restrictions and behavior
    fn test_observer_modes() {
        let observer = Observer::new()
            .with_mode(ObserverMode::Stage1Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());

        let observer = Observer::new()
            .with_mode(ObserverMode::Stage2Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(!observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());

        let observer = Observer::new()
            .with_mode(ObserverMode::Both)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    /// Test Observer Stage 1 state access and lifecycle
    fn test_observer_stage1_state_access() {
        let mut observer = Observer::new().with_stage1_tracking();

        // Initially should return Some
        assert!(observer.stage1().is_some());

        // After marking complete, should return None
        observer.mark_stage1_complete();
        assert!(observer.stage1().is_none());

        // But stage1_final should still return Some
        assert!(observer.stage1_final().is_some());
    }

    #[test]
    /// Test Observer Stage 2 state access and lifecycle
    fn test_observer_stage2_state_access() {
        let mut observer = Observer::new().with_stage2_tracking();

        // Initially should return None (not started)
        assert!(observer.stage2().is_none());

        // After marking started, should return Some
        observer.mark_stage2_started();
        assert!(observer.stage2().is_some());
    }

    #[test]
    /// Test Observer timing functionality and elapsed time tracking
    fn test_observer_timing() {
        let mut observer = Observer::new().with_timing();

        // No elapsed time initially
        assert!(observer.elapsed_time().is_none());

        observer.start_timer();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let elapsed = observer.elapsed_time();
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() > 0.0);
    }

    #[test]
    /// Test Observer callback functionality and invocation
    fn test_observer_callbacks() {
        let callback_count = Arc::new(Mutex::new(0));
        let callback_count_clone = Arc::clone(&callback_count);

        let mut observer = Observer::new().with_callback(move |_| {
            let mut count = callback_count_clone.lock().unwrap();
            *count += 1;
        });

        // Invoke callback
        observer.invoke_callback();

        assert_eq!(*callback_count.lock().unwrap(), 1);
    }

    #[test]
    /// Test Observer callback frequency and invocation timing
    fn test_observer_callback_frequency() {
        let callback_count = Arc::new(Mutex::new(0));
        let callback_count_clone = Arc::clone(&callback_count);

        let observer = Observer::new().with_callback_frequency(3).with_callback(move |_| {
            let mut count = callback_count_clone.lock().unwrap();
            *count += 1;
        });

        // Should invoke at iterations 3, 6, 9
        assert!(!observer.should_invoke_callback(1));
        assert!(!observer.should_invoke_callback(2));
        assert!(observer.should_invoke_callback(3));
        assert!(!observer.should_invoke_callback(4));
        assert!(!observer.should_invoke_callback(5));
        assert!(observer.should_invoke_callback(6));
    }

    #[test]
    /// Test Observer default callback implementations
    fn test_observer_default_callbacks() {
        // Test that default callbacks can be created without panicking
        let _observer1 = Observer::new().with_default_callback();
        let _observer2 = Observer::new().with_stage1_callback();
        let _observer3 = Observer::new().with_stage2_callback();
    }

    #[test]
    /// Test Observer clone behavior and configuration preservation
    fn test_observer_clone_behavior() {
        let observer = Observer::new()
            .with_stage1_tracking()
            .with_stage2_tracking()
            .with_timing()
            .with_mode(ObserverMode::Stage1Only);

        let cloned = observer.clone();

        // Should have same configuration
        assert!(cloned.should_observe_stage1());
        assert!(!cloned.should_observe_stage2()); // Due to Stage1Only mode
        assert!(cloned.is_timing_enabled());

        // But callbacks should be None (not cloned)
        // We can't directly test this, but the clone should work
    }

    #[test]
    /// Test Observer default implementation and configuration
    fn test_observer_default_implementation() {
        let observer = Observer::default();
        assert!(!observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());
        assert!(!observer.is_timing_enabled());
    }

    #[test]
    /// Test Observer Stage 1 mutable state access and updates
    fn test_observer_stage1_mut_access() {
        let mut observer = Observer::new().with_stage1_tracking();

        {
            let stage1 = observer.stage1_mut().unwrap();
            stage1.set_reference_set_size(10);
            stage1.set_best_objective(5.0);
        }

        let stage1 = observer.stage1().unwrap();
        assert_eq!(stage1.reference_set_size(), 10);
        assert_eq!(stage1.best_objective(), 5.0);
    }

    #[test]
    /// Test Observer Stage 2 mutable state access and updates
    fn test_observer_stage2_mut_access() {
        let mut observer = Observer::new().with_stage2_tracking();
        observer.mark_stage2_started();

        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(5);
            stage2.set_best_objective(3.0);
        }

        let stage2 = observer.stage2().unwrap();
        assert_eq!(stage2.current_iteration(), 5);
        assert_eq!(stage2.best_objective(), 3.0);
    }

    #[test]
    /// Test Observer mode restrictions and stage tracking behavior
    fn test_observer_mode_restrictions() {
        // Stage1Only mode
        let observer = Observer::new()
            .with_mode(ObserverMode::Stage1Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());

        // Stage2Only mode
        let observer = Observer::new()
            .with_mode(ObserverMode::Stage2Only)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(!observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());

        // Both mode
        let observer = Observer::new()
            .with_mode(ObserverMode::Both)
            .with_stage1_tracking()
            .with_stage2_tracking();

        assert!(observer.should_observe_stage1());
        assert!(observer.should_observe_stage2());
    }

    #[test]
    /// Test Observer callback with frequency configuration
    fn test_observer_callback_with_frequency() {
        let observer = Observer::new().with_callback_frequency(5);

        // Should have default callback when frequency is set without explicit callback
        // This is tested implicitly by the fact that it doesn't panic
        assert!(observer.callback.is_some());
    }

    #[test]
    /// Test Observer stage transitions and state lifecycle
    fn test_observer_stage_transitions() {
        let mut observer = Observer::new().with_stage1_tracking().with_stage2_tracking();

        // Stage 1 should be accessible initially
        assert!(observer.stage1().is_some());
        assert!(observer.stage2().is_none()); // Not started yet

        // Mark Stage 1 complete
        observer.mark_stage1_complete();
        assert!(observer.stage1().is_none()); // No longer accessible
        assert!(observer.stage1_final().is_some()); // But final is still accessible

        // Mark Stage 2 started
        observer.mark_stage2_started();
        assert!(observer.stage2().is_some()); // Now accessible
    }

    #[test]
    /// Test Observer behavior without any stage tracking enabled
    fn test_observer_without_tracking() {
        let observer = Observer::new();

        // Should not observe either stage
        assert!(!observer.should_observe_stage1());
        assert!(!observer.should_observe_stage2());

        // State access should return None
        assert!(observer.stage1().is_none());
        assert!(observer.stage1_final().is_none());
        assert!(observer.stage2().is_none());

        // Mutable access should return None
        let mut observer = observer;
        assert!(observer.stage1_mut().is_none());
        assert!(observer.stage2_mut().is_none());
    }

    #[test]
    /// Test Observer with simple quadratic optimization problem integration
    fn test_observer_with_simple_optimization_problem() {
        use crate::local_solver::builders::COBYLABuilder;
        use crate::oqnlp::OQNLP;
        use crate::problem::Problem;
        use crate::types::{EvaluationError, LocalSolverType, OQNLPParams};
        use ndarray::{Array1, Array2};

        /// Simple quadratic problem: sum x_i^2 for i=1 to n
        /// Global minimum at x = [0, 0, ..., 0] with f(x) = 0
        #[derive(Debug, Clone)]
        struct QuadraticSum {
            dimension: usize,
        }

        impl QuadraticSum {
            fn new(dimension: usize) -> Self {
                Self { dimension }
            }
        }

        impl Problem for QuadraticSum {
            fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
                Ok(x.iter().map(|xi| xi * xi).sum())
            }

            fn variable_bounds(&self) -> Array2<f64> {
                // Create bounds array: each row is [lower, upper] for each dimension
                let mut bounds = Array2::zeros((self.dimension, 2));
                for i in 0..self.dimension {
                    bounds[[i, 0]] = -4.0; // lower bound
                    bounds[[i, 1]] = 4.0; // upper bound
                }
                bounds
            }
        }

        // Create a 2D quadratic problem
        let problem = QuadraticSum::new(2);

        // Optimization parameters
        let params = OQNLPParams {
            iterations: 100,
            wait_cycle: 5,
            threshold_factor: 0.5,
            distance_factor: 0.5,
            population_size: 150,
            local_solver_type: LocalSolverType::COBYLA,
            local_solver_config: COBYLABuilder::default().max_iter(25).build(),
            seed: 0,
        };

        // Create observer with both stages tracking
        let observer = Observer::new().with_stage1_tracking().with_stage2_tracking().with_timing();

        // Run optimization
        let mut oqnlp = OQNLP::new(problem, params).unwrap().add_observer(observer);
        let solution_set = oqnlp.run().unwrap();

        // Get the observer back
        let observer = oqnlp.observer().unwrap();

        // Test Stage 1 metrics
        if let Some(stage1) = observer.stage1_final() {
            assert!(stage1.function_evaluations() > 0);
            assert!(stage1.reference_set_size() > 0);
            assert!(!stage1.best_objective().is_nan());
            assert!(stage1.best_objective() >= 0.0); // Quadratic sum is always >= 0
            assert!(stage1.trial_points_generated() > 0);
            if let Some(time) = stage1.total_time() {
                assert!(time > 0.0);
            }
        }

        // Test Stage 2 metrics
        if let Some(stage2) = observer.stage2() {
            println!("Stage 2 ran with {} function evaluations", stage2.function_evaluations());
            // Just check that Stage 2 has valid data
            assert!(!stage2.best_objective().is_nan());
            assert!(stage2.best_objective() >= 0.0); // Quadratic sum is always >= 0
            assert!(stage2.threshold_value() >= 0.0);
            if let Some(time) = stage2.total_time() {
                assert!(time >= 0.0);
            }
        } else {
            println!("Stage 2 did not run");
        }

        // Test that we found a reasonable solution
        // The global minimum is 0, but we expect to get close
        let best_solution = solution_set.best_solution().unwrap();
        let best_objective = best_solution.objective;
        assert!(best_objective >= 0.0);
        assert!(best_objective < 1e-3);

        println!("Optimization test completed successfully!");
        println!("Best objective found: {:.6}", best_objective);
        println!("Solution: {:?}", best_solution.point);
    }

    #[test]
    /// Test Observer Stage 2 unique updates functionality
    fn test_observer_stage2_unique_updates() {
        use std::sync::{Arc, Mutex};

        let messages = Arc::new(Mutex::new(Vec::new()));
        let messages_clone = Arc::clone(&messages);

        let mut observer = Observer::new()
            .with_stage2_tracking()
            .unique_updates()
            .with_callback_frequency(1) // Invoke callback every iteration
            .with_callback(move |obs| {
                if let Some(stage2) = obs.stage2() {
                    if stage2.current_iteration() > 0 {
                        // Extract all stage2 data first to avoid borrowing conflicts
                        let current_iter = stage2.current_iteration();
                        let best_obj = stage2.best_objective();
                        let sol_size = stage2.solution_set_size();
                        let threshold = stage2.threshold_value();

                        // Check if we should print this iteration (same logic as default callback)
                        let should_print = if obs.filter_stage2_changes {
                            // Use RwLock for thread-safe interior mutability to avoid borrowing conflicts
                            let prev_state = obs.previous_stage2_state.as_ref().map(|cell| cell.read().unwrap().clone());

                            // Check if state changed
                            let has_changed = prev_state.as_ref().map_or(true, |prev| prev.has_changed(stage2));

                            // Update the previous state for next comparison
                            let current_state = PreviousStage2State::from_stage2(stage2);
                            obs.previous_stage2_state = Some(std::sync::RwLock::new(current_state));

                            has_changed
                        } else {
                            true // Always print if filtering is disabled
                        };

                        if should_print {
                            let message = format!(
                                "[Stage 2] Iter {} | Best: {:.6} | Solutions: {} | Threshold: {:.6}",
                                current_iter, best_obj, sol_size, threshold
                            );
                            messages_clone.lock().unwrap().push(message);
                        }
                    }
                }
            });

        observer.mark_stage2_started();

        // Simulate Stage 2 iterations with some changes and some identical states
        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(1);
            stage2.set_best_objective(10.0);
            stage2.set_solution_set_size(5);
            stage2.set_threshold_value(1.0);
        }
        if observer.should_invoke_callback(1) {
            observer.invoke_callback(); // Should print (first iteration)
        }

        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(2);
            // Same values - should not print
        }
        if observer.should_invoke_callback(2) {
            observer.invoke_callback(); // Should NOT print (no change)
        }

        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(3);
            stage2.set_best_objective(8.0); // Changed - should print
        }
        if observer.should_invoke_callback(3) {
            observer.invoke_callback(); // Should print (best objective changed)
        }

        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(4);
            // Same values - should not print
        }
        if observer.should_invoke_callback(4) {
            observer.invoke_callback(); // Should NOT print (no change)
        }

        {
            let stage2 = observer.stage2_mut().unwrap();
            stage2.set_iteration(5);
            stage2.set_solution_set_size(6); // Changed - should print
        }
        if observer.should_invoke_callback(5) {
            observer.invoke_callback(); // Should print (solution set size changed)
        }

        let captured_messages = messages.lock().unwrap();
        println!("Captured {} messages:", captured_messages.len());
        for msg in captured_messages.iter() {
            println!("  {}", msg);
        }

        assert_eq!(captured_messages.len(), 3, "Should have 3 messages (iterations 1, 3, and 5)");

        // Verify the messages contain the expected iteration numbers
        assert!(captured_messages[0].contains("Iter 1"));
        assert!(captured_messages[1].contains("Iter 3"));
        assert!(captured_messages[2].contains("Iter 5"));
    }
}
