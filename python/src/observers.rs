use globalsearch::observers::{Observer, ObserverMode, Stage1State, Stage2State};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

/// Observer mode determines which stages to track
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyObserverMode {
    /// Only track Stage 1 (reference set construction)
    Stage1Only,
    /// Only track Stage 2 (iterative improvement)
    Stage2Only,
    /// Track both stages
    Both,
}

impl From<PyObserverMode> for ObserverMode {
    fn from(mode: PyObserverMode) -> Self {
        match mode {
            PyObserverMode::Stage1Only => ObserverMode::Stage1Only,
            PyObserverMode::Stage2Only => ObserverMode::Stage2Only,
            PyObserverMode::Both => ObserverMode::Both,
        }
    }
}

/// State tracker for Stage 1 of the algorithm
///
/// Tracks comprehensive metrics during the scatter search phase that builds
/// the initial reference set. This includes reference set construction,
/// trial point generation, function evaluations, and substage progression.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyStage1State {
    inner: Stage1State,
}

#[pymethods]
impl PyStage1State {
    /// Get reference set size
    ///
    /// Returns the current number of solutions in the reference set.
    /// The reference set maintains a diverse collection of high-quality solutions
    /// that guide the intensification phase of scatter search.
    #[getter]
    fn reference_set_size(&self) -> usize {
        self.inner.reference_set_size()
    }

    /// Get best objective value
    ///
    /// Returns the best (lowest) objective function value found so far in Stage 1.
    /// This represents the highest quality solution discovered during scatter search.
    #[getter]
    fn best_objective(&self) -> f64 {
        self.inner.best_objective()
    }

    /// Get current substage name
    ///
    /// Returns a string identifier for the current phase of Stage 1 execution.
    /// This helps track progress through the scatter search algorithm.
    #[getter]
    fn current_substage(&self) -> &str {
        self.inner.current_substage()
    }

    /// Get total elapsed time since Stage 1 started (seconds)
    ///
    /// Returns the time elapsed since Stage 1 began. If Stage 1 is still running,
    /// returns the current elapsed time. If Stage 1 has completed, returns the
    /// total time spent in Stage 1.
    #[getter]
    fn total_time(&self) -> Option<f64> {
        self.inner.total_time()
    }

    /// Get total number of function evaluations
    ///
    /// Returns the cumulative count of objective function evaluations performed
    /// during Stage 1. This includes evaluations for initial points, diversification,
    /// intensification trial points, and local optimization.
    #[getter]
    fn function_evaluations(&self) -> usize {
        self.inner.function_evaluations()
    }

    /// Get number of trial points generated
    ///
    /// Returns the total number of trial points generated during the intensification
    /// phase of scatter search. Trial points are candidate solutions created by
    /// combining and perturbing reference set members.
    #[getter]
    fn trial_points_generated(&self) -> usize {
        self.inner.trial_points_generated()
    }

    /// Get best solution coordinates
    ///
    /// Returns the coordinates (decision variables) of the best solution found
    /// so far in Stage 1, or None if no solution has been evaluated yet.
    /// Returns a Python list of floats representing the solution point.
    #[getter]
    fn best_point(&self) -> Option<Vec<f64>> {
        self.inner.best_point().map(|arr| arr.to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStage1State(reference_set_size={}, best_objective={:.6}, current_substage='{}', function_evaluations={}, trial_points_generated={})",
            self.reference_set_size(),
            self.best_objective(),
            self.current_substage(),
            self.function_evaluations(),
            self.trial_points_generated()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Stage 1 State\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nReference Set Size: {}\nBest Objective: {:.6e}\nCurrent Substage: {}\nFunction Evaluations: {}\nTrial Points Generated: {}\nTotal Time: {:.3}s",
            self.reference_set_size(),
            self.best_objective(),
            self.current_substage(),
            self.function_evaluations(),
            self.trial_points_generated(),
            self.total_time().unwrap_or(0.0)
        )
    }
}

impl From<Stage1State> for PyStage1State {
    fn from(state: Stage1State) -> Self {
        PyStage1State { inner: state }
    }
}

/// State tracker for Stage 2 of the algorithm
///
/// Tracks comprehensive metrics during the iterative refinement phase that
/// improves the solution set through merit filtering and local optimization.
/// This phase focuses on intensifying search around high-quality regions.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyStage2State {
    inner: Stage2State,
}

#[pymethods]
impl PyStage2State {
    /// Get best objective value
    ///
    /// Returns the best (lowest) objective function value found across all
    /// solutions in the current solution set. This represents the highest
    /// quality solution discovered during Stage 2.
    #[getter]
    fn best_objective(&self) -> f64 {
        self.inner.best_objective()
    }

    /// Get solution set size
    ///
    /// Returns the current number of solutions maintained in the working solution set.
    /// The solution set maintains a diverse collection of high-quality solutions
    /// that balance quality and coverage of the search space.
    #[getter]
    fn solution_set_size(&self) -> usize {
        self.inner.solution_set_size()
    }

    /// Get current iteration
    ///
    /// Returns the current iteration number in Stage 2. Each iteration represents
    /// a complete cycle of selection, generation, evaluation, and filtering.
    #[getter]
    fn current_iteration(&self) -> usize {
        self.inner.current_iteration()
    }

    /// Get threshold value
    ///
    /// Returns the current merit filter threshold value. Solutions must have
    /// an objective value better than this threshold to be accepted into the
    /// solution set during filtering operations.
    #[getter]
    fn threshold_value(&self) -> f64 {
        self.inner.threshold_value()
    }

    /// Get number of local solver calls
    ///
    /// Returns the total number of times local optimization algorithms have been
    /// invoked during Stage 2. Each call attempts to improve a candidate solution
    /// through gradient-based or derivative-free local search.
    #[getter]
    fn local_solver_calls(&self) -> usize {
        self.inner.local_solver_calls()
    }

    /// Get number of local solver calls that improved the solution set
    ///
    /// Returns the number of local solver calls that successfully improved the
    /// solution set by finding better solutions. This measures the effectiveness
    /// of local optimization in finding improvements.
    #[getter]
    fn improved_local_calls(&self) -> usize {
        self.inner.improved_local_calls()
    }

    /// Get total function evaluations
    ///
    /// Returns the cumulative count of objective function evaluations performed
    /// during Stage 2. This includes evaluations for trial points generated
    /// during each iteration and function evaluations performed by local solvers.
    #[getter]
    fn function_evaluations(&self) -> usize {
        self.inner.function_evaluations()
    }

    /// Get unchanged cycles count
    ///
    /// Returns the number of consecutive iterations where the solution set
    /// has not improved. This is a key convergence indicator used to detect
    /// when the algorithm should terminate due to stagnation.
    #[getter]
    fn unchanged_cycles(&self) -> usize {
        self.inner.unchanged_cycles()
    }

    /// Get total time spent in Stage 2 (seconds)
    ///
    /// Returns the time elapsed since Stage 2 began. If Stage 2 is still running,
    /// returns the current elapsed time. If Stage 2 has completed, returns the
    /// total time spent in Stage 2.
    #[getter]
    fn total_time(&self) -> Option<f64> {
        self.inner.total_time()
    }

    /// Get best solution coordinates
    ///
    /// Returns the coordinates (decision variables) of the best solution found
    /// so far in Stage 2, or None if no solution has been evaluated yet.
    /// Returns a Python list of floats representing the solution point.
    #[getter]
    fn best_point(&self) -> Option<Vec<f64>> {
        self.inner.best_point().map(|arr| arr.to_vec())
    }

    /// Get last added solution coordinates
    ///
    /// Returns the coordinates (decision variables) of the most recently added
    /// solution to the solution set, or None if no solution has been added yet.
    /// This is particularly useful for tracking new discoveries in multimodal
    /// optimization problems. Returns a Python list of floats.
    #[getter]
    fn last_added_point(&self) -> Option<Vec<f64>> {
        self.inner.last_added_point().map(|arr| arr.to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStage2State(best_objective={:.6}, solution_set_size={}, current_iteration={}, threshold_value={:.6}, local_solver_calls={}, improved_local_calls={}, function_evaluations={}, unchanged_cycles={})",
            self.best_objective(),
            self.solution_set_size(),
            self.current_iteration(),
            self.threshold_value(),
            self.local_solver_calls(),
            self.improved_local_calls(),
            self.function_evaluations(),
            self.unchanged_cycles()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Stage 2 State\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nBest Objective: {:.6e}\nSolution Set Size: {}\nCurrent Iteration: {}\nThreshold Value: {:.6e}\nLocal Solver Calls: {}\nImproved Calls: {}\nFunction Evaluations: {}\nUnchanged Cycles: {}\nTotal Time: {:.3}s",
            self.best_objective(),
            self.solution_set_size(),
            self.current_iteration(),
            self.threshold_value(),
            self.local_solver_calls(),
            self.improved_local_calls(),
            self.function_evaluations(),
            self.unchanged_cycles(),
            self.total_time().unwrap_or(0.0)
        )
    }
}

impl From<Stage2State> for PyStage2State {
    fn from(state: Stage2State) -> Self {
        PyStage2State { inner: state }
    }
}

/// Main observer struct that tracks algorithm state
///
/// The observer can be configured to track different metrics during
/// Stage 1 (reference set construction) and Stage 2 (iterative improvement).
/// It supports real-time monitoring through callbacks and provides detailed
/// statistics about algorithm performance and convergence.
#[pyclass]
#[derive(Debug)]
pub struct PyObserver {
    pub inner: RwLock<Observer>,
    pub python_callback: Option<Arc<Py<PyAny>>>,
}

#[pymethods]
impl PyObserver {
    /// Create a new observer with no tracking enabled
    ///
    /// Returns a minimal observer that tracks nothing by default.
    /// Use the builder methods to enable specific tracking features.
    #[new]
    fn new() -> Self {
        PyObserver { inner: RwLock::new(Observer::new()), python_callback: None }
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
    fn with_stage1_tracking(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_stage1_tracking();
        Ok(())
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
    fn with_stage2_tracking(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_stage2_tracking();
        Ok(())
    }

    /// Enable timing tracking for stages
    ///
    /// When enabled, tracks elapsed time for:
    /// - Total Stage 1 duration
    /// - Total Stage 2 duration
    /// - Sub-stage timing within Stage 1
    ///
    /// Timing data is accessible via the `total_time()` methods on
    /// [`PyStage1State`] and [`PyStage2State`].
    fn with_timing(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_timing();
        Ok(())
    }

    /// Set observer mode
    ///
    /// Controls which stages of the optimization algorithm are monitored.
    /// This allows fine-grained control over tracking scope and performance.
    ///
    /// # Arguments
    ///
    /// * `mode` - The observer mode determining which stages to track
    fn with_mode(&mut self, mode: PyObserverMode) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_mode(mode.into());
        Ok(())
    }

    /// Set the frequency for callback invocation
    ///
    /// Controls how often the callback is invoked during Stage 2. For example,
    /// a frequency of 10 means the callback is called every 10 iterations.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Number of iterations between callback calls
    fn with_callback_frequency(&mut self, frequency: usize) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_callback_frequency(frequency);
        Ok(())
    }

    /// Set a Python callback function to be called during optimization
    ///
    /// The callback function will be called with two arguments: (stage1_state, stage2_state)
    /// where each can be None or the corresponding state object.
    ///
    /// # Arguments
    ///
    /// * `callback` - Python callable that takes (stage1_state, stage2_state) as arguments
    ///
    /// # Example
    ///
    /// ```python
    /// def my_callback(stage1, stage2):
    ///     if stage1:
    ///         print(f"Stage 1: {stage1.function_evaluations()} evaluations")
    ///     if stage2:
    ///         print(f"Stage 2: Iteration {stage2.current_iteration()}")
    ///
    /// observer = PyObserver()
    /// observer.with_callback(my_callback)
    /// ```
    fn with_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        self.python_callback = Some(Arc::new(callback));
        Ok(())
    }

    /// Use a default console logging callback for Stage 1 and Stage 2
    ///
    /// This is a convenience method that provides sensible default logging
    /// for both stages of the optimization. The default callback prints progress
    /// information to stderr (using `eprintln!`).
    fn with_default_callback(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_default_callback();
        Ok(())
    }

    /// Use a default console logging callback for Stage 1 only
    ///
    /// This prints updates during scatter search and local optimization in Stage 1.
    fn with_stage1_callback(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_stage1_callback();
        Ok(())
    }

    /// Use a default console logging callback for Stage 2 only
    ///
    /// This prints iteration progress during Stage 2. Use `with_callback_frequency()`
    /// to control how often updates are printed.
    fn with_stage2_callback(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().with_stage2_callback();
        Ok(())
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
    /// ```python
    /// observer = PyObserver()
    /// observer.with_stage2_tracking()
    /// observer.with_default_callback()
    /// observer.unique_updates()  # Only print when state changes
    /// ```
    fn unique_updates(&mut self) -> PyResult<()> {
        let mut inner = self.inner.write().expect("RwLock poisoned");
        *inner = inner.clone().unique_updates();
        Ok(())
    }

    /// Get Stage 1 state reference
    ///
    /// Returns the current Stage 1 state if Stage 1 tracking is enabled and
    /// Stage 1 is still active. Returns `None` after Stage 1 completes to
    /// prevent repeated callback invocations.
    ///
    /// For final Stage 1 statistics after completion, use `stage1_final()`.
    fn stage1(&self) -> Option<PyStage1State> {
        self.inner.read().expect("RwLock poisoned").stage1().map(|s| s.clone().into())
    }

    /// Get Stage 1 state reference even after completion (for final statistics)
    ///
    /// Returns the final Stage 1 state regardless of whether Stage 1 is still
    /// active. This method should be used for accessing final statistics after
    /// optimization completes.
    fn stage1_final(&self) -> Option<PyStage1State> {
        self.inner.read().expect("RwLock poisoned").stage1_final().map(|s| s.clone().into())
    }

    /// Get Stage 2 state reference
    ///
    /// Returns the current Stage 2 state if Stage 2 tracking is enabled and
    /// Stage 2 has started. Returns `None` before Stage 2 begins to prevent
    /// premature callback invocations.
    fn stage2(&self) -> Option<PyStage2State> {
        self.inner.read().expect("RwLock poisoned").stage2().map(|s| s.clone().into())
    }

    /// Check if Stage 1 should be observed
    ///
    /// Returns true if Stage 1 tracking is enabled and the observer mode
    /// allows Stage 1 observation (Stage1Only or Both modes).
    #[getter]
    fn should_observe_stage1(&self) -> bool {
        self.inner.read().expect("RwLock poisoned").should_observe_stage1()
    }

    /// Check if Stage 2 should be observed
    ///
    /// Returns true if Stage 2 tracking is enabled and the observer mode
    /// allows Stage 2 observation (Stage2Only or Both modes).
    #[getter]
    fn should_observe_stage2(&self) -> bool {
        self.inner.read().expect("RwLock poisoned").should_observe_stage2()
    }

    /// Check if timing is enabled
    ///
    /// Returns true if the observer is configured to track timing information.
    #[getter]
    fn is_timing_enabled(&self) -> bool {
        self.inner.read().unwrap().is_timing_enabled()
    }

    /// Get elapsed time in seconds
    ///
    /// Returns the time elapsed since `start_timer()` was called.
    /// Returns `None` if timing is not enabled or timer hasn't started.
    #[getter]
    fn elapsed_time(&self) -> Option<f64> {
        self.inner.read().expect("RwLock poisoned").elapsed_time()
    }

    fn __str__(&self) -> String {
        let mut result =
            String::from("Observer Configuration\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        result.push_str(&format!("Stage 1 Tracking: {}\n", self.should_observe_stage1()));
        result.push_str(&format!("Stage 2 Tracking: {}\n", self.should_observe_stage2()));
        result.push_str(&format!("Timing Enabled: {}\n", self.is_timing_enabled()));

        if let Some(stage1) = self.stage1() {
            result.push_str("\nStage 1 State:\n");
            result.push_str(&format!("  Reference Set Size: {}\n", stage1.reference_set_size()));
            result.push_str(&format!("  Best Objective: {:.6e}\n", stage1.best_objective()));
            result.push_str(&format!("  Current Substage: {}\n", stage1.current_substage()));
            result
                .push_str(&format!("  Function Evaluations: {}\n", stage1.function_evaluations()));
            result.push_str(&format!("  Trial Points: {}\n", stage1.trial_points_generated()));
        }

        if let Some(stage2) = self.stage2() {
            result.push_str("\nStage 2 State:\n");
            result.push_str(&format!("  Current Iteration: {}\n", stage2.current_iteration()));
            result.push_str(&format!("  Best Objective: {:.6e}\n", stage2.best_objective()));
            result.push_str(&format!("  Solution Set Size: {}\n", stage2.solution_set_size()));
            result.push_str(&format!("  Threshold Value: {:.6e}\n", stage2.threshold_value()));
            result.push_str(&format!("  Local Solver Calls: {}\n", stage2.local_solver_calls()));
            result.push_str(&format!("  Improved Calls: {}\n", stage2.improved_local_calls()));
            result
                .push_str(&format!("  Function Evaluations: {}\n", stage2.function_evaluations()));
            result.push_str(&format!("  Unchanged Cycles: {}\n", stage2.unchanged_cycles()));
        }

        result
    }
}

impl From<Observer> for PyObserver {
    fn from(observer: Observer) -> Self {
        PyObserver { inner: RwLock::new(observer), python_callback: None }
    }
}

impl PyObserver {
    pub fn into_inner(self) -> Observer {
        self.inner.into_inner().expect("RwLock poisoned")
    }

    pub fn clone_inner(&self) -> Observer {
        let mut observer = self.inner.read().expect("RwLock poisoned").clone();
        if let Some(ref py_callback) = self.python_callback {
            let py_callback = Arc::clone(py_callback);
            let rust_callback = move |obs: &mut Observer| {
                // Acquire the GIL and call the Python callable
                Python::attach(|py| {
                    let py_callable = py_callback.as_ref();
                    let stage1 = obs.stage1().map(|s| PyStage1State::from(s.clone()));
                    let stage2 = obs.stage2().map(|s| PyStage2State::from(s.clone()));
                    let args = (stage1, stage2);
                    // Ignore Python-side errors to avoid panicking the worker thread
                    let _ = py_callable.call1(py, args);
                });
            };
            observer = observer.with_callback(rust_callback);
        }
        observer
    }
}

/// Initialize the observers module
pub fn init_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObserverMode>()?;
    m.setattr("ObserverMode", m.getattr("PyObserverMode")?)?;

    m.add_class::<PyStage1State>()?;
    m.setattr("Stage1State", m.getattr("PyStage1State")?)?;

    m.add_class::<PyStage2State>()?;
    m.setattr("Stage2State", m.getattr("PyStage2State")?)?;

    m.add_class::<PyObserver>()?;
    m.setattr("Observer", m.getattr("PyObserver")?)?;

    Ok(())
}
