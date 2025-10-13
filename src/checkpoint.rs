//! # Checkpointing Module
//!
//! This module provides robust checkpointing functionality for OQNLP optimizations,
//! enabling users to save and resume long-running optimization processes. This is
//! particularly valuable for expensive function evaluations or time-constrained environments.
//!
//! ## Features
//!
//! ### Automatic State Persistence
//! - **Complete State Capture**: Saves all algorithm state including reference sets,
//!   filter parameters, iteration counters, and random number generator state
//! - **Configurable Frequency**: Control how often checkpoints are saved
//! - **Binary Format**: Efficient serialization using bincode for fast I/O
//!
//! ### Flexible Resume Options
//! - **Exact Resume**: Continue optimization with identical parameters
//! - **Modified Resume**: Restart with updated parameters (e.g., more iterations)
//! - **Auto-Resume**: Automatically detect and load existing checkpoints
//!
//! ### Checkpoint Management
//! - **Multiple Strategies**: Keep all checkpoints or maintain only the latest
//! - **Custom Naming**: User-defined checkpoint file naming schemes
//! - **Directory Organization**: Configurable checkpoint storage locations
//! - **Cleanup Utilities**: Automatic management of old checkpoint files
//!
//! ## Usage Patterns
//!
//! ### Basic Checkpointing
//! ```rust
//! use globalsearch::checkpoint::CheckpointManager;
//! use globalsearch::types::CheckpointConfig;
//! use std::path::PathBuf;
//!
//! let config = CheckpointConfig {
//!     checkpoint_dir: PathBuf::from("./checkpoints"),
//!     checkpoint_name: "optimization".to_string(),
//!     save_frequency: 50,  // Save every 50 iterations
//!     keep_all: false,     // Keep only latest checkpoint
//!     auto_resume: true,   // Auto-resume if checkpoint exists
//! };
//!
//! let manager = CheckpointManager::new(config)?;
//! # Ok::<(), globalsearch::checkpoint::CheckpointError>(())
//! ```
//!
//! ### Long-Running Optimizations
//! Ideal for scenarios where:
//! - Function evaluations are expensive (minutes to hours per evaluation)
//! - Optimization runs for days or weeks
//! - System reliability is a concern
//! - Parameter tuning requires multiple restart attempts
//!
//! ## Error Handling
//!
//! The module provides comprehensive error handling for:
//! - File system I/O failures
//! - Serialization/deserialization errors
//! - Missing or corrupted checkpoint files
//! - Invalid checkpoint data

use crate::types::{CheckpointConfig, OQNLPCheckpoint};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Wrapper for bincode v2 errors that can occur during encoding or decoding
#[derive(Debug, Error)]
pub enum BincodeError {
    /// Encoding error
    #[error("Encode error: {0}")]
    EncodeError(#[from] bincode::error::EncodeError),

    /// Decoding error
    #[error("Decode error: {0}")]
    DecodeError(#[from] bincode::error::DecodeError),
}

#[derive(Debug, Error)]
/// Errors that can occur during checkpointing operations.
///
/// This enum covers all possible failure modes when working with checkpoint
/// files, providing detailed error information for debugging and error handling.
///
/// ## Error Categories
///
/// ### I/O Errors
/// File system operations can fail due to:
/// - Insufficient disk space
/// - Permission issues
/// - Network storage problems
/// - Directory creation failures
///
/// ### Serialization Errors
/// Data encoding/decoding failures from:
/// - Corrupted checkpoint files
/// - Version incompatibilities
/// - Incomplete file writes
/// - Memory allocation issues
///
/// ### Missing Checkpoints
/// Occurs when trying to load non-existent checkpoints:
/// - File was deleted or moved
/// - Incorrect file path specification
/// - First run without existing checkpoints
///
/// ### Invalid Data
/// Checkpoint files that cannot be processed:
/// - Wrong file format
/// - Truncated or corrupted data
/// - Incompatible algorithm versions
pub enum CheckpointError {
    /// IO error when reading/writing checkpoint files
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] BincodeError),

    /// Checkpoint file not found
    #[error("Checkpoint file not found: {0}")]
    CheckpointNotFound(PathBuf),

    /// Invalid checkpoint data
    #[error("Invalid checkpoint data: {0}")]
    InvalidCheckpoint(String),
}

/// Manages checkpoint creation, storage, and retrieval for OQNLP optimizations.
///
/// The `CheckpointManager` handles all aspects of optimization state persistence,
/// from initial configuration to final cleanup. It abstracts away the complexity
/// of file management and serialization.
///
/// ## Core Responsibilities
///
/// - **Configuration Management**: Handle checkpoint directory and naming settings
/// - **State Serialization**: Convert optimization state to/from binary format
/// - **File Operations**: Manage checkpoint file creation, reading, and cleanup
/// - **Error Handling**: Provide detailed error information for troubleshooting
///
/// ## Checkpoint Strategies
///
/// ### Single Checkpoint Mode (`keep_all = false`)
/// - Maintains only the most recent checkpoint
/// - Overwrites previous checkpoint on each save
/// - Minimal disk space usage
/// - Best for routine checkpointing
///
/// ### Archive Mode (`keep_all = true`)
/// - Preserves all checkpoint files with iteration numbers
/// - Enables rollback to any previous state
/// - Higher disk space requirements
/// - Best for experimental optimization
///
/// ## Example Usage
///
/// ```rust
/// use globalsearch::checkpoint::CheckpointManager;
/// use globalsearch::types::{CheckpointConfig, OQNLPCheckpoint};
/// use std::path::PathBuf;
///
/// // Configure checkpoint management
/// let config = CheckpointConfig {
///     checkpoint_dir: PathBuf::from("./my_optimization_checkpoints"),
///     checkpoint_name: "expensive_problem".to_string(),
///     save_frequency: 25,
///     keep_all: true,  // Archive all checkpoints
///     auto_resume: false,
/// };
///
/// let manager = CheckpointManager::new(config)?;
///
/// // Check if previous optimization exists
/// if manager.checkpoint_exists() {
///     println!("Found existing checkpoint, resuming optimization...");
///     let checkpoint = manager.load_latest_checkpoint()?;
///     // Resume optimization from checkpoint
/// } else {
///     println!("Starting fresh optimization...");
///     // Start new optimization
/// }
/// # Ok::<(), globalsearch::checkpoint::CheckpointError>(())
/// ```
pub struct CheckpointManager {
    config: CheckpointConfig,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given configuration
    pub fn new(config: CheckpointConfig) -> Result<Self, CheckpointError> {
        if !config.checkpoint_dir.exists() {
            fs::create_dir_all(&config.checkpoint_dir)?;
        }

        Ok(Self { config })
    }

    /// Save a checkpoint to disk
    pub fn save_checkpoint(
        &self,
        checkpoint: &OQNLPCheckpoint,
        iteration: usize,
    ) -> Result<PathBuf, CheckpointError> {
        let filename = if self.config.keep_all {
            format!("{}_{:06}.bin", self.config.checkpoint_name, iteration)
        } else {
            format!("{}.bin", self.config.checkpoint_name)
        };

        let filepath = self.config.checkpoint_dir.join(filename);
        let encoded = bincode::serde::encode_to_vec(checkpoint, bincode::config::legacy())
            .map_err(BincodeError::EncodeError)?;
        fs::write(&filepath, encoded)?;

        Ok(filepath)
    }

    /// Load the latest checkpoint from disk
    pub fn load_latest_checkpoint(&self) -> Result<OQNLPCheckpoint, CheckpointError> {
        let checkpoint_path = if self.config.keep_all {
            self.find_latest_checkpoint()?
        } else {
            let filename = format!("{}.bin", self.config.checkpoint_name);
            self.config.checkpoint_dir.join(filename)
        };

        self.load_checkpoint_from_path(&checkpoint_path)
    }

    /// Load a specific checkpoint from a file path
    pub fn load_checkpoint_from_path(
        &self,
        path: &Path,
    ) -> Result<OQNLPCheckpoint, CheckpointError> {
        if !path.exists() {
            return Err(CheckpointError::CheckpointNotFound(path.to_path_buf()));
        }

        let encoded = fs::read(path)?;
        let (checkpoint, _): (OQNLPCheckpoint, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::legacy())
                .map_err(BincodeError::DecodeError)?;

        Ok(checkpoint)
    }

    /// Check if a checkpoint exists
    pub fn checkpoint_exists(&self) -> bool {
        if self.config.keep_all {
            self.find_latest_checkpoint().is_ok()
        } else {
            let filename = format!("{}.bin", self.config.checkpoint_name);
            self.config.checkpoint_dir.join(filename).exists()
        }
    }

    /// Find the latest checkpoint file when keep_all is enabled
    fn find_latest_checkpoint(&self) -> Result<PathBuf, CheckpointError> {
        let entries = fs::read_dir(&self.config.checkpoint_dir)?;
        let pattern = format!("{}_", self.config.checkpoint_name);

        let mut latest_iteration = 0;
        let mut latest_path = None;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&pattern) && filename.ends_with(".bin") {
                    let iteration_str = &filename[pattern.len()..filename.len() - 4];
                    if let Ok(iteration) = iteration_str.parse::<usize>() {
                        if iteration > latest_iteration {
                            latest_iteration = iteration;
                            latest_path = Some(path);
                        }
                    }
                }
            }
        }

        latest_path
            .ok_or_else(|| CheckpointError::CheckpointNotFound(self.config.checkpoint_dir.clone()))
    }

    /// Clean up old checkpoint files (keep only the latest N files)
    pub fn cleanup_old_checkpoints(&self, keep_count: usize) -> Result<(), CheckpointError> {
        if !self.config.keep_all || keep_count == 0 {
            return Ok(());
        }

        let entries = fs::read_dir(&self.config.checkpoint_dir)?;
        let pattern = format!("{}_", self.config.checkpoint_name);

        let mut checkpoints = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&pattern) && filename.ends_with(".bin") {
                    let iteration_str = &filename[pattern.len()..filename.len() - 4];
                    if let Ok(iteration) = iteration_str.parse::<usize>() {
                        checkpoints.push((iteration, path));
                    }
                }
            }
        }

        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in checkpoints.iter().skip(keep_count) {
            fs::remove_file(path)?;
        }

        Ok(())
    }

    /// Get the checkpoint configuration
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(CheckpointConfig::default()).unwrap()
    }
}

/// Read a checkpoint file directly from a given path
///
/// This is a convenience function that allows reading checkpoint files
/// without creating a `CheckpointManager` instance.
/// # Example
///
/// ```rust,no_run
/// use globalsearch::checkpoint::read_checkpoint_file;
/// use std::path::Path;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let checkpoint = read_checkpoint_file(Path::new("./checkpoints/optimization.bin"))?;
/// println!("Loaded checkpoint:");
/// println!("{}", checkpoint);
/// # Ok(())
/// # }
/// ```
pub fn read_checkpoint_file(path: &Path) -> Result<OQNLPCheckpoint, CheckpointError> {
    if !path.exists() {
        return Err(CheckpointError::CheckpointNotFound(path.to_path_buf()));
    }

    let encoded = fs::read(path)?;
    let (checkpoint, _): (OQNLPCheckpoint, usize) =
        bincode::serde::decode_from_slice(&encoded, bincode::config::legacy())
            .map_err(BincodeError::DecodeError)?;

    Ok(checkpoint)
}

#[cfg(test)]
mod tests_checkpointing {
    use crate::checkpoint::{
        CheckpointConfig, CheckpointError, CheckpointManager, OQNLPCheckpoint,
    };
    use crate::types::{LocalSolution, OQNLPParams, SolutionSet};
    use ndarray::{array, Array1};
    use std::env;
    use std::fs;

    // Helper function to count checkpoint files matching the pattern
    fn count_checkpoint_files(dir: &std::path::Path, prefix: &str) -> usize {
        let entries = fs::read_dir(dir).unwrap();
        entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                if let Some(filename) = e.file_name().to_str() {
                    let pattern = format!("{}_", prefix);
                    if filename.starts_with(&pattern) && filename.ends_with(".bin") {
                        let iteration_str = &filename[pattern.len()..filename.len() - 4];
                        iteration_str.parse::<usize>().is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .count()
    }

    #[test]
    fn test_checkpoint_manager_creation() {
        let temp_dir = env::temp_dir().join("test_checkpoints");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let _manager = CheckpointManager::new(config).unwrap();
        assert!(temp_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let temp_dir = env::temp_dir().join("test_checkpoints_2");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 42,
            merit_threshold: 1.5,
            solution_set: Some(SolutionSet {
                solutions: Array1::from(vec![LocalSolution {
                    point: array![1.0, 2.0],
                    objective: -1.0,
                }]),
            }),
            reference_set: vec![array![1.0, 2.0], array![3.0, 4.0]],
            unchanged_cycles: 5,
            elapsed_time: 120.5,
            distance_filter_solutions: vec![],
            current_seed: 10,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save checkpoint
        let saved_path = manager.save_checkpoint(&checkpoint, 42).unwrap();
        assert!(saved_path.exists());

        // Load checkpoint
        let loaded_checkpoint = manager.load_latest_checkpoint().unwrap();
        assert_eq!(loaded_checkpoint.current_iteration, 42);
        assert_eq!(loaded_checkpoint.merit_threshold, 1.5);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_checkpoint_exists() {
        let temp_dir = env::temp_dir().join("test_checkpoints_3");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();
        assert!(!manager.checkpoint_exists());

        // Create a dummy checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        manager.save_checkpoint(&checkpoint, 0).unwrap();
        assert!(manager.checkpoint_exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_find_latest_checkpoint_with_keep_all() {
        let temp_dir = env::temp_dir().join("test_checkpoints_find_latest");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: true, // Enable keep_all to test find_latest_checkpoint
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save multiple checkpoints with different iterations
        manager.save_checkpoint(&checkpoint, 1).unwrap();
        manager.save_checkpoint(&checkpoint, 5).unwrap();
        manager.save_checkpoint(&checkpoint, 3).unwrap();
        manager.save_checkpoint(&checkpoint, 10).unwrap();
        manager.save_checkpoint(&checkpoint, 7).unwrap();

        // Verify that find_latest_checkpoint returns the highest iteration (10)
        let latest_path = manager.find_latest_checkpoint().unwrap();
        let filename = latest_path.file_name().unwrap().to_str().unwrap();
        assert!(
            filename.contains("000010"),
            "Expected filename to contain '000010', got: {}",
            filename
        );

        // Test loading the latest checkpoint
        let loaded_checkpoint = manager.load_latest_checkpoint().unwrap();
        assert_eq!(loaded_checkpoint.current_iteration, 0); // The checkpoint data itself

        // Test that checkpoint_exists returns true when keep_all is enabled
        assert!(manager.checkpoint_exists());

        // Create some non-matching files to ensure they're ignored
        let dummy_file1 = temp_dir.join("other_file.bin");
        let dummy_file2 = temp_dir.join("test_abc.bin"); // Wrong pattern
        let dummy_file3 = temp_dir.join("test_999.txt"); // Wrong extension
        fs::write(&dummy_file1, b"dummy").unwrap();
        fs::write(&dummy_file2, b"dummy").unwrap();
        fs::write(&dummy_file3, b"dummy").unwrap();

        // Verify that find_latest_checkpoint still returns the correct file
        let latest_path_after_dummies = manager.find_latest_checkpoint().unwrap();
        assert_eq!(latest_path, latest_path_after_dummies);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_find_latest_checkpoint_no_files() {
        let temp_dir = env::temp_dir().join("test_checkpoints_no_files");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: true,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Test when no checkpoint files exist
        let result = manager.find_latest_checkpoint();
        assert!(result.is_err());
        match result {
            Err(CheckpointError::CheckpointNotFound(path)) => {
                assert_eq!(path, temp_dir);
            }
            _ => panic!("Expected CheckpointNotFound error"),
        }

        // Test that checkpoint_exists returns false
        assert!(!manager.checkpoint_exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cleanup_old_checkpoints() {
        let temp_dir = env::temp_dir().join("test_checkpoints_cleanup");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: true, // Enable keep_all to test cleanup
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save multiple checkpoints
        manager.save_checkpoint(&checkpoint, 1).unwrap();
        manager.save_checkpoint(&checkpoint, 2).unwrap();
        manager.save_checkpoint(&checkpoint, 3).unwrap();
        manager.save_checkpoint(&checkpoint, 4).unwrap();
        manager.save_checkpoint(&checkpoint, 5).unwrap();
        manager.save_checkpoint(&checkpoint, 6).unwrap();
        manager.save_checkpoint(&checkpoint, 7).unwrap();

        // Verify all files exist
        assert_eq!(count_checkpoint_files(&temp_dir, "test"), 7);

        // Keep only 3 files
        manager.cleanup_old_checkpoints(3).unwrap();

        // Verify only 3 files remain
        assert_eq!(count_checkpoint_files(&temp_dir, "test"), 3);

        // Verify that the latest files are kept (5, 6, 7)
        let entries = fs::read_dir(&temp_dir).unwrap();
        let remaining_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                if let Some(filename) = e.file_name().to_str() {
                    if filename.starts_with("test_") && filename.ends_with(".bin") {
                        let iteration_str = &filename[5..filename.len() - 4];
                        iteration_str.parse::<usize>().is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .collect();
        let mut filenames: Vec<_> =
            remaining_files.iter().map(|e| e.file_name().to_str().unwrap().to_string()).collect();
        filenames.sort();
        assert!(filenames.contains(&"test_000005.bin".to_string()));
        assert!(filenames.contains(&"test_000006.bin".to_string()));
        assert!(filenames.contains(&"test_000007.bin".to_string()));

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cleanup_old_checkpoints_keep_all_disabled() {
        let temp_dir = env::temp_dir().join("test_checkpoints_cleanup_disabled");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false, // Disable keep_all
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save a checkpoint (this will overwrite the same file since keep_all is false)
        manager.save_checkpoint(&checkpoint, 1).unwrap();

        // Cleanup should do nothing when keep_all is false
        let result = manager.cleanup_old_checkpoints(1);
        assert!(result.is_ok());

        // File should still exist
        let filename = "test.bin".to_string();
        let filepath = temp_dir.join(filename);
        assert!(filepath.exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cleanup_old_checkpoints_keep_count_zero() {
        let temp_dir = env::temp_dir().join("test_checkpoints_cleanup_zero");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: true,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save multiple checkpoints
        manager.save_checkpoint(&checkpoint, 1).unwrap();
        manager.save_checkpoint(&checkpoint, 2).unwrap();

        // Cleanup with keep_count = 0 should do nothing
        let result = manager.cleanup_old_checkpoints(0);
        assert!(result.is_ok());

        // Both files should still exist
        assert_eq!(count_checkpoint_files(&temp_dir, "test"), 2);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cleanup_old_checkpoints_with_non_matching_files() {
        let temp_dir = env::temp_dir().join("test_checkpoints_cleanup_mixed");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: true,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 0,
            merit_threshold: f64::INFINITY,
            solution_set: None,
            reference_set: vec![],
            unchanged_cycles: 0,
            elapsed_time: 0.0,
            distance_filter_solutions: vec![],
            current_seed: 0,
            target_objective: None,
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: None,
            #[cfg(feature = "rayon")]
            enable_parallel: false,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
        };

        // Save checkpoint files
        manager.save_checkpoint(&checkpoint, 1).unwrap();
        manager.save_checkpoint(&checkpoint, 2).unwrap();
        manager.save_checkpoint(&checkpoint, 3).unwrap();

        // Create some non-matching files that should be ignored
        let dummy_file1 = temp_dir.join("other_file.bin");
        let dummy_file2 = temp_dir.join("test_abc.bin"); // Wrong pattern
        let dummy_file3 = temp_dir.join("test_999.txt"); // Wrong extension
        fs::write(&dummy_file1, b"dummy").unwrap();
        fs::write(&dummy_file2, b"dummy").unwrap();
        fs::write(&dummy_file3, b"dummy").unwrap();

        // Keep only 1 checkpoint file
        manager.cleanup_old_checkpoints(1).unwrap();

        // Verify only 1 checkpoint file remains (the latest one)
        assert_eq!(count_checkpoint_files(&temp_dir, "test"), 1);

        // Verify the dummy files are still there (not affected by cleanup)
        assert!(dummy_file1.exists());
        assert!(dummy_file2.exists());
        assert!(dummy_file3.exists());

        // Verify the remaining checkpoint is the latest one
        let entries = fs::read_dir(&temp_dir).unwrap();
        let checkpoint_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                if let Some(filename) = e.file_name().to_str() {
                    if filename.starts_with("test_") && filename.ends_with(".bin") {
                        let iteration_str = &filename[5..filename.len() - 4];
                        iteration_str.parse::<usize>().is_ok()
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
            .collect();
        let filename = checkpoint_files[0].file_name();
        let remaining_filename = filename.to_str().unwrap();
        assert_eq!(remaining_filename, "test_000003.bin");

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_read_checkpoint_file() {
        use crate::checkpoint::read_checkpoint_file;

        let temp_dir = env::temp_dir().join("test_read_checkpoint_file");
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.clone(),
            checkpoint_name: "test".to_string(),
            save_frequency: 5,
            keep_all: false,
            auto_resume: true,
        };

        let manager = CheckpointManager::new(config).unwrap();

        // Create a test checkpoint
        let checkpoint = OQNLPCheckpoint {
            params: OQNLPParams::default(),
            current_iteration: 123,
            merit_threshold: 2.5,
            solution_set: Some(SolutionSet {
                solutions: Array1::from(vec![LocalSolution {
                    point: array![3.0, 4.0],
                    objective: -2.5,
                }]),
            }),
            reference_set: vec![array![1.0, 2.0], array![5.0, 6.0]],
            unchanged_cycles: 10,
            elapsed_time: 250.75,
            distance_filter_solutions: vec![],
            current_seed: 42,
            target_objective: Some(-3.0),
            exclude_out_of_bounds: false,
            #[cfg(feature = "rayon")]
            batch_iterations: Some(7),
            #[cfg(feature = "rayon")]
            enable_parallel: true,
            timestamp: "2025-08-01T12:00:00Z".to_string(),
        };

        // Save the checkpoint using the manager
        let saved_path = manager.save_checkpoint(&checkpoint, 123).unwrap();
        assert!(saved_path.exists());

        // Test reading the checkpoint file directly
        let loaded_checkpoint = read_checkpoint_file(&saved_path).unwrap();

        // Verify all fields match
        assert_eq!(loaded_checkpoint.current_iteration, 123);
        assert_eq!(loaded_checkpoint.merit_threshold, 2.5);
        assert_eq!(loaded_checkpoint.unchanged_cycles, 10);
        assert_eq!(loaded_checkpoint.elapsed_time, 250.75);
        assert_eq!(loaded_checkpoint.current_seed, 42);
        assert_eq!(loaded_checkpoint.target_objective, Some(-3.0));
        assert_eq!(loaded_checkpoint.timestamp, "2025-08-01T12:00:00Z");
        assert_eq!(loaded_checkpoint.reference_set.len(), 2);

        // Verify solution set
        if let Some(solution_set) = loaded_checkpoint.solution_set {
            assert_eq!(solution_set.solutions.len(), 1);
            assert_eq!(solution_set.solutions[0].point, array![3.0, 4.0]);
            assert_eq!(solution_set.solutions[0].objective, -2.5);
        } else {
            panic!("Expected solution set to be Some");
        }

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_read_checkpoint_file_not_found() {
        use crate::checkpoint::read_checkpoint_file;

        let non_existent_path = env::temp_dir().join("non_existent_checkpoint.bin");

        // Ensure the file doesn't exist
        assert!(!non_existent_path.exists());

        // Test reading non-existent file
        let result = read_checkpoint_file(&non_existent_path);
        assert!(result.is_err());

        match result {
            Err(CheckpointError::CheckpointNotFound(path)) => {
                assert_eq!(path, non_existent_path);
            }
            _ => panic!("Expected CheckpointNotFound error"),
        }
    }

    #[test]
    fn test_read_checkpoint_file_corrupted_data() {
        use crate::checkpoint::read_checkpoint_file;

        let temp_dir = env::temp_dir().join("test_read_checkpoint_corrupted");
        fs::create_dir_all(&temp_dir).unwrap();

        // Create a file with invalid/corrupted data
        let corrupted_file = temp_dir.join("corrupted.bin");
        fs::write(&corrupted_file, b"invalid checkpoint data").unwrap();

        // Test reading corrupted file
        let result = read_checkpoint_file(&corrupted_file);
        assert!(result.is_err());

        match result {
            Err(CheckpointError::SerializationError(_)) => {
                // Expected error type
            }
            _ => panic!("Expected SerializationError"),
        }

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_read_checkpoint_file_empty_file() {
        use crate::checkpoint::read_checkpoint_file;

        let temp_dir = env::temp_dir().join("test_read_checkpoint_empty");
        fs::create_dir_all(&temp_dir).unwrap();

        // Create an empty file
        let empty_file = temp_dir.join("empty.bin");
        fs::write(&empty_file, b"").unwrap();

        // Test reading empty file
        let result = read_checkpoint_file(&empty_file);
        assert!(result.is_err());

        match result {
            Err(CheckpointError::SerializationError(_)) => {
                // Expected error type
            }
            _ => panic!("Expected SerializationError"),
        }

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }
}
