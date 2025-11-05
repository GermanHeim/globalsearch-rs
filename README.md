<p align="center">
    <img
        width="500"
        src="https://raw.githubusercontent.com/GermanHeim/globalsearch-rs/main/media/logo.png"
        alt="GlobalSearch-rs"
    />
    <p align="center">
        A multistart framework for global optimization with scatter search and local NLP solvers written in Rust
    </p>
    <p align="center">
        <a href="https://germanheim.github.io/globalsearch-rs-website/">Website</a> | <a href="https://docs.rs/globalsearch/latest/globalsearch/">Docs</a> | <a href="https://github.com/GermanHeim/globalsearch-rs/tree/main/examples">Examples</a>
    </p>
</p>

<div align="center">
    <a href="https://crates.io/crates/globalsearch">
        <img src="https://img.shields.io/crates/v/globalsearch?logo=rust&color=E05D44" alt="crates version" />
    </a> 
    <a href="https://pypi.org/project/pyglobalsearch/">
        <img src="https://img.shields.io/pypi/v/pyglobalsearch?logo=pypi&logoColor=%23ffd343&color=%230060df">
    </a>
    <a href="https://github.com/GermanHeim/globalsearch-rs/actions/workflows/globalsearch-rs-CI.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/GermanHeim/globalsearch-rs/globalsearch-rs-CI.yml?branch=main&label=globalsearch%20CI&logo=github" alt="CI" />
    </a> 
    <a href="https://app.codecov.io/gh/GermanHeim/globalsearch-rs">
        <img src="https://img.shields.io/codecov/c/github/GermanHeim/globalsearch-rs?logo=codecov&color=FF0077&token=C2FI2Z26ME" alt="Codecov" />
    </a>
    <a href="https://github.com/GermanHeim/globalsearch-rs/blob/main/LICENSE.txt">
        <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" />
    </a>
</div>

`globalsearch-rs`: Rust implementation of a modified version of the _OQNLP_ (_OptQuest/NLP_) algorithm with the core ideas from "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization" by Ugray et al. (2007). It combines scatter search metaheuristics with local minimization for global optimization of nonlinear problems.

Similar to MATLAB's `GlobalSearch` \[2\], using cobyla, argmin, rayon and ndarray.

## Features

- 🐍 [Python Bindings](https://github.com/GermanHeim/globalsearch-rs/tree/main/python)

- 🎯 Multistart heuristic framework for global optimization

- 📦 Local optimization using the cobyla \[3\] and argmin crate \[4\]

- 🚀 Parallel execution using Rayon

- 🔄 Checkpointing support for long-running optimizations

## Installation

### Using as a dependency

Add this to your `Cargo.toml`:

```toml
[dependencies]
globalsearch = "0.4.0"
```

Or use `cargo add globalsearch` in your project directory.

### Building from source

1. Install Rust toolchain using [rustup](https://rustup.rs/).
2. Clone repository:

   ```bash
   git clone https://github.com/GermanHeim/globalsearch-rs.git
   cd globalsearch-rs
   ```

3. Build the project:

   ```bash
   cargo build --release
   ```

## Usage

1. Define a problem by implementing the `Problem` trait.

   ```rust
   use ndarray::{array, Array1, Array2};
   use globalsearch::problem::Problem;
   use globalsearch::types::EvaluationError;

   pub struct MinimizeProblem;
   impl Problem for MinimizeProblem {
       fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
           Ok(
               ..., // Your objective function here
           )
       }

       fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
           Ok(array![
               ..., // Optional: Gradient of your objective function here
           ])
       }

       fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
           Ok(array![
               ..., // Optional: Hessian of your objective function here
           ])
       }

       fn variable_bounds(&self) -> Array2<f64> {
           array![[..., ...], [..., ...]] // Lower and upper bounds for each variable
       }

       fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
            vec![
              ..., // Optional: Constraint functions here, only valid with COBYLA
            ]
       }
   }
   ```

   Where the `Problem` trait is defined as:

   ```rust
   pub trait Problem {
       fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError>;
       fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError>;
       fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError>;
       fn variable_bounds(&self) -> Array2<f64>;
       fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64>;
   }
   ```

   The `constraints` method allows you to define constraint functions for constrained optimization problems. Constraints should follow the sign convention:
   - **Positive or zero**: constraint satisfied  
   - **Negative**: constraint violated

   Example:

   ```rust
   impl Problem for MinimizeProblem {
       // ...
       fn constraints(&self) -> Vec<fn(&[f64], &mut ()) -> f64> {
           vec![
               |x: &[f64], _: &mut ()| 1.0 - x[0] - x[1], // x[0] + x[1] <= 1.0
               |x: &[f64], _: &mut ()| x[0] - 0.5,        // x[0] >= 0.5
           ]
       }
   }
   ```

   Depending on your choice of local solver, you might need to implement the `gradient` and `hessian` methods. Learn more about the local solver configuration in the [argmin docs](https://docs.rs/argmin/latest/argmin/solver/index.html) or the [`LocalSolverType`](https://docs.rs/globalsearch/latest/globalsearch/types/enum.LocalSolverType.html).

   > 🔴 **Note:** If using a solver that isn't COBYLA, variable bounds are only used in the scatter search phase of the algorithm. The local solver is unconstrained (See [argmin issue #137](https://github.com/argmin-rs/argmin/issues/137)) and therefor can return solutions out of bounds. You can use OQNLP's `exclude_out_of_bounds` method to handle this if needed.

2. Set OQNLP parameters

   ```rust
   use globalsearch::types::{LocalSolverType, OQNLPParams};
   use globalsearch::local_solver::builders::SteepestDescentBuilder;

   let params: OQNLPParams = OQNLPParams {
       iterations: 125,
       wait_cycle: 10,
       threshold_factor: 0.2,
       distance_factor: 0.75,
       population_size: 250,
       local_solver_type: LocalSolverType::SteepestDescent,
       local_solver_config: SteepestDescentBuilder::default().build(),
       seed: 0,
   };
   ```

   Or use the default parameters (which use COBYLA):

   ```rust
   let params = OQNLPParams::default();
   ```

   Where `OQNLPParams` is defined as:

   ```rust
   pub struct OQNLPParams {
       pub iterations: usize,
       pub wait_cycle: usize,
       pub threshold_factor: f64,
       pub distance_factor: f64,
       pub population_size: usize,
       pub local_solver_type: LocalSolverType,
       pub local_solver_config: LocalSolverConfig,
       pub seed: u64,
   }
   ```

   And `LocalSolverType` is defined as:

   ```rust
   pub enum LocalSolverType {
       LBFGS,
       NelderMead,
       SteepestDescent,
       TrustRegion,
       NewtonCG,
       COBYLA,
   }
   ```

   You can also modify the local solver configuration for each type of local solver. See [`builders.rs`](https://github.com/GermanHeim/globalsearch-rs/tree/main/src/local_solver/builders.rs) for more details.

3. Run the optimizer

   ```rust
   use oqnlp::{OQNLP, OQNLPParams};
   use types::{SolutionSet}

   fn main() -> Result<(), Box<dyn std::error::Error>> {
        let problem = MinimizeProblem;
        let params: OQNLPParams = OQNLPParams {
                iterations: 125,
                wait_cycle: 10,
                threshold_factor: 0.2,
                distance_factor: 0.75,
                population_size: 250,
                local_solver_type: LocalSolverType::SteepestDescent,
                local_solver_config: SteepestDescentBuilder::default().build(),
                seed: 0,
            };

        let mut optimizer: OQNLP<MinimizeProblem> = OQNLP::new(problem, params)?;

        // OQNLP returns a solution set with the best solutions found
        let solution_set: SolutionSet = optimizer.run()?;
        println!("{}", solution_set)

        Ok(())
   }
   ```

## Project Structure

```plaintext
src/
├── lib.rs # Module declarations
├── oqnlp.rs # Core OQNLP algorithm implementation
├── scatter_search.rs # Scatter search component
├── local_solver/
│   ├── builders.rs # Local solver configuration builders
│   └── runner.rs # Local solver runner
├── filters.rs # Merit and distance filtering logic
├── problem.rs # Problem trait
├── types.rs # Data structures and parameters
└── checkpoint.rs # Checkpointing module
python/ # Python bindings
```

## Dependencies

- [argmin](https://github.com/argmin-rs/argmin)
- [COBYLA](https://github.com/relf/cobyla)
- [ndarray](https://github.com/rust-ndarray/ndarray)
- [rayon](https://github.com/rayon-rs/rayon) [feature: `rayon`]
- [kdam](https://github.com/clitic/kdam) [feature: `progress_bar`]
- [rand](https://github.com/rust-random/rand)
- [thiserror](https://github.com/dtolnay/thiserror)
- [criterion.rs](https://github.com/bheisler/criterion.rs) [dev-dependency]
- [serde](https://github.com/serde-rs/serde) [feature: `checkpointing`]
- [chrono](https://github.com/chronotope/chrono) [feature: `checkpointing`]
- [bincode](https://github.com/bincode-org/bincode) [feature: `checkpointing`]

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/GermanHeim/globalsearch-rs/blob/main/LICENSE.txt) for more information.

## References

\[1\] Zsolt Ugray, Leon Lasdon, John Plummer, Fred Glover, James Kelly, Rafael Martí, (2007) Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization. INFORMS Journal on Computing 19(3):328-340. <http://dx.doi.org/10.1287/ijoc.1060.0175>

\[2\] GlobalSearch. The MathWorks, Inc. Available at: <https://www.mathworks.com/help/gads/globalsearch.html> (Accessed: 27 January 2025)

\[3\] Rémi Lafage. cobyla - a pure Rust implementation. GitHub repository. MIT License. Available at: <https://github.com/relf/cobyla> (Accessed: 17 September 2025)

\[4\] Kroboth, S. argmin{}. Available at: <https://argmin-rs.org/> (Accessed: 25 January 2025)
