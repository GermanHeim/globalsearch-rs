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

Similar to MATLAB's `GlobalSearch` \[2\], using Basin, argmin, Rayon, and ndarray.

## Features

- 🐍 [Python Bindings](https://github.com/GermanHeim/globalsearch-rs/tree/main/python)

- 🎯 Multistart heuristic framework for global optimization

- 📦 Local optimization using the Basin \[3\] and argmin \[4\] crates

- 🚀 Parallel execution using Rayon

- 🔄 Checkpointing support for long-running optimizations

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

       fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
            Ok(array![
              ..., // Optional: Constraint values here, only valid with COBYLA
            ])
       }
   }
   ```

   The `constraints` method (only available with the COBYLA local solver) evaluates every constraint at a point. Its output length and order must remain stable throughout optimization. Constraints follow this sign convention:
   - **Positive or zero**: constraint satisfied  
   - **Negative**: constraint violated

   Example:

   ```rust
   impl Problem for MinimizeProblem {
       // ...
       fn constraints(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
           Ok(array![
               1.0 - x[0] - x[1], // x[0] + x[1] <= 1.0
               x[0] - 0.5,        // x[0] >= 0.5
           ])
       }
   }
   ```

   Depending on your choice of local solver, you might need to implement the `gradient` and `hessian` methods. Learn more in the [Basin docs](https://docs.rs/basin/1.11.0/basin/) and [argmin docs](https://docs.rs/argmin/latest/argmin/solver/index.html), or see [`LocalSolverType`](https://docs.rs/globalsearch/latest/globalsearch/types/enum.LocalSolverType.html).

   > **Bounds:** COBYLA, Basin L-BFGS-B, Basin bounded Nelder-Mead, and Basin BOBYQA enforce variable bounds during local optimization. Other local solvers use bounds only during scatter search and can return points outside them. Use `exclude_out_of_bounds` to filter those solutions if needed. Only COBYLA supports nonlinear constraints.

2. Set OQNLP parameters

   ```rust
   use globalsearch::types::OQNLPParams;
   use globalsearch::local_solver::builders::SteepestDescentBuilder;

   let params: OQNLPParams = OQNLPParams {
       iterations: 125,
       wait_cycle: 10,
       threshold_factor: 0.2,
       distance_factor: 0.75,
       population_size: 250,
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
       pub local_solver_config: LocalSolverConfig,
       pub seed: u64,
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

## Installation

### Using as a dependency

Add this to your `Cargo.toml`:

```toml
[dependencies]
globalsearch = "0.6.0"
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

## Choosing a local-solver backend

The default `argmin` feature provides the existing L-BFGS, Nelder-Mead, steepest
descent, trust-region, and Newton-CG solvers. COBYLA always uses Basin and remains
the default local solver, including when all default features are disabled.

Enable the `basin` feature for additional, explicitly named Basin solvers:

```toml
[dependencies]
globalsearch = { version = "0.6", default-features = false, features = ["basin"] }
```

Keep default features enabled to use both backends in the same application.
Enabling `basin` does not change existing argmin solver selections. Both paths
use the existing `Problem` trait and ndarray arrays. Rust 1.87 is required.

| Rust builder | Python factory / solver name | Derivatives | Local constraints |
| --- | --- | --- | --- |
| `BasinLBFGSBuilder` | `basin_lbfgs` | Gradient | None |
| `BasinGradientDescentBuilder` | `basin_gradient_descent` | Gradient | None |
| `BasinTrustRegionBuilder` | `basin_trust_region` | Gradient and Hessian | None |
| `BasinNelderMeadBuilder` | `basin_nelder_mead` | None | None |
| `BasinLBFGSBBuilder` | `basin_lbfgsb` | Gradient | Box bounds |
| `BasinBoundedNelderMeadBuilder` | `basin_bounded_nelder_mead` | None | Box bounds |
| `BasinBOBYQABuilder` | `basin_bobyqa` | None | Box bounds |
| `COBYLABuilder` | `cobyla` | None | Box bounds and nonlinear inequalities |

```rust
use globalsearch::local_solver::builders::BasinLBFGSBBuilder;
use globalsearch::types::OQNLPParams;

let params = OQNLPParams {
    local_solver_config: BasinLBFGSBBuilder::default()
        .max_iter(500)
        .tolerance_projected_grad(1e-8)
        .history_size(10)
        .build(),
    ..OQNLPParams::default()
};
```

The seven new solvers reject problems with nonempty nonlinear constraints and
report that COBYLA is required. They propagate callback errors and require
analytic derivatives where listed. Configuration errors are returned when the
local solve starts. The bounded methods project infeasible starting points
before callbacks; bounded Nelder-Mead initializes its simplex toward the box
interior. Projection can still collapse vertices during later Nelder-Mead steps.

New Basin builders default to 1,000 executor iterations. Initialization and line
searches can evaluate the objective multiple times per iteration. COBYLA retains
its existing interpretation of `max_iter` as an objective-evaluation budget.
Gradient tolerances default to `1e-6` (Euclidean norm for unconstrained methods,
projected-gradient infinity norm for L-BFGS-B). Optional tolerances accept `None`
to disable or zero for an exact threshold. Cost-change stopping is disabled by
default for L-BFGS, L-BFGS-B, and gradient descent. They use Basin's default
More–Thuente line search; argmin line-search configurations do not apply.

Nelder-Mead uses standard coefficients, an absolute simplex step of `0.1`, and
requires both simplex-size (`1e-6`, infinity norm) and simplex-cost (`1e-8`)
tolerances when both are enabled. Trust region defaults to Steihaug, initial
radius `1`, maximum radius `100`, and acceptance threshold `0.125`; Cauchy is
also available. BOBYQA defaults to initial radius `1`, final radius `1e-6`, and
`2n+1` interpolation points. It automatically reduces radii for narrow boxes.
Its interpolation count must lie in `[2n+1, (n+1)(n+2)/2]`.

Python distributions include both backends. For example:

```python
config = gs.builders.basin_lbfgsb(max_iter=500, tolerance_projected_grad=1e-8)
result = gs.optimize(problem, params, local_solver_config=config)
# Or select default settings by name:
result = gs.optimize(problem, params, local_solver="basin_bobyqa")
```

Names are case-insensitive and accept underscores, hyphens, or compact spelling.
When both a name and a configuration are provided, they must select the same
solver. The configuration classes are available as `gs.builders.PyBasinLBFGS`,
`PyBasinLBFGSB`, and corresponding names for the other methods.

Checkpoint files retain the existing enum encoding when enabling the `basin`
feature with the same argmin feature setting. Reading a checkpoint containing a
Basin solver requires enabling that feature. Checkpoints are not portable across
changes to the argmin feature setting.

## Dependencies

- [ndarray](https://github.com/rust-ndarray/ndarray)
- [Basin](https://github.com/jolars/basin) [COBYLA always available; other solvers: `basin` feature]
- [argmin](https://github.com/argmin-rs/argmin) [feature: `argmin`]
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

## Citing Globalsearch-rs

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09234/status.svg)](https://doi.org/10.21105/joss.09234)

If `GlobalSearch-rs` has been significant in your research, and you would like to acknowledge the project in your academic publication, we suggest citing the following paper:

```bibtex
@article{Heim2025,
  author    = {Heim, Germán Martín},
  doi       = {10.21105/joss.09234},
  journal   = {Journal of Open Source Software},
  number    = {115},
  pages     = {9234},
  publisher = {The Open Journal},
  title     = {GlobalSearch-rs: A multistart framework for global optimization written in Rust},
  url       = {https://doi.org/10.21105/joss.09234},
  volume    = {10},
  year      = {2025}
}
```

## References

\[1\] Zsolt Ugray, Leon Lasdon, John Plummer, Fred Glover, James Kelly, Rafael Martí, (2007) Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization. INFORMS Journal on Computing 19(3):328-340. <http://dx.doi.org/10.1287/ijoc.1060.0175>

\[2\] GlobalSearch. The MathWorks, Inc. Available at: <https://www.mathworks.com/help/gads/globalsearch.html> (Accessed: 27 January 2025)

\[3\] Johan Larsson. Basin—numerical optimization in pure Rust. Available at: <https://basin.rs> (Accessed: 7 September 2026)

\[4\] Kroboth, S. argmin{}. Available at: <https://argmin-rs.org/> (Accessed: 25 January 2025)
