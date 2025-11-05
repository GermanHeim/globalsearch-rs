<p align="center">
    <img
        width="500"
        src="https://raw.githubusercontent.com/GermanHeim/globalsearch-rs/main/python/media/logo.png"
        alt="pyGlobalSearch"
    />
    <p align="center">
        Python bindings for the globalsearch Rust crate
    </p>
    <p align="center">
        <a href="https://germanheim.github.io/globalsearch-rs-website/">Website</a> | <a href="https://germanheim.github.io/pyglobalsearch-website/api.html"> Python API Docs</a> | <a href="https://github.com/GermanHeim/globalsearch-rs/tree/main/python/examples">Examples</a>
    </p>
</p>

<div align="center">
    <a href="https://pypi.org/project/pyglobalsearch/">
        <img src="https://img.shields.io/pypi/v/pyglobalsearch?logo=pypi&logoColor=%23ffd343&color=%230060df">
    </a>
    <a href="https://docs.astral.sh/uv/">
        <img src="https://img.shields.io/badge/uv-black?logo=uv" alt="uv" />
    </a> 
    <a href="https://docs.astral.sh/ruff/">
        <img src="https://img.shields.io/badge/ruff-black?logo=ruff" alt="ruff" />
    </a> 
    <a href="https://github.com/GermanHeim/globalsearch-rs/actions/workflows/pyglobalsearch-ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/GermanHeim/globalsearch-rs/pyglobalsearch-ci.yml?branch=main&label=pyglobalsearch%20CI&logo=github" alt="PyGlobalSearch CI"/>
    </a>
    <a href="https://github.com/GermanHeim/globalsearch-rs/blob/main/LICENSE.txt">
        <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" />
    </a>
</div>

`pyglobalsearch`: Python interface for the Rust crate `globalsearch`. The Rust crate implements a modified version of the _OQNLP_ (_OptQuest/NLP_) algorithm with the core ideas from "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization" by Ugray et al. (2007). It combines scatter search metaheuristics with local minimization for global optimization of nonlinear problems.

Similar to MATLAB's `GlobalSearch` \[2\]. The bindings are built using [pyo3](https://github.com/PyO3/pyo3) and [Maturin](https://github.com/PyO3/maturin).

## Installation

Install the Python package from PyPI:

```bash
pip install pyglobalsearch
```

## Usage

1. Import the `pyglobalsearch` module:

   ```python
   import pyglobalsearch as gs
   ```

2. Define the parameters of the optimization problem:

   ```python
   params = gs.PyOQNLPParams(
    iterations=100,
    population_size=500,
    wait_cycle=10,
    threshold_factor=0.75,
    distance_factor=0.1,
   )
   ```

3. Define your problem:

   ```python
   from numpy.typing import NDArray
   
   def objective(x: NDArray[np.float64]) -> float:
        return x[0] ** 2 + x[1] ** 2

   def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2 * x[0], 2 * x[1]])

   def variable_bounds() -> NDArray[np.float64]:
        return np.array([[-3, 3], [-2, 2]])

   problem = gs.PyProblem(objective, variable_bounds, gradient)
   ```

4. Run the optimization:

   ```python
   result = gs.optimize(problem, params, local_solver="COBYLA", seed=0)
   print(result)
   ```

## Developing

### Prerequisites

- [Rust](https://rustup.rs/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Python](https://www.python.org/downloads/) (Python can also be installed using `uv python install`)
- [Maturin](https://www.maturin.rs/installation.html)

### Setup

1. Clone the repository and navigate to the Python directory:

   ```bash
   git clone https://github.com/GermanHeim/globalsearch-rs.git
   cd globalsearch-rs/python
   ```

2. Create a virtual environment and activate it:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Modify the code and use `Maturin` to build the Python package:

   ```bash
   maturin develop
   ```

## Limitations

The Python bindings are still in development and may not be fully functional. The API is subject to change. If you encounter any issues, please open an issue in the [issue tracker](https://github.com/GermanHeim/globalsearch-rs/issues) or submit a pull request.

Additionally, the Python bindings have limitations:

- No support for checkpointing
- Maximum number of constraints is 1000

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/GermanHeim/globalsearch-rs/blob/main/LICENSE.txt) for more information.

## Citing PyGlobalSearch

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09234/status.svg)](https://doi.org/10.21105/joss.09234)

If `PyGlobalSearch` has been significant in your research, and you would like to acknowledge the project in your academic publication, we suggest citing the following paper:

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
