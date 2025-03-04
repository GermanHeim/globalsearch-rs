use globalsearch::local_solver::builders::{
    LBFGSBuilder, NelderMeadBuilder, NewtonCGBuilder, SteepestDescentBuilder, TrustRegionBuilder,
};
use globalsearch::oqnlp::OQNLP;
use globalsearch::problem::Problem;
use globalsearch::types::{EvaluationError, LocalSolverType, OQNLPParams};
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyOQNLPParams {
    #[pyo3(get, set)]
    pub iterations: usize,
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub wait_cycle: usize,
    #[pyo3(get, set)]
    pub threshold_factor: f64,
    #[pyo3(get, set)]
    pub distance_factor: f64,
}

#[pymethods]
impl PyOQNLPParams {
    #[new]
    #[pyo3(signature = (
        iterations = 300,
        population_size = 1000,
        wait_cycle = 15,
        threshold_factor = 0.2,
        distance_factor = 0.75,
    ))]
    #[pyo3(
        text_signature = "(iterations=300, population_size=1000, wait_cycle=15, threshold_factor=0.2, distance_factor=0.75)"
    )]
    fn new(
        iterations: usize,
        population_size: usize,
        wait_cycle: usize,
        threshold_factor: f64,
        distance_factor: f64,
    ) -> Self {
        PyOQNLPParams {
            iterations,
            population_size,
            wait_cycle,
            threshold_factor,
            distance_factor,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyProblem {
    #[pyo3(get, set)]
    objective: PyObject,
    #[pyo3(get, set)]
    variable_bounds: PyObject,
    #[pyo3(get, set)]
    gradient: Option<PyObject>,
    #[pyo3(get, set)]
    hessian: Option<PyObject>,
}

impl Clone for PyProblem {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            objective: self.objective.clone_ref(py),
            variable_bounds: self.variable_bounds.clone_ref(py),
            gradient: self.gradient.as_ref().map(|g| g.clone_ref(py)),
            hessian: self.hessian.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (objective, variable_bounds, gradient=None, hessian=None))]
    fn new(
        objective: PyObject,
        variable_bounds: PyObject,
        gradient: Option<PyObject>,
        hessian: Option<PyObject>,
    ) -> Self {
        PyProblem {
            objective,
            variable_bounds,
            gradient,
            hessian,
        }
    }
}

impl Problem for PyProblem {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        Python::with_gil(|py| {
            let x_py = x
                .to_vec()
                .into_pyobject(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            let result = self
                .objective
                .call1(py, (x_py,))
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
            result
                .extract(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
        })
    }

    fn variable_bounds(&self) -> Array2<f64> {
        Python::with_gil(|py| {
            let result = self
                .variable_bounds
                .call0(py)
                .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                .and_then(|res| {
                    res.extract::<Vec<Vec<f64>>>(py)
                        .map_err(|e| EvaluationError::InvalidInput(e.to_string()))
                });

            match result {
                Ok(bounds) => {
                    let rows = bounds.len();
                    let cols = if rows > 0 { bounds[0].len() } else { 0 };
                    Array2::from_shape_vec((rows, cols), bounds.into_iter().flatten().collect())
                        .unwrap()
                }
                Err(_) => panic!("Variable bounds must be a 2D array of floats"),
            }
        })
    }

    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        if let Some(grad_fn) = &self.gradient {
            Python::with_gil(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = grad_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let grad_vec: Vec<f64> = result
                    .extract(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                Ok(Array1::from(grad_vec))
            })
        } else {
            Err(EvaluationError::GradientNotImplemented)
        }
    }

    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        if let Some(hess_fn) = &self.hessian {
            Python::with_gil(|py| {
                let x_py = x
                    .to_vec()
                    .into_pyobject(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;
                let result = hess_fn
                    .call1(py, (x_py,))
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let hess_vec: Vec<Vec<f64>> = result
                    .extract(py)
                    .map_err(|e| EvaluationError::InvalidInput(e.to_string()))?;

                let size = hess_vec.len();
                let flat_hess: Vec<f64> = hess_vec.into_iter().flatten().collect();

                Array2::from_shape_vec((size, size), flat_hess).map_err(|_| {
                    EvaluationError::InvalidInput("Hessian shape mismatch".to_string())
                })
            })
        } else {
            Err(EvaluationError::HessianNotImplemented)
        }
    }
}

/// Python wrapper around the OQNLP optimizer
///
/// This function takes a problem, parameters and optionally a local solver and its configuration
/// and returns the best solution found by the optimizer.
#[pyfunction]
#[pyo3(signature = (problem, params, local_solver=None, seed=None))]
fn optimize(
    problem: PyProblem,
    params: PyOQNLPParams,
    local_solver: Option<&str>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        // Convert local_solver string to enum
        let solver_type = LocalSolverType::from_string(local_solver.unwrap_or("lbfgs"))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        let seed = seed.unwrap_or(0);

        // Create local solver configuration (default)
        let local_solver_config = match solver_type {
            LocalSolverType::LBFGS => LBFGSBuilder::default().build(),
            LocalSolverType::NewtonCG => NewtonCGBuilder::default().build(),
            LocalSolverType::TrustRegion => TrustRegionBuilder::default().build(),
            LocalSolverType::NelderMead => NelderMeadBuilder::default().build(),
            LocalSolverType::SteepestDescent => SteepestDescentBuilder::default().build(),
        };

        let params: OQNLPParams = OQNLPParams {
            iterations: params.iterations,
            population_size: params.population_size,
            wait_cycle: params.wait_cycle,
            threshold_factor: params.threshold_factor,
            distance_factor: params.distance_factor,
            seed: seed,
            local_solver_type: solver_type,
            local_solver_config: local_solver_config,
        };

        let mut optimizer =
            OQNLP::new(problem, params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let solution_set = optimizer.run();

        let binding = solution_set.map_err(|e| PyValueError::new_err(e.to_string()))?;

        let solutions: Vec<PyObject> = binding
            .solutions()
            .map(|sol| {
                let dict = PyDict::new(py);
                dict.set_item("x", sol.point.to_vec()).unwrap();
                dict.set_item("fun", sol.objective).unwrap();
                dict.into()
            })
            .collect();

        Ok(solutions.into_py(py))
    })
}

#[pymodule]
/// pyGlobalSearch
///
/// This library provides a Python interface to the `globalsearch-rs` crate.
/// `globalsearch` is a Rust crate for global optimization.
///
/// `globalsearch-rs` is a Rust implementation of the OQNLP (OptQuest/NLP) algorithm
/// with the core ideas from "Scatter Search and Local NLP Solvers: A Multistart Framework for Global Optimization"
/// by Ugray et al. (2007). It combines scatter search metaheuristics with local
/// minimization for global optimization of nonlinear problems.
///
/// It is similar to MATLAB's `GlobalSearch`.
fn pyglobalsearch(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_class::<PyOQNLPParams>()?;
    m.add_class::<PyProblem>()?;

    Ok(())
}
