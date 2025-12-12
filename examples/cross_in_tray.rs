/// Cross-in-Tray function
/// The Cross-in-Tray function is defined as:
///
/// f(x, y) = -0.0001 × (|h(x, y)| + 1)^0.1
/// where h(x, y) = sin(x) sin(y) exp(|100 - √(x² + y²)| / π)
///
/// The function is defined on the domain x ∈ [-10, 10] and y ∈ [-10, 10].
/// The function has four global minima at f(1.34941, -1.34941) = -2.06261, f(1.34941, 1.34941) = -2.06261, f(-1.34941, 1.34941) = -2.06261, and f(-1.34941, -1.34941) = -2.06261.
/// The function is continuous, differentiable and non-convex.
///
/// References:
/// Surjanovic, S. and Bingham, D. (no date) Virtual Library of Simulation Experiments: Test Functions and Datasets, Cross-in-Tray Function. Available at: https://www.sfu.ca/~ssurjano/crossit.html (Accessed: 02 February 2025).
use globalsearch::local_solver::builders::LBFGSBuilder;
use globalsearch::problem::Problem;
use globalsearch::{
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{Array1, Array2, array};

#[derive(Debug, Clone)]
pub struct CrossInTray;

impl CrossInTray {
    /// Helper function to compute h(x,y)
    fn h(x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        let u = 100.0 - r / std::f64::consts::PI;
        let t = u.abs();
        x.sin() * y.sin() * t.exp()
    }
}

impl Problem for CrossInTray {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let x_val = x[0];
        let y_val = x[1];
        let h = Self::h(x_val, y_val);
        let val = -0.0001 * ((h.abs() + 1.0).powf(0.1));
        Ok(val)
    }

    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        let x_val = x[0];
        let y_val = x[1];

        // Compute common terms
        let r = (x_val * x_val + y_val * y_val).sqrt();
        let u = 100.0 - r / std::f64::consts::PI;
        let t = u.abs();
        let exp_t = t.exp();

        let sinx = x_val.sin();
        let siny = y_val.sin();
        let cosx = x_val.cos();
        let cosy = y_val.cos();

        // h(x,y) and its absolute value
        let h = sinx * siny * exp_t;
        let abs_h = h.abs();
        let g = abs_h + 1.0;

        // Derivative of f(x,y) = -0.0001*(g^0.1) with respect to g:
        // f'(g) = -0.0001 * 0.1 * g^(0.1-1)
        let common = -0.0001 * 0.1 * g.powf(0.1 - 1.0);

        // Derivative of |h| is sign(h)*h'
        let sign_h = if h > 0.0 {
            1.0
        } else if h < 0.0 {
            -1.0
        } else {
            0.0
        };

        // Derivatives of t = |u| where u = 100 - r/π.
        let (t_x, t_y) = if r != 0.0 {
            let sign_u = if u > 0.0 {
                1.0
            } else if u < 0.0 {
                -1.0
            } else {
                0.0
            };
            (
                sign_u * (-x_val / (std::f64::consts::PI * r)),
                sign_u * (-y_val / (std::f64::consts::PI * r)),
            )
        } else {
            (0.0, 0.0)
        };

        let h_x = cosx * siny * exp_t + sinx * siny * exp_t * t_x;
        let h_y = sinx * cosy * exp_t + sinx * siny * exp_t * t_y;

        let dfdx = common * sign_h * h_x;
        let dfdy = common * sign_h * h_y;

        Ok(array![dfdx, dfdy])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![[-10.0, 10.0], [-10.0, 10.0]]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem: CrossInTray = CrossInTray;

    let params: OQNLPParams = OQNLPParams {
        iterations: 300,
        wait_cycle: 10,
        threshold_factor: 0.1,
        distance_factor: 0.75,
        population_size: 350,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default().build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<CrossInTray> = OQNLP::new(problem.clone(), params)?.verbose();
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("{}", solution_set);

    Ok(())
}
