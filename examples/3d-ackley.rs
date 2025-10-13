/// 3-D Ackley function
/// The 3-D Ackley function is defined as:
///
/// f(x) = -a exp(-b √(∑ᵢ₌₁³ xᵢ²/3)) - exp((1/3) ∑ᵢ₌₁³ cos(cxᵢ)) + a + exp(1)
///
/// With a = 20, b = 0.2 and c = 2π.
/// The function is defined on the hypercube `[-32.768, 32.768]`.
/// The function has a global minimum at `x = [0, 0, 0]` with `f(x) = 0`.
/// The function is a multimodal continuous, differentiable and non-convex.
///
/// References:
///
/// Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 15-16. Retrieved January 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
use globalsearch::local_solver::builders::{HagerZhangBuilder, LBFGSBuilder};
use globalsearch::problem::Problem;
use globalsearch::{
    oqnlp::OQNLP,
    types::{EvaluationError, LocalSolverType, OQNLPParams, SolutionSet},
};
use ndarray::{array, Array1, Array2};

/// IMPORTANT: For some reason, this example doesn't work using steepest descent or LBFGS.
/// The local solver gets stuck trying to minimize in the stage 1 if I don't pass HagerZhangBuilder (default is MoreThuente).
/// We have to check this implementation. I believe this is a problem with the implementation of argmin.

#[derive(Debug, Clone)]
pub struct ThreeDAckley {
    a: f64,
    b: f64,
    c: f64,
}

impl ThreeDAckley {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }
}

impl Problem for ThreeDAckley {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let norm = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2)) / 3.0;
        let cos_sum = (x[0] * self.c).cos() + (x[1] * self.c).cos() + (x[2] * self.c).cos();

        Ok(-self.a * (-self.b * norm.sqrt()).exp() - (cos_sum / 3.0).exp()
            + self.a
            + std::f64::consts::E)
    }

    // Calculated analytically, reference didn't provide gradient
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, EvaluationError> {
        let norm = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2)) / 3.0;
        let sqrt_norm = norm.sqrt();
        let exp_term1 = (-self.b * sqrt_norm).exp();
        let exp_term2 =
            (((x[0] * self.c).cos() + (x[1] * self.c).cos() + (x[2] * self.c).cos()) / 3.0).exp();

        Ok(array![
            (self.a * self.b * x[0] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[0] * self.c).sin() * exp_term2,
            (self.a * self.b * x[1] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[1] * self.c).sin() * exp_term2,
            (self.a * self.b * x[2] / (3.0 * sqrt_norm)) * exp_term1
                + self.c / 3.0 * (x[2] * self.c).sin() * exp_term2,
        ])
    }

    fn variable_bounds(&self) -> Array2<f64> {
        array![
            [-32.768 + 1.0, 32.768 + 1.0],
            [-32.768 + 1.0, 32.768 + 1.0],
            [-32.768 + 1.0, 32.768 + 1.0]
        ]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: f64 = 20.0;
    let b: f64 = 0.2;
    let c: f64 = 2.0 * std::f64::consts::PI;

    let problem: ThreeDAckley = ThreeDAckley::new(a, b, c);

    let params: OQNLPParams = OQNLPParams {
        iterations: 30,
        wait_cycle: 5,
        threshold_factor: 0.2,
        distance_factor: 0.75,
        population_size: 150,
        local_solver_type: LocalSolverType::LBFGS,
        local_solver_config: LBFGSBuilder::default()
            .line_search_params(HagerZhangBuilder::default().build())
            .build(),
        seed: 0,
    };

    let mut oqnlp: OQNLP<ThreeDAckley> = OQNLP::new(problem, params)?;
    let solution_set: SolutionSet = oqnlp.run()?;

    println!("{}", solution_set);

    Ok(())
}
