/// Catalytic Cracking of Gas Oil - Parameter Estimation Problem
///
/// This example solves a parameter estimation problem for the catalytic cracking of gas
/// oil into gas and other byproducts. The problem involves estimating three reaction
/// coefficients (θ₁, θ₂, θ₃) for a system of nonlinear ordinary
/// differential equations (ODEs).
///
/// The example comes from COPS 2.0: Large-Scale Optimization Problems (Problem #21)
/// and it is based on the GAMS model (GASOIL.gms).
///
/// ## Mathematical Model
///
/// The catalytic cracking process is described by the following ODE system:
///
/// dy₁/dt = -(θ₁ + θ₃) × y₁²
/// dy₂/dt = θ₁ × y₁² - θ₂ × y₂
///
/// Where:
/// - y₁(t): concentration of gas oil at time t
/// - y₂(t): concentration of gas and byproducts at time t
/// - θ₁, θ₂, θ₃: reaction rate coefficients (parameters to estimate)
///
/// Initial Conditions: y₁(0) = 1.0, y₂(0) = 0.0
///
/// ## Parameter Estimation Problem
///
/// Given experimental observations at 21 time points from t=0.000 to t=0.950,
/// we minimize the sum of squared residuals between the ODE solution and
/// experimental data:
///
/// minimize: Σᵢ Σⱼ (yⱼ(tᵢ; θ) - zᵢⱼ)²
///
/// Where:
/// - yⱼ(tᵢ; θ): ODE solution for variable j at time tᵢ with parameters θ
/// - zᵢⱼ: experimental observation for variable j at time tᵢ
///
/// ## Notes on the implementation
///
/// The GAMS model for this problem is solved using a collocation method
/// with partition intervals. In this Rust implementation, we solve the ODEs
/// using a 4th-order Runge-Kutta method with a small step size.
///
/// ## References
///
/// - Dolan, E D, and More, J J, "Benchmarking Optimization Software with COPS."
///   Tech. rep., Mathematics and Computer Science Division, 2000.
/// - Tjoa, I B, and Biegler, L T, "Simultaneous Solution and Optimization Strategies
///   for Parameter Estimation of Differential-Algebraic Equations Systems."
///   Ind. Eng. Chem. Res. 30 (1991), 376-385.
use globalsearch::problem::Problem;
use globalsearch::{
    oqnlp::OQNLP,
    types::{EvaluationError, OQNLPParams, SolutionSet},
};
use ndarray::{array, Array1, Array2};

/// Experimental data structure holding time points and corresponding observations
#[derive(Debug, Clone)]
struct ExperimentalData {
    /// Time points at which observations were made (21 points from 0.000 to 0.950)
    pub times: Vec<f64>,
    /// Observations for gas oil concentration (y₁) at each time point
    pub y1_observations: Vec<f64>,
    /// Observations for gas/byproducts concentration (y₂) at each time point  
    pub y2_observations: Vec<f64>,
}

impl ExperimentalData {
    /// Experimental data from the COPS benchmarking problem
    fn new() -> Self {
        #[allow(clippy::approx_constant)]
        let times = vec![
            0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.300,
            0.350, 0.400, 0.450, 0.500, 0.550, 0.650, 0.750, 0.850, 0.950,
        ];

        #[allow(clippy::approx_constant)]
        let y1_observations = vec![
            1.0000, 0.8105, 0.6208, 0.5258, 0.4345, 0.3903, 0.3342, 0.3034, 0.2735, 0.2405, 0.2283,
            0.2071, 0.1669, 0.1530, 0.1339, 0.1265, 0.1200, 0.0990, 0.0870, 0.0770, 0.0690,
        ];

        #[allow(clippy::approx_constant)]
        let y2_observations = vec![
            0.0000, 0.2000, 0.2886, 0.3010, 0.3215, 0.3123, 0.2716, 0.2551, 0.2258, 0.1959, 0.1789,
            0.1457, 0.1198, 0.0909, 0.0719, 0.0561, 0.0460, 0.0280, 0.0190, 0.0140, 0.0100,
        ];

        ExperimentalData {
            times,
            y1_observations,
            y2_observations,
        }
    }
}

/// ODE solver state representing the concentration values at a specific time
#[derive(Debug, Clone, Copy)]
struct ODEState {
    /// Gas oil concentration
    pub y1: f64,
    /// Gas and byproducts concentration  
    pub y2: f64,
}

impl ODEState {
    /// Creates new ODE state with given concentrations
    fn new(y1: f64, y2: f64) -> Self {
        ODEState { y1, y2 }
    }
}

/// Simple 4th-order Runge-Kutta ODE solver for the catalytic cracking system
struct ODESolver {
    /// Current solver state
    state: ODEState,
    /// Current time
    time: f64,
    /// Integration step size
    step_size: f64,
}

impl ODESolver {
    /// Creates new ODE solver with initial conditions
    fn new(initial_state: ODEState, step_size: f64) -> Self {
        ODESolver {
            state: initial_state,
            time: 0.0,
            step_size,
        }
    }

    /// Computes the right-hand side of the ODE system
    ///
    /// Returns (dy₁/dt, dy₂/dt) given current state and parameters
    fn rhs(&self, state: ODEState, theta: &[f64]) -> (f64, f64) {
        let theta1 = theta[0];
        let theta2 = theta[1];
        let theta3 = theta[2];

        let y1_squared = state.y1 * state.y1;

        let dy1_dt = -(theta1 + theta3) * y1_squared;
        let dy2_dt = theta1 * y1_squared - theta2 * state.y2;

        (dy1_dt, dy2_dt)
    }

    /// Performs one step of 4th-order Runge-Kutta integration with default step size
    fn step(&mut self, theta: &[f64]) {
        self.step_with_h(self.step_size, theta);
    }

    /// Internal method: performs one step of 4th-order Runge-Kutta integration with specified step size
    /// This avoids mutating the struct's step_size field when adjusting for overshoot
    fn step_with_h(&mut self, h: f64, theta: &[f64]) {
        let y = self.state;

        // k1 = f(t, y)
        let (k1_y1, k1_y2) = self.rhs(y, theta);

        // k2 = f(t + h/2, y + h*k1/2)
        let y2_state = ODEState::new(y.y1 + h * k1_y1 * 0.5, y.y2 + h * k1_y2 * 0.5);
        let (k2_y1, k2_y2) = self.rhs(y2_state, theta);

        // k3 = f(t + h/2, y + h*k2/2)
        let y3_state = ODEState::new(y.y1 + h * k2_y1 * 0.5, y.y2 + h * k2_y2 * 0.5);
        let (k3_y1, k3_y2) = self.rhs(y3_state, theta);

        // k4 = f(t + h, y + h*k3)
        let y4_state = ODEState::new(y.y1 + h * k3_y1, y.y2 + h * k3_y2);
        let (k4_y1, k4_y2) = self.rhs(y4_state, theta);

        // y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.state.y1 += h / 6.0 * (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1);
        self.state.y2 += h / 6.0 * (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2);

        self.time += h;
    }

    /// Solves ODE system up to specified time points
    ///
    /// Returns vectors of (y₁, y₂) values at the requested time points
    fn solve_to_times(
        &mut self,
        target_times: &[f64],
        theta: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut y1_results = Vec::with_capacity(target_times.len());
        let mut y2_results = Vec::with_capacity(target_times.len());

        let mut time_index = 0;

        // Add initial condition
        if !target_times.is_empty() && target_times[0] == 0.0 {
            y1_results.push(self.state.y1);
            y2_results.push(self.state.y2);
            time_index = 1;
        }

        while time_index < target_times.len() {
            let target_time = target_times[time_index];

            // Integrate until we reach or exceed the target time
            while self.time < target_time {
                // Adjust step size if we would overshoot
                if self.time + self.step_size > target_time {
                    let original_step = self.step_size;
                    self.step_size = target_time - self.time;
                    self.step(theta);
                    self.step_size = original_step;
                    break;
                } else {
                    self.step(theta);
                }
            }

            y1_results.push(self.state.y1);
            y2_results.push(self.state.y2);
            time_index += 1;
        }

        Ok((y1_results, y2_results))
    }
}

#[derive(Debug, Clone)]
pub struct CatalyticCracking {
    /// Experimental data for parameter estimation
    experimental_data: ExperimentalData,
    /// ODE integration step size
    integration_step_size: f64,
}

impl Default for CatalyticCracking {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalyticCracking {
    /// Creates a new catalytic cracking parameter estimation problem
    pub fn new() -> Self {
        CatalyticCracking {
            experimental_data: ExperimentalData::new(),
            integration_step_size: 0.001, // Small step size for accurate integration
        }
    }

    /// Computes the sum of squared residuals between ODE solution and experimental data
    fn residual_sum_of_squares(&self, theta: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Validate parameters - must be positive for physical meaning
        for &param in theta.iter() {
            if param <= 0.0 {
                return Err(EvaluationError::InvalidInput(
                    "All reaction rate coefficients must be positive".to_string(),
                ));
            }
        }

        // Convert to slice for ODE solver
        let theta_slice = theta.as_slice().unwrap();

        // Set up ODE solver with initial conditions
        let initial_state = ODEState::new(1.0, 0.0); // y₁(0) = 1, y₂(0) = 0
        let mut solver = ODESolver::new(initial_state, self.integration_step_size);

        // Solve ODE system at experimental time points
        let (y1_solution, y2_solution) = solver
            .solve_to_times(&self.experimental_data.times, theta_slice)
            .map_err(|_| EvaluationError::ObjectiveFunctionEvaluationFailed)?;

        // Calculate sum of squared residuals
        let mut ssr = 0.0;

        for i in 0..self.experimental_data.times.len() {
            // Residuals for y₁ (gas oil concentration)
            let y1_residual = y1_solution[i] - self.experimental_data.y1_observations[i];
            ssr += y1_residual * y1_residual;

            // Residuals for y₂ (gas/byproducts concentration)
            let y2_residual = y2_solution[i] - self.experimental_data.y2_observations[i];
            ssr += y2_residual * y2_residual;
        }

        Ok(ssr)
    }
}

impl Problem for CatalyticCracking {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        // Validate input dimension - must have exactly 3 parameters
        if x.len() != 3 {
            return Err(EvaluationError::InvalidInput(format!(
                "Expected 3 parameters (θ₁, θ₂, θ₃), got {}",
                x.len()
            )));
        }

        self.residual_sum_of_squares(x)
    }

    fn variable_bounds(&self) -> Array2<f64> {
        // All parameters must be positive, with upper bounds based on typical values
        // for catalytic cracking reaction rates
        array![
            [0.001, 100.0], // θ₁: primary cracking rate
            [0.001, 100.0], // θ₂: secondary reaction rate
            [0.001, 100.0], // θ₃: additional cracking rate
        ]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Catalytic Cracking of Gas Oil - Parameter Estimation");
    println!("===================================================");
    println!();
    println!("Problem Description:");
    println!("- Estimating reaction rate coefficients for catalytic cracking");
    println!("- ODE System: dy₁/dt = -(θ₁ + θ₃)·y₁², dy₂/dt = θ₁·y₁² - θ₂·y₂");
    println!("- Initial conditions: y₁(0) = 1.0, y₂(0) = 0.0");
    println!("- Parameters: θ₁, θ₂, θ₃ (reaction rate coefficients)");
    println!("- Data: 21 experimental observations from t=0.000 to t=0.950");
    println!("- Objective: Minimize sum of squared residuals");
    println!();

    let problem = CatalyticCracking::new();

    // Configure optimization parameters
    let params: OQNLPParams = OQNLPParams {
        iterations: 100,
        population_size: 300,
        wait_cycle: 15,
        threshold_factor: 0.15,
        distance_factor: 0.6,
        seed: 0,
        ..Default::default()
    };

    let mut oqnlp: OQNLP<CatalyticCracking> = OQNLP::new(problem.clone(), params)?.verbose();

    println!("Starting optimization...");
    println!();

    let solution_set: SolutionSet = oqnlp.run()?;

    println!();
    println!("Optimization Results:");
    println!("{}", solution_set);

    // Analyze the best solution in detail
    if let Some(best_solution) = solution_set.solutions.first() {
        let theta_opt = &best_solution.point;
        let ssr_min = best_solution.objective;

        println!();
        println!("Best Parameter Estimates:");
        println!("=======================");
        println!("θ₁ (primary cracking rate):     {:.6}", theta_opt[0]);
        println!("θ₂ (secondary reaction rate):   {:.6}", theta_opt[1]);
        println!("θ₃ (additional cracking rate):  {:.6}", theta_opt[2]);
        println!();
        println!("Sum of Squared Residuals: {:.8}", ssr_min);
        println!(
            "Root Mean Square Error:   {:.6}",
            (ssr_min / (21.0 * 2.0)).sqrt()
        );

        // Validate solution by comparing with experimental data
        println!();
        println!("Model Validation:");
        println!("===============");

        // Solve ODE with optimal parameters
        let initial_state = ODEState::new(1.0, 0.0);
        let mut solver = ODESolver::new(initial_state, problem.integration_step_size);
        let theta_slice = theta_opt.as_slice().unwrap();

        match solver.solve_to_times(&problem.experimental_data.times, theta_slice) {
            Ok((y1_solution, y2_solution)) => {
                println!("Time\tExp_y₁\tMod_y₁\tError₁\tExp_y₂\tMod_y₂\tError₂");
                println!("----\t-----\t-----\t-----\t-----\t-----\t-----");

                let mut max_error_y1 = 0.0;
                let mut max_error_y2 = 0.0;

                for i in 0..problem.experimental_data.times.len() {
                    let t = problem.experimental_data.times[i];
                    let exp_y1 = problem.experimental_data.y1_observations[i];
                    let mod_y1 = y1_solution[i];
                    let err_y1 = (mod_y1 - exp_y1).abs();

                    let exp_y2 = problem.experimental_data.y2_observations[i];
                    let mod_y2 = y2_solution[i];
                    let err_y2 = (mod_y2 - exp_y2).abs();

                    max_error_y1 = f64::max(max_error_y1, err_y1);
                    max_error_y2 = f64::max(max_error_y2, err_y2);

                    if i % 3 == 0 || i < 3 || i >= problem.experimental_data.times.len() - 3 {
                        println!(
                            "{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                            t, exp_y1, mod_y1, err_y1, exp_y2, mod_y2, err_y2
                        );
                    }
                }

                println!("...\t(intermediate points omitted)\t...");
                println!();
                println!("Maximum Errors:");
                println!("- Gas oil (y₁):      {:.6}", max_error_y1);
                println!("- Gas/byproducts (y₂): {:.6}", max_error_y2);

                let conversion_rate = theta_opt[0] / (theta_opt[0] + theta_opt[2]);
                println!(
                    "Primary conversion efficiency: {:.1}%",
                    conversion_rate * 100.0
                );

                if theta_opt[1] > 0.0 {
                    let residence_time = 1.0 / theta_opt[1];
                    println!(
                        "Gas/byproducts residence time: {:.3} time units",
                        residence_time
                    );
                }
            }
            Err(e) => {
                println!("Error in solution validation: {}", e);
            }
        }

        // Compare with GAMS reference solution
        println!();
        println!("GAMS Reference Solution Comparison:");
        println!();

        let gams_theta = [11.847, 8.345, 1.001]; // GAMS reference: np1, np2, np3

        println!("Parameter\tGAMS\t\tRust\t\tDifference\tRel. Error (%)");
        println!("---------\t----\t\t----\t\t----------\t--------------");

        for i in 0..gams_theta.len() {
            let param_names = ["θ₁", "θ₂", "θ₃"];
            let gams_val = gams_theta[i];
            let rust_val = theta_opt[i];
            let diff = rust_val - gams_val;
            let rel_error = (diff / gams_val) * 100.0;

            println!(
                "{}\t\t{:.3}\t\t{:.3}\t\t{:.6}\t{:.4}",
                param_names[i], gams_val, rust_val, diff, rel_error
            );
        }

        // Calculate GAMS solution SSR for comparison
        let gams_theta_array = Array1::from(gams_theta.to_vec());
        match problem.residual_sum_of_squares(&gams_theta_array) {
            Ok(gams_ssr) => {
                let rust_ssr = best_solution.objective;
                let ssr_improvement = ((gams_ssr - rust_ssr) / gams_ssr) * 100.0;

                println!();
                println!("Objective Function Comparison:");
                println!("GAMS SSR:     {:.8}", gams_ssr);
                println!("Rust SSR:     {:.8}", rust_ssr);
                println!(
                    "Improvement:  {:.6}% ({} SSR)",
                    ssr_improvement.abs(),
                    if rust_ssr < gams_ssr {
                        "lower"
                    } else {
                        "higher"
                    }
                );

                // Statistical significance test
                let ssr_ratio = rust_ssr / gams_ssr;
                if ssr_ratio < 0.999 {
                    println!("Rust solution is statistically better than GAMS");
                } else if ssr_ratio > 1.001 {
                    println!("GAMS solution is better than Rust");
                } else {
                    println!("Solutions are statistically equivalent");
                }
            }
            Err(e) => {
                println!("Error evaluating GAMS solution: {:?}", e);
            }
        }
    }

    Ok(())
}
