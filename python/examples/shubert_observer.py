"""
Shubert Function Optimization with Observers

This example demonstrates Observer usage with the Shubert function, a complex
multimodal function with many global minima.

Function definition: f(x₁, x₂) = [Σᵢ₌₁⁵ i·cos((i+1)x₁ + i)] · [Σᵢ₌₁⁵ i·cos((i+1)x₂ + i)]

Properties:
- Bounds: [-10, 10] for each dimension
- Global minimum: f(x*) = -186.7309
- Has 18 global minima
- Highly multimodal with 760 local minima

This example demonstrates:
- Solving a challenging 2D multimodal problem
- Using observers to track both Stage 1 (scatter search) and Stage 2 (local refinement)
- Finding multiple global minima
- Real-time progress monitoring

Please note that we use the definition of the Shubert function as given in the reference by Molga, M., & Smutnicki, C.
X. Wang, S. -s. Wang and L. Xiao solved a modified version of the Shubert function that has two additional terms.

References:
Molga, M., & Smutnicki, C. Test functions for optimization needs (April 3, 2005), pp. 36-38.
Retrieved October 2025, from https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf

X. Wang, S. -s. Wang and L. Xiao, "Solving Shubert Function Optimization Problem by Using Thermodynamics Evolutionary Algorithm,"
2010 International Conference on Biomedical Engineering and Computer Science, Wuhan, China, 2010, pp. 1-4,
doi: 10.1109/ICBECS.2010.5462350.
"""

import numpy as np
import pyglobalsearch as gs


def shubert_objective(x):
    """
    Shubert function: A highly multimodal 2D function with 18 global minima.

    f(x₁, x₂) = [Σᵢ₌₁⁵ i·cos((i+1)x₁ + i)] · [Σᵢ₌₁⁵ i·cos((i+1)x₂ + i)]

    Global minimum: f(x*) = -186.7309
    Has 18 global minima and 760 local minima total
    """
    x1, x2 = x[0], x[1]

    # First sum: Σᵢ₌₁⁵ i·cos((i+1)x₁ + i)
    sum1 = sum(i * np.cos((i + 1) * x1 + i) for i in range(1, 6))

    # Second sum: Σᵢ₌₁⁵ i·cos((i+1)x₂ + i)
    sum2 = sum(i * np.cos((i + 1) * x2 + i) for i in range(1, 6))

    return sum1 * sum2


def shubert_bounds():
    """
    Variable bounds for the Shubert function.
    """
    return np.array([[-10.0, 10.0], [-10.0, 10.0]])


def main():
    """Main optimization function."""
    print("Shubert Function Optimization with Observer\n")

    print("Global minimum: f(x*) = -186.7309")
    print("Number of global minima: 18")
    print("Bounds: [-10.0, 10.0] for each dimension\n")

    # Create the problem
    problem = gs.PyProblem(shubert_objective, shubert_bounds)

    # Optimization parameters
    params = gs.PyOQNLPParams(
        iterations=4500,
        wait_cycle=10,
        threshold_factor=0.3,
        distance_factor=0.8,
        population_size=8000,
    )

    # Create COBYLA solver configuration
    cobyla_config = gs.builders.cobyla(max_iter=75, step_size=1.0)

    # Create observer with callbacks for both Stage 1 and Stage 2 tracking
    observer = gs.observers.Observer()
    observer.with_stage1_tracking()
    observer.with_stage2_tracking()
    observer.with_timing()
    observer.with_default_callback()
    observer.with_callback_frequency(1)

    print("Running optimization...\n")

    # Run optimization with observer
    result = gs.optimize(
        problem=problem,
        params=params,
        seed=0,
        local_solver_config=cobyla_config,
        observer=observer,
        parallel=True,  # Enable parallel processing
    )

    print("\nOptimization Complete")
    print(result)

    # Print final observer statistics
    stage1_final = observer.stage1_final()
    stage2_final = observer.stage2()

    if stage1_final:
        print("\nStage 1 Final Statistics:")
        print(f"  Function evaluations: {stage1_final.function_evaluations}")
        print(f"  Reference set size: {stage1_final.reference_set_size}")
        print(f"  Trial points generated: {stage1_final.trial_points_generated}")
        print(f"  Best objective found: {stage1_final.best_objective:.6f}")
        if stage1_final.total_time:
            print(f"  Total time: {stage1_final.total_time:.3f}s")

    if stage2_final:
        print("\nStage 2 Final Statistics:")
        print(f"  Function evaluations: {stage2_final.function_evaluations}")
        print(f"  Solution set size: {stage2_final.solution_set_size}")
        print(f"  Current iteration: {stage2_final.current_iteration}")
        print(f"  Best objective found: {stage2_final.best_objective:.6f}")
        print(f"  Threshold value: {stage2_final.threshold_value:.6f}")
        print(f"  Local solver calls: {stage2_final.local_solver_calls}")
        print(f"  Improved local calls: {stage2_final.improved_local_calls}")
        if stage2_final.total_time:
            print(f"  Total time: {stage2_final.total_time:.3f}s")


if __name__ == "__main__":
    main()
