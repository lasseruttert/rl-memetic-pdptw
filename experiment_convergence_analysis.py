"""
Convergence Analysis for Memetic Algorithm on PDPTW Instances
Runs multiple trials with different seeds and plots convergence over time
"""

from pathlib import Path
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.instance_manager import InstanceManager
from memetic.memetic_algorithm import MemeticSolver


def run_convergence_experiment(
    instance_names: list[str],
    size: int = 100,
    num_runs: int = 5,
    max_time_seconds: int = 60,
    data_dir: str = 'data',
    results_dir: str = 'results/convergence',
    **solver_params
):
    """
    Run memetic algorithm multiple times on specified instances and collect convergence data.

    Args:
        instance_names: List of instance names (e.g., ['lc101', 'lc102'])
        size: Problem size
        num_runs: Number of runs per instance with different seeds
        max_time_seconds: Maximum time per run in seconds
        data_dir: Directory with benchmark data
        results_dir: Directory to save plots and results
        **solver_params: Additional parameters for MemeticSolver
    """
    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Setup
    manager = InstanceManager(base_dir=data_dir)

    # Store all convergence data
    all_convergence_data = {}

    print("="*80)
    print(f"CONVERGENCE ANALYSIS")
    print(f"Instances: {instance_names}")
    print(f"Size: {size}")
    print(f"Runs per instance: {num_runs}")
    print(f"Max time per run: {max_time_seconds}s")
    print("="*80)

    # Run experiments for each instance
    for instance_name in instance_names:
        print(f"\n{'='*80}")
        print(f"Processing instance: {instance_name}")
        print(f"{'='*80}")

        # Load problem
        manager.jump_to(instance_name).jump_to_size(size)
        problem = manager.current()

        print(f"Problem: {problem.num_requests} requests, {problem.num_vehicles} vehicles")

        # Store convergence data for this instance
        convergence_runs = []

        # Run multiple times with different seeds
        for run_idx in range(num_runs):
            seed = 42 + run_idx  # Different seed for each run
            random.seed(seed)
            np.random.seed(seed)

            print(f"\n  Run {run_idx + 1}/{num_runs} (seed={seed})...")

            # Configure solver
            solver = MemeticSolver(
                max_time_seconds=max_time_seconds,
                track_convergence=True,  # Important: enable convergence tracking
                verbose=False,
                **solver_params
            )

            # Solve
            start_time = time.time()
            solution = solver.solve(problem)
            runtime = time.time() - start_time

            # Get convergence data
            convergence_data = solver.convergence.copy()

            print(f"    Final: {solution.num_vehicles_used} vehicles, "
                  f"distance={solution.total_distance:.2f}, "
                  f"time={runtime:.2f}s")

            convergence_runs.append({
                'seed': seed,
                'convergence': convergence_data,
                'final_solution': solution,
                'runtime': runtime
            })

        all_convergence_data[instance_name] = convergence_runs

        # Plot results for this instance immediately
        print(f"\n  Generating plot for {instance_name}...")
        plot_single_instance(instance_name, convergence_runs, results_path, max_time_seconds)

    # Generate combined comparison plot at the end
    if len(all_convergence_data) > 1:
        print(f"\n{'='*80}")
        print("Generating combined comparison plot...")
        print(f"{'='*80}")
        plot_combined_comparison(all_convergence_data, results_path, max_time_seconds)

    # Save raw data
    save_convergence_data(all_convergence_data, results_path)

    print(f"\nResults saved to: {results_path}")
    print("="*80)

    return all_convergence_data


def compute_average_convergence(series_list: list, max_time: int, num_points: int = 200):
    """
    Compute average convergence across multiple runs by interpolating to common time points.

    Args:
        series_list: List of (times, values) tuples from different runs
        max_time: Maximum time to consider
        num_points: Number of points to sample for interpolation

    Returns:
        avg_times: Array of time points
        avg_values: Array of averaged values at those time points
    """
    # Create uniform time grid
    time_grid = np.linspace(0, max_time, num_points)

    # Interpolate each series to the common time grid
    interpolated_series = []

    for times, values in series_list:
        if len(times) == 0:
            continue

        # Sort by time (should already be sorted, but just in case)
        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices]
        values = np.array(values)[sorted_indices]

        # Forward fill: convergence values don't decrease for best fitness
        interpolated_values = np.interp(time_grid, times, values)

        interpolated_series.append(interpolated_values)

    # Compute average across all interpolated series
    if interpolated_series:
        avg_values = np.mean(interpolated_series, axis=0)
        return time_grid, avg_values
    else:
        return np.array([]), np.array([])


def plot_single_instance(instance_name: str, runs: list, results_path: Path, max_time: int):
    """
    Create convergence plots for a single instance (separate files for fitness, vehicles, and avg fitness).

    Args:
        instance_name: Name of the instance
        runs: List of run data for this instance
        results_path: Path to save plots
        max_time: Maximum time in seconds

    Creates three plot files:
        - convergence_{instance_name}_fitness.png: Best fitness over time
        - convergence_{instance_name}_vehicles.png: Number of vehicles over time
        - convergence_{instance_name}_avg_fitness.png: Average population fitness over time
    """
    if not runs:
        return

    colors = cm.viridis(np.linspace(0, 1, len(runs)))

    # Collect all data for averaging
    all_fitness_series = []
    all_vehicle_series = []
    all_avg_fitness_series = []

    for idx, run_data in enumerate(runs):
        convergence = run_data['convergence']
        seed = run_data['seed']

        if convergence:
            times = list(convergence.keys())
            best_fitnesses = [convergence[t]['best_fitness'] for t in times]
            num_vehicles = [convergence[t]['num_vehicles'] for t in times]
            avg_fitnesses = [convergence[t]['avg_fitness'] for t in times]

            # Store for averaging
            all_fitness_series.append((times, best_fitnesses))
            all_vehicle_series.append((times, num_vehicles))
            all_avg_fitness_series.append((times, avg_fitnesses))

    # Plot 1: Best Fitness over Time 
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    for idx, run_data in enumerate(runs):
        convergence = run_data['convergence']
        seed = run_data['seed']

        if convergence:
            times = list(convergence.keys())
            best_fitnesses = [convergence[t]['best_fitness'] for t in times]

            ax1.plot(times, best_fitnesses,
                    label=f'Run {idx+1} (seed={seed})',
                    color=colors[idx],
                    alpha=0.5,
                    linewidth=1.2)

    # Compute and plot average fitness
    if all_fitness_series:
        avg_times, avg_fitness = compute_average_convergence(all_fitness_series, max_time)
        ax1.plot(avg_times, avg_fitness,
                label='Average',
                color='red',
                alpha=1.0,
                linewidth=3.0,
                linestyle='-',
                zorder=10)

    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title(f'{instance_name} - Best Fitness Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_time)

    plt.tight_layout()
    plot_file_fitness = results_path / f'convergence_{instance_name}_fitness.png'
    plt.savefig(plot_file_fitness, dpi=300, bbox_inches='tight')
    print(f"    Saved fitness plot: {plot_file_fitness}")
    plt.close()

    # Plot 2: Number of Vehicles over Time 
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

    for idx, run_data in enumerate(runs):
        convergence = run_data['convergence']
        seed = run_data['seed']

        if convergence:
            times = list(convergence.keys())
            num_vehicles = [convergence[t]['num_vehicles'] for t in times]

            ax2.plot(times, num_vehicles,
                    label=f'Run {idx+1} (seed={seed})',
                    color=colors[idx],
                    alpha=0.5,
                    linewidth=1.2)

    # Compute and plot average vehicles
    if all_vehicle_series:
        avg_times, avg_vehicles = compute_average_convergence(all_vehicle_series, max_time)
        ax2.plot(avg_times, avg_vehicles,
                label='Average',
                color='red',
                alpha=1.0,
                linewidth=3.0,
                linestyle='-',
                zorder=10)

    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Number of Vehicles', fontsize=12)
    ax2.set_title(f'{instance_name} - Vehicles Used', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_time)

    plt.tight_layout()
    plot_file_vehicles = results_path / f'convergence_{instance_name}_vehicles.png'
    plt.savefig(plot_file_vehicles, dpi=300, bbox_inches='tight')
    print(f"    Saved vehicles plot: {plot_file_vehicles}")
    plt.close()

    # Plot 3: Average Fitness over Time (population average)
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

    for idx, run_data in enumerate(runs):
        convergence = run_data['convergence']
        seed = run_data['seed']

        if convergence:
            times = list(convergence.keys())
            avg_fitnesses = [convergence[t]['avg_fitness'] for t in times]

            ax3.plot(times, avg_fitnesses,
                    label=f'Run {idx+1} (seed={seed})',
                    color=colors[idx],
                    alpha=0.5,
                    linewidth=1.2)

    # Compute and plot average of avg fitness
    if all_avg_fitness_series:
        avg_times, avg_avg_fitness = compute_average_convergence(all_avg_fitness_series, max_time)
        ax3.plot(avg_times, avg_avg_fitness,
                label='Average',
                color='red',
                alpha=1.0,
                linewidth=3.0,
                linestyle='-',
                zorder=10)

    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Average Fitness (Population)', fontsize=12)
    ax3.set_title(f'{instance_name} - Average Population Fitness Convergence', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_time)

    plt.tight_layout()
    plot_file_avg_fitness = results_path / f'convergence_{instance_name}_avg_fitness.png'
    plt.savefig(plot_file_avg_fitness, dpi=300, bbox_inches='tight')
    print(f"    Saved avg fitness plot: {plot_file_avg_fitness}")
    plt.close()


def plot_combined_comparison(all_data: dict, results_path: Path, max_time: int):
    """
    Create combined comparison plot across all instances.

    Args:
        all_data: Dictionary with convergence data per instance
        results_path: Path to save plots
        max_time: Maximum time in seconds
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    instance_colors = cm.Set1(np.linspace(0, 1, len(all_data)))

    for inst_idx, (instance_name, runs) in enumerate(all_data.items()):
        all_fitness_series = []

        # Plot all runs for this instance with same color (lighter)
        for run_data in runs:
            convergence = run_data['convergence']
            if convergence:
                times = list(convergence.keys())
                fitnesses = [convergence[t]['best_fitness'] for t in times]

                ax.plot(times, fitnesses,
                       color=instance_colors[inst_idx],
                       alpha=0.2,
                       linewidth=1)

                # Store for averaging
                all_fitness_series.append((times, fitnesses))

        # Compute and plot average for this instance (bold)
        if all_fitness_series:
            avg_times, avg_fitness = compute_average_convergence(all_fitness_series, max_time)
            ax.plot(avg_times, avg_fitness,
                   color=instance_colors[inst_idx],
                   label=f'{instance_name} (avg)',
                   linewidth=3.0,
                   alpha=0.9,
                   linestyle='-',
                   zorder=10)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Convergence Comparison Across Instances', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)

    plt.tight_layout()
    plot_file = results_path / 'convergence_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot: {plot_file}")
    plt.close()


def save_convergence_data(all_data: dict, results_path: Path):
    """Save convergence data to text file for later analysis."""
    output_file = results_path / 'convergence_data.txt'

    with open(output_file, 'w') as f:
        f.write("CONVERGENCE DATA\n")
        f.write("="*80 + "\n\n")

        for instance_name, runs in all_data.items():
            f.write(f"\nInstance: {instance_name}\n")
            f.write("-"*80 + "\n")

            for idx, run_data in enumerate(runs):
                seed = run_data['seed']
                solution = run_data['final_solution']
                runtime = run_data['runtime']

                f.write(f"\n  Run {idx+1} (seed={seed}):\n")
                f.write(f"    Final vehicles: {solution.num_vehicles_used}\n")
                f.write(f"    Final distance: {solution.total_distance:.2f}\n")
                f.write(f"    Feasible: {solution.is_feasible}\n")
                f.write(f"    Runtime: {runtime:.2f}s\n")

                # Write convergence points
                convergence = run_data['convergence']
                if convergence:
                    f.write(f"    Convergence points: {len(convergence)}\n")
                    f.write(f"    Time -> (Fitness, Vehicles)\n")
                    for t in sorted(convergence.keys()):
                        data = convergence[t]
                        f.write(f"      {t:.2f}s -> ({data['best_fitness']:.2f}, {data['num_vehicles']})\n")

    print(f"  Saved data: {output_file}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = 'G:/Meine Ablage/rl-memetic-pdptw/data'
    RESULTS_DIR = 'G:/Meine Ablage/rl-memetic-pdptw/results/convergence'

    # Select a few instances to test
    INSTANCES = ['lc101', 'lc109', 'lc201', 'lc208', 'lr102', 'lr112', 'lr201', 'lr211', 'lrc101', 'lrc201']
    SIZE = 100
    NUM_RUNS = 5
    MAX_TIME = 30

    # Solver configuration
    SOLVER_CONFIG = {
        'ensure_diversity_interval': 3,
        'evaluation_interval': 10,
        'track_history': False
    }

    # Run experiment
    convergence_data = run_convergence_experiment(
        instance_names=INSTANCES,
        size=SIZE,
        num_runs=NUM_RUNS,
        max_time_seconds=MAX_TIME,
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        **SOLVER_CONFIG
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
