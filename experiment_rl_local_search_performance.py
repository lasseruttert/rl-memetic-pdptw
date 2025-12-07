"""
RL Local Search Performance Experiment

Tests trained RL local search across all benchmark instances with both random
and greedy initial solutions. Tracks comprehensive metrics and saves detailed results.
"""

from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions

from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch
from memetic.solution_generators.random_generator import RandomGenerator
from memetic.solution_generators.greedy_generator import GreedyGenerator
from memetic.fitness.fitness import fitness

import time
import json
import csv
import random
import numpy as np
import copy
from pathlib import Path
from typing import Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# Problem sizes to test
PROBLEM_SIZES = [100]

# Number of runs per initialization type per instance
NUM_RUNS = 10

# Path to trained RL local search model
MODEL_PATH = "models/rl_local_search_dqn_100_greedy_binary_100_set3_final.pt"

# Results directory
RESULTS_DIR = "results/rl_local_search_performance"

# Random seed for reproducibility
SEED = 42

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def compute_averaged_metrics(run_results: List[Dict]) -> Dict:
    """Compute averaged metrics across multiple runs.

    Args:
        run_results: List of result dictionaries from multiple runs

    Returns:
        Dictionary with mean and std for all metrics
    """
    if not run_results:
        return {}

    # Extract scalar metrics
    initial_fitnesses = [r['initial_fitness'] for r in run_results]
    final_fitnesses = [r['final_fitness'] for r in run_results]
    improvements = [r['improvement_percent'] for r in run_results]
    iterations = [r['total_iterations'] for r in run_results]
    times = [r['total_time'] for r in run_results]
    acceptance_rates = [r['acceptance_rate'] for r in run_results]

    # Route metrics
    num_routes = [r['avg_num_routes'] for r in run_results]
    avg_route_distances = [r['avg_route_distance'] for r in run_results]

    # Operator statistics
    operator_uses = {}
    operator_success_rates = {}

    # Aggregate operator stats
    for run in run_results:
        for op_idx, stats in run['operator_statistics'].items():
            if op_idx not in operator_uses:
                operator_uses[op_idx] = []
                operator_success_rates[op_idx] = []
            operator_uses[op_idx].append(stats['uses'])
            operator_success_rates[op_idx].append(stats['success_rate'])

    # Compute operator averages
    operator_avg_uses = {op_idx: np.mean(uses) for op_idx, uses in operator_uses.items()}
    operator_avg_success = {op_idx: np.mean(rates) for op_idx, rates in operator_success_rates.items()}

    averaged = {
        'num_runs': len(run_results),
        'mean_initial_fitness': float(np.mean(initial_fitnesses)),
        'std_initial_fitness': float(np.std(initial_fitnesses)),
        'mean_final_fitness': float(np.mean(final_fitnesses)),
        'std_final_fitness': float(np.std(final_fitnesses)),
        'mean_improvement_percent': float(np.mean(improvements)),
        'std_improvement_percent': float(np.std(improvements)),
        'mean_iterations': float(np.mean(iterations)),
        'std_iterations': float(np.std(iterations)),
        'mean_time_seconds': float(np.mean(times)),
        'std_time_seconds': float(np.std(times)),
        'mean_acceptance_rate': float(np.mean(acceptance_rates)),
        'std_acceptance_rate': float(np.std(acceptance_rates)),
        'mean_num_routes': float(np.mean(num_routes)),
        'std_num_routes': float(np.std(num_routes)),
        'mean_avg_route_distance': float(np.mean(avg_route_distances)),
        'std_avg_route_distance': float(np.std(avg_route_distances)),
        'operator_avg_uses': operator_avg_uses,
        'operator_avg_success_rates': operator_avg_success
    }

    return averaged

def analyze_run_history(run_history: Dict) -> Dict:
    """Analyze a single run's iteration history to extract summary statistics.

    Args:
        run_history: Dictionary mapping iteration -> metrics

    Returns:
        Dictionary with analyzed statistics
    """
    if not run_history:
        return {}

    iterations = sorted(run_history.keys())

    # Extract sequences
    fitness_trajectory = [run_history[i]['fitness'] for i in iterations]
    operator_sequence = [run_history[i]['action'] for i in iterations]
    acceptance_sequence = [run_history[i]['accepted'] for i in iterations]

    # Compute statistics
    num_accepted = sum(acceptance_sequence)
    acceptance_rate = num_accepted / len(acceptance_sequence) if acceptance_sequence else 0.0

    # Route metrics
    num_routes = [run_history[i]['num_routes'] for i in iterations]
    avg_route_distances = [run_history[i]['avg_route_distance'] for i in iterations]

    # Operator statistics
    operator_stats = {}
    for i in iterations:
        action = run_history[i]['action']
        if action not in operator_stats:
            operator_stats[action] = {'uses': 0, 'successes': 0, 'acceptances': 0}

        operator_stats[action]['uses'] += 1
        if run_history[i]['fitness_improvement'] > 0:
            operator_stats[action]['successes'] += 1
        if run_history[i]['accepted']:
            operator_stats[action]['acceptances'] += 1

    # Compute rates
    for op_idx in operator_stats:
        uses = operator_stats[op_idx]['uses']
        operator_stats[op_idx]['success_rate'] = operator_stats[op_idx]['successes'] / uses if uses > 0 else 0.0
        operator_stats[op_idx]['acceptance_rate'] = operator_stats[op_idx]['acceptances'] / uses if uses > 0 else 0.0

    initial_fitness = fitness_trajectory[0] if fitness_trajectory else 0.0
    final_fitness = fitness_trajectory[-1] if fitness_trajectory else 0.0
    improvement_percent = ((initial_fitness - final_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0.0

    return {
        'initial_fitness': initial_fitness,
        'final_fitness': final_fitness,
        'improvement_percent': improvement_percent,
        'total_iterations': len(iterations),
        'total_time': run_history[iterations[-1]]['time'] if iterations else 0.0,
        'acceptance_rate': acceptance_rate,
        'avg_num_routes': float(np.mean(num_routes)) if num_routes else 0.0,
        'avg_route_distance': float(np.mean(avg_route_distances)) if avg_route_distances else 0.0,
        'operator_sequence': operator_sequence,
        'operator_statistics': operator_stats
    }

def save_json(data: Dict, filepath: str):
    """Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to output file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def run_experiment():
    """Run RL local search performance experiment.

    Tests trained RL local search on all instances with random and greedy initialization.
    Saves comprehensive tracking data and statistics.
    """
    print("=" * 80)
    print("RL LOCAL SEARCH PERFORMANCE EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Runs per initialization type: {NUM_RUNS}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 80)

    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)

    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load trained model
    print(f"\nLoading trained model from {MODEL_PATH}...")
    rl_local_search = RLLocalSearch.load_from_checkpoint(MODEL_PATH, verbose=True)

    # Enable tracking
    rl_local_search.tracking = True
    print("Tracking enabled")

    # Initialize instance managers
    li_lim_manager = LiLimInstanceManager()
    mendeley_manager = MendeleyInstanceManager()
    best_known_solutions = BestKnownSolutions()

    # Get all instances
    all_instances = []
    for size in PROBLEM_SIZES:
        all_instances.extend(li_lim_manager.get_all(size=size))
        all_instances.extend(mendeley_manager.get_all(size=size))

    print(f"Total instances: {len(all_instances)}\n")

    # Initialize solution generators
    random_generator = RandomGenerator()
    greedy_generator = GreedyGenerator()

    # Store all results for summary
    all_instance_results = {}

    # Process each instance
    for inst_idx, instance in enumerate(all_instances):
        instance_name = Path(instance.name).stem if ('/' in instance.name or '\\' in instance.name) else instance.name

        print(f"\n[{inst_idx+1}/{len(all_instances)}] Processing {instance_name}")
        print("-" * 80)

        instance_start_time = time.time()

        # Results for this instance
        instance_results = {
            'instance_name': instance_name,
            'random_initialization': {},
            'greedy_initialization': {}
        }

        # Process each initialization type
        for init_type, generator in [('random', random_generator), ('greedy', greedy_generator)]:
            print(f"\n  {init_type.capitalize()} initialization:")

            run_results = []
            best_run_idx = -1
            best_run_fitness = float('inf')
            best_run_history = None

            # Run multiple times
            for run_idx in range(NUM_RUNS):
                print(f"    Run {run_idx+1}/{NUM_RUNS}...", end=" ")

                # Generate initial solution
                initial_solution = generator.generate(instance, 1)[0]

                # Apply RL local search with tracking
                result = rl_local_search.search(
                    problem=instance,
                    solution=initial_solution,
                    epsilon=0.0,  # Greedy evaluation
                    deterministic_rng=True,
                    base_seed=SEED + inst_idx * 1000 + run_idx
                )

                # Unpack result (tracking enabled returns 3 values)
                best_solution, best_fitness, run_history = result

                # Analyze run
                run_stats = analyze_run_history(run_history)
                run_results.append(run_stats)

                # Track best run
                if run_stats['final_fitness'] < best_run_fitness:
                    best_run_fitness = run_stats['final_fitness']
                    best_run_idx = run_idx
                    best_run_history = copy.deepcopy(run_history)

                print(f"Fitness: {run_stats['initial_fitness']:.2f} -> {run_stats['final_fitness']:.2f} "
                      f"({run_stats['improvement_percent']:.1f}%)")

            # Compute averaged metrics
            averaged_metrics = compute_averaged_metrics(run_results)

            # Store results
            instance_results[f'{init_type}_initialization'] = {
                'averaged_metrics': averaged_metrics
            }

            # Save best run history to separate file
            best_run_data = {
                'instance_name': instance_name,
                'initialization_type': init_type,
                'run_index': best_run_idx,
                'summary': run_results[best_run_idx],
                'iteration_history': best_run_history
            }

            best_run_filename = f"rl_local_search_instance_{instance_name}_{init_type}_init_best_run_full_history.json"
            save_json(best_run_data, Path(RESULTS_DIR) / best_run_filename)

            print(f"    Best run: #{best_run_idx+1} with final fitness {best_run_fitness:.2f}")
            print(f"    Saved to: {best_run_filename}")

        # Save averaged results for this instance
        averaged_filename = f"rl_local_search_instance_{instance_name}_averaged_across_runs.json"
        save_json(instance_results, Path(RESULTS_DIR) / averaged_filename)

        # Store for summary
        all_instance_results[instance_name] = instance_results

        instance_elapsed = time.time() - instance_start_time
        print(f"\n  Instance completed in {instance_elapsed:.2f} seconds")

    # Save overall summary
    summary_data = {
        'experiment': 'RL Local Search Performance',
        'model_path': MODEL_PATH,
        'num_runs_per_init': NUM_RUNS,
        'problem_sizes': PROBLEM_SIZES,
        'total_instances': len(all_instances),
        'instances': all_instance_results
    }

    summary_filename = "rl_local_search_all_instances_summary.json"
    save_json(summary_data, Path(RESULTS_DIR) / summary_filename)
    print(f"\nSaved overall summary to: {summary_filename}")

    # Create CSV summary
    create_csv_summary(all_instance_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)

    return all_instance_results

def create_csv_summary(all_results: Dict):
    """Create CSV summary of results.

    Args:
        all_results: Dictionary with all instance results
    """
    csv_rows = []

    for instance_name, instance_data in all_results.items():
        for init_type in ['random', 'greedy']:
            metrics = instance_data[f'{init_type}_initialization']['averaged_metrics']

            row = {
                'Instance': instance_name,
                'Initialization': init_type,
                'Mean_Initial_Fitness': metrics['mean_initial_fitness'],
                'Mean_Final_Fitness': metrics['mean_final_fitness'],
                'Mean_Improvement_%': metrics['mean_improvement_percent'],
                'Std_Final_Fitness': metrics['std_final_fitness'],
                'Mean_Iterations': metrics['mean_iterations'],
                'Mean_Time_Seconds': metrics['mean_time_seconds'],
                'Mean_Acceptance_Rate': metrics['mean_acceptance_rate'],
                'Mean_Num_Routes': metrics['mean_num_routes']
            }
            csv_rows.append(row)

    # Write CSV
    csv_filename = Path(RESULTS_DIR) / "rl_local_search_all_instances_averaged_metrics.csv"
    if csv_rows:
        fieldnames = csv_rows[0].keys()
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved CSV summary to: {csv_filename.name}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_experiment()
        print("\nAll results successfully saved!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to {RESULTS_DIR}")

    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be saved to {RESULTS_DIR}")
