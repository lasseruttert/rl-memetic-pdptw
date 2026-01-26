from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions

from memetic.memetic_algorithm import MemeticSolver
from memetic.fitness.fitness import fitness

from memetic.crossover.srex import SREXCrossover
from memetic.mutation.dummy_mutation import DummyMutation

from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.merge import MergeOperator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator

from or_tools.or_tools import ORToolsSolver

import time
import json
import csv
import random
import numpy as np
import copy
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROBLEM_SIZES = [100, 200, 400]
MEMETIC_MAX_TIME_SECONDS = 180
ORTOOLS_MAX_TIME_SECONDS = 180
NUM_RUNS = 3
POPULATION_SIZE = 10

# Model paths per size
MODEL_PATHS = {
    100: "models/rl_local_search_dqn_100_greedy_binary_100_set2_final.pt",
    200: "models/rl_local_search_dqn_200_greedy_binary_100_set2_200_final.pt",
    400: "models/rl_local_search_dqn_400_greedy_binary_100_set2_400_final.pt",
}

# Output files
RESULTS_BASE_DIR = "results"
RESULTS_OUTPUT_FILE = "results/memetic_vs_ortools_results.json"
SUMMARY_CSV_FILE = "results/memetic_vs_ortools_summary.csv"

# ============================================================================
# COMPONENT CREATION FUNCTIONS
# ============================================================================

def create_operators_set2():
    """Create set2 operator list."""
    return [
        ReinsertOperator(),
        ReinsertOperator(clustered=True, max_attempts=5),
        ReinsertOperator(allow_same_vehicle=False),
        ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        RouteEliminationOperator(),
        TwoOptOperator(),
        SwapBetweenOperator(type="best"),
        MergeOperator(type="min"),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator(),
    ]


def create_rl_local_search(size, operators):
    """Create RLLocalSearch with correct model for size."""
    rl_local_search = RLLocalSearch(
        operators=operators,
        rl_algorithm="dqn",
        hidden_dims=[128, 128, 64],
        learning_rate=0.0001,
        gamma=0.90,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9975,
        target_update_interval=100,
        alpha=10,
        beta=0,
        acceptance_strategy="greedy",
        type="OneShot",
        max_iterations=200,
        max_no_improvement=50,
        replay_buffer_capacity=10000,
        batch_size=64,
        n_step=3,
        use_prioritized_replay=False,
        use_operator_attention=False,
        verbose=False
    )
    rl_local_search = RLLocalSearch.load_from_checkpoint(MODEL_PATHS[size])
    return rl_local_search


def create_memetic_solver(size, operators):
    """Create MemeticSolver with RL local search."""
    rl_local_search = create_rl_local_search(size, operators)

    solver = MemeticSolver(
        population_size=POPULATION_SIZE,
        max_time_seconds=MEMETIC_MAX_TIME_SECONDS,
        crossover_operator=SREXCrossover(),
        mutation_operator=DummyMutation(),
        local_search_operator=rl_local_search,
        track_convergence=True,
        verbose=False
    )
    return solver


def calculate_fitness(problem, solution):
    """Calculate fitness using the same formula as the solver."""
    return fitness(problem, solution)


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_experiment():
    """Run memetic vs OR-Tools comparison experiment."""
    print("=" * 80)
    print("MEMETIC VS OR-TOOLS COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Memetic max time: {MEMETIC_MAX_TIME_SECONDS} seconds")
    print(f"OR-Tools max time: {ORTOOLS_MAX_TIME_SECONDS} seconds")
    print(f"Number of runs (memetic): {NUM_RUNS}")
    print(f"Population size: {POPULATION_SIZE}")
    print("=" * 80)

    # Initialize instance managers
    li_lim_manager = LiLimInstanceManager()
    mendeley_manager = MendeleyInstanceManager()
    best_known_solutions = BestKnownSolutions()

    # Create operators (shared across sizes)
    operators = create_operators_set2()

    # Store all results
    all_results = {}

    for size in PROBLEM_SIZES:
        print(f"\n{'='*80}")
        print(f"PROCESSING SIZE {size}")
        print(f"{'='*80}")

        # Get instances for this size
        instances = []
        instances.extend(li_lim_manager.get_all(size=size))
        instances.extend(mendeley_manager.get_all(size=size))

        print(f"Total instances for size {size}: {len(instances)}")

        # Create solvers for this size
        memetic_solver = create_memetic_solver(size, operators)
        ortools_solver = ORToolsSolver(max_seconds=ORTOOLS_MAX_TIME_SECONDS, minimize_num_vehicles=False)

        size_results = {
            'size': size,
            'instances': {}
        }

        for inst_idx, instance in enumerate(instances):
            instance_name = instance.name
            print(f"\n  [{inst_idx+1}/{len(instances)}] Processing {instance_name}...")

            # Get BKS
            clean_instance_name = Path(instance_name).stem if ('/' in instance_name or '\\' in instance_name) else instance_name
            original_name = instance.name
            instance.name = clean_instance_name

            try:
                bks_num_vehicles, bks_total_distance = best_known_solutions.get_bks_as_tuple(instance)
                bks_fitness = bks_total_distance * (1 + bks_num_vehicles / instance.num_vehicles)
            except Exception as e:
                print(f"    Warning: Could not retrieve BKS for {clean_instance_name}: {e}")
                bks_fitness = None
                bks_num_vehicles = None
                bks_total_distance = None
            finally:
                instance.name = original_name

            # Initialize result entry
            instance_result = {
                'instance_name': instance_name,
                'size': size,
                'bks_fitness': bks_fitness,
                'bks_distance': bks_total_distance,
                'bks_vehicles': bks_num_vehicles,
                'memetic_fitness': None,
                'memetic_distance': None,
                'memetic_vehicles': None,
                'memetic_time_to_final': None,
                'ortools_fitness': None,
                'ortools_distance': None,
                'ortools_vehicles': None,
            }

            # Run Memetic (multiple runs, keep best)
            print(f"    Running Memetic algorithm ({NUM_RUNS} runs)...")
            best_memetic_result = None

            for run in range(NUM_RUNS):
                # Set deterministic seed based on instance and run
                seed = inst_idx * NUM_RUNS + run
                random.seed(seed)
                np.random.seed(seed)

                solution = memetic_solver.solve(instance)
                solution_fitness = calculate_fitness(instance, solution)

                time_to_final = memetic_solver.final_evaluation.get('time_to_final_fitness', None)

                run_result = {
                    'fitness': solution_fitness,
                    'distance': solution.total_distance,
                    'vehicles': solution.num_vehicles_used,
                    'time_to_final': time_to_final,
                    'is_feasible': solution.is_feasible,
                }

                print(f"      Run {run+1}/{NUM_RUNS}: Fitness={solution_fitness:.2f}, "
                      f"Vehicles={solution.num_vehicles_used}, Distance={solution.total_distance:.2f}, "
                      f"TimeToFinal={time_to_final}")

                if best_memetic_result is None or solution_fitness < best_memetic_result['fitness']:
                    best_memetic_result = run_result

            instance_result['memetic_fitness'] = best_memetic_result['fitness']
            instance_result['memetic_distance'] = best_memetic_result['distance']
            instance_result['memetic_vehicles'] = best_memetic_result['vehicles']
            instance_result['memetic_time_to_final'] = best_memetic_result['time_to_final']
            instance_result['memetic_is_feasible'] = best_memetic_result['is_feasible']

            # Run OR-Tools (single run, deterministic)
            print(f"    Running OR-Tools...")
            try:
                ortools_solution = ortools_solver.solve(instance)
                ortools_fitness = calculate_fitness(instance, ortools_solution)

                instance_result['ortools_fitness'] = ortools_fitness
                instance_result['ortools_distance'] = ortools_solution.total_distance
                instance_result['ortools_vehicles'] = ortools_solution.num_vehicles_used
                instance_result['ortools_is_feasible'] = ortools_solution.is_feasible

                print(f"      OR-Tools: Fitness={ortools_fitness:.2f}, "
                      f"Vehicles={ortools_solution.num_vehicles_used}, Distance={ortools_solution.total_distance:.2f}")
            except Exception as e:
                print(f"      OR-Tools failed: {e}")
                instance_result['ortools_fitness'] = None
                instance_result['ortools_distance'] = None
                instance_result['ortools_vehicles'] = None
                instance_result['ortools_is_feasible'] = False

            # Store result
            size_results['instances'][instance_name] = instance_result

            # Print comparison summary
            print(f"    Summary: BKS={bks_fitness:.2f if bks_fitness else 'N/A'}, "
                  f"Memetic={instance_result['memetic_fitness']:.2f}, "
                  f"OR-Tools={instance_result['ortools_fitness']:.2f if instance_result['ortools_fitness'] else 'N/A'}")

        all_results[f"size_{size}"] = size_results

        # Save intermediate results for this size
        intermediate_file = f"{RESULTS_BASE_DIR}/memetic_vs_ortools_size_{size}.json"
        Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)
        with open(intermediate_file, 'w') as f:
            json.dump(size_results, f, indent=2)
        print(f"\n  Intermediate results saved to {intermediate_file}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return all_results


def save_results(results):
    """Save results to JSON file."""
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)
    with open(RESULTS_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {RESULTS_OUTPUT_FILE}")


def save_summary_csv(results):
    """Create and save CSV summary of results."""
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)

    csv_rows = []
    for size_key, size_data in results.items():
        for instance_name, instance_data in size_data['instances'].items():
            row = {
                'Instance': instance_data['instance_name'],
                'Size': instance_data['size'],
                'BKS_Fitness': instance_data['bks_fitness'],
                'BKS_Distance': instance_data['bks_distance'],
                'BKS_Vehicles': instance_data['bks_vehicles'],
                'Memetic_Fitness': instance_data['memetic_fitness'],
                'Memetic_Distance': instance_data['memetic_distance'],
                'Memetic_Vehicles': instance_data['memetic_vehicles'],
                'Memetic_TimeToFinal': instance_data['memetic_time_to_final'],
                'Memetic_IsFeasible': instance_data.get('memetic_is_feasible', None),
                'ORTools_Fitness': instance_data['ortools_fitness'],
                'ORTools_Distance': instance_data['ortools_distance'],
                'ORTools_Vehicles': instance_data['ortools_vehicles'],
                'ORTools_IsFeasible': instance_data.get('ortools_is_feasible', None),
            }
            csv_rows.append(row)

    if csv_rows:
        fieldnames = csv_rows[0].keys()
        with open(SUMMARY_CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Summary CSV saved to {SUMMARY_CSV_FILE}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_experiment()

        # Save final results
        save_results(results)

        # Generate CSV summary
        print("\nGenerating CSV summary...")
        save_summary_csv(results)

        print("\nAll results successfully saved!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")

    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
