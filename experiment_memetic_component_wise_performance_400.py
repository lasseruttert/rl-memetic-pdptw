from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager
from utils.best_known_solutions import BestKnownSolutions

from memetic.memetic_algorithm import MemeticSolver

from memetic.selection.k_tournament import KTournamentSelection

from memetic.crossover.srex import SREXCrossover
from memetic.crossover.dummy_crossover import DummyCrossover

from memetic.mutation.naive_mutation import NaiveMutation
from memetic.mutation.dummy_mutation import DummyMutation
from memetic.mutation.rl_mutation.rl_mutation import RLMutation

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.local_search.adaptive_local_search import AdaptiveLocalSearch
from memetic.local_search.random_local_search import RandomLocalSearch
from memetic.local_search.dummy_local_search import DummyLocalSearch
from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.merge import MergeOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.shift import ShiftOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
from memetic.solution_operators.request_shift_within import RequestShiftWithinOperator
from memetic.solution_operators.node_swap_within import NodeSwapWithinOperator

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

# Problem sizes to test
PROBLEM_SIZES = [400]

# Memetic algorithm parameters
POPULATION_SIZE = 10
MAX_TIME_SECONDS = 60
EVALUATION_INTERVAL = 5

# Run parameters
NUM_RUNS = 3

# Output files
RESULTS_BASE_DIR = "results"
RESULTS_OUTPUT_FILE = "results/memetic_component_results_400.json"
SUMMARY_CSV_FILE = "results/memetic_component_summary_400.csv"

# ============================================================================
# COMPONENT CREATION FUNCTIONS
# ============================================================================
def create_operators():
    operator_dict = {}
    
    reinsert_base = ReinsertOperator()
    reinsert_5c = ReinsertOperator(max_attempts=5, clustered=True)
    reinsert_no_same = ReinsertOperator(allow_same_vehicle=False)
    reinsert_no_same_no_new = ReinsertOperator(allow_new_vehicles=False, allow_same_vehicle=False)
    route_elim = RouteEliminationOperator()
    flip = FlipOperator()
    flip_sr = FlipOperator(single_route=True)
    merge_min = MergeOperator(type="min")
    merge_random = MergeOperator(type="random")
    two_opt = TwoOptOperator()
    swap_within = SwapWithinOperator()
    swap_within_sr = SwapWithinOperator(single_route=True)
    transfer = TransferOperator()
    transfer_sr = TransferOperator(single_route=True)
    shift_random = ShiftOperator(type="random")
    shift_best = ShiftOperator(type="best")
    cls_m2 = CLSM2Operator()
    cls_m3 = CLSM3Operator()
    cls_m4 = CLSM4Operator()
    request_shift_within = RequestShiftWithinOperator()
    node_swap_within = NodeSwapWithinOperator()
    
    operator_dict["local_search_default"] = [
            ReinsertOperator(),
            ReinsertOperator(clustered=True, max_attempts=5),
            RouteEliminationOperator(),
            TwoOptOperator(),
            SwapWithinOperator(),
            SwapBetweenOperator(),
            TransferOperator(),
        ]
    
    operator_dict["mutation_default"] = [
            ReinsertOperator(),
            RouteEliminationOperator(),
            FlipOperator(single_route=True),
            SwapWithinOperator(single_route=True),
            SwapBetweenOperator(),
            TransferOperator(single_route=True),
            RequestShiftWithinOperator(),
            NodeSwapWithinOperator(),
        ]
    
    operator_dict["set1"] = [
            ReinsertOperator(),
            ReinsertOperator(clustered=True, max_attempts=5),
            RouteEliminationOperator(),
            TwoOptOperator(),
            SwapWithinOperator(),
            SwapBetweenOperator(),
            TransferOperator(),
        ]
    operator_dict["set2"] = [
            ReinsertOperator(),
            ReinsertOperator(clustered=True, max_attempts=5),
            ReinsertOperator(allow_same_vehicle = False),
            ReinsertOperator(allow_same_vehicle = False, allow_new_vehicles = False),
            RouteEliminationOperator(),
            TwoOptOperator(),
            SwapBetweenOperator(type="best"),
            MergeOperator(type = "min"),
            CLSM2Operator(),
            CLSM3Operator(),
            CLSM4Operator(),
        ]
    operator_dict["set3"] = [
            ReinsertOperator(),
            ReinsertOperator(clustered=True, max_attempts=5),
            RouteEliminationOperator(),
            FlipOperator(),
            MergeOperator(type = "min"),
            MergeOperator(type = "random"),
            TwoOptOperator(),
            SwapWithinOperator(),
            SwapWithinOperator(single_route = True),
            SwapBetweenOperator(),
            TransferOperator(),
            TransferOperator(single_route = True),
            ShiftOperator(type="random"),
            ShiftOperator(type="best"),
            CLSM2Operator(),
            CLSM3Operator(),
            CLSM4Operator(),
        ]
    operator_dict["set4"] = [
            RouteEliminationOperator(),
            FlipOperator(),
            MergeOperator(type = "min"),
            MergeOperator(type = "random"),
            TwoOptOperator(),
            SwapWithinOperator(),
            SwapWithinOperator(single_route = True),
            SwapBetweenOperator(),
            TransferOperator(),
            TransferOperator(single_route = True),
            ShiftOperator(type="random"),
            ShiftOperator(type="best"),
            CLSM2Operator(),
            CLSM3Operator(),
            CLSM4Operator(),
        ]
    operator_dict["set5"] = [
            ReinsertOperator(),
            RouteEliminationOperator(),
            FlipOperator(single_route=True),
            SwapWithinOperator(single_route=True),
            SwapBetweenOperator(),
            TransferOperator(single_route=True),
            TwoOptOperator(),
            CLSM2Operator(),
            CLSM3Operator(),
            CLSM4Operator(),
            RequestShiftWithinOperator(),
            NodeSwapWithinOperator(),
        ]
    return operator_dict

def create_selection_instances():
    """Create and return list of selection operator instances.

    Returns:
        list: List of selection operator instances
    """
    return [
        KTournamentSelection(k=2),
        KTournamentSelection(k=3),
    ]

def create_crossover_instances():
    """Create and return list of crossover operator instances.

    Returns:
        list: List of crossover operator instances
    """
    return [
        SREXCrossover(),
        DummyCrossover(),
    ]

def create_mutation_instances(operator_dict):
    """Create and return list of mutation wrapper instances.

    Args:
        mutation_operators: List of base operators for mutation

    Returns:
        list: List of mutation wrapper instances
    """
    mutations = [
        DummyMutation(),
        NaiveMutation(operators=operator_dict["mutation_default"], max_iterations=1),
        NaiveMutation(operators=operator_dict["mutation_default"], max_iterations=10),
    ]
    
    return mutations

def create_local_search_instances(operator_dict):
    """Create and return list of local search wrapper instances.

    Args:
        local_search_operators: List of base operators for local search

    Returns:
        list: List of local search wrapper instances
    """
    local_searches = [
    ]
    
    ls_00_dummy = DummyLocalSearch()
    local_searches.append(ls_00_dummy)
    
    ls_01_naive_10_30 = NaiveLocalSearch(operators=operator_dict["set2"], max_no_improvement=10, max_iterations=30, random_operator_order=True)
    local_searches.append(ls_01_naive_10_30)
    
    ls_02_naive_50_200 = NaiveLocalSearch(operators=operator_dict["set2"], max_no_improvement=50, max_iterations=200, random_operator_order=True)
    local_searches.append(ls_02_naive_50_200)
    
    ls_03_naive_10_30_best = NaiveLocalSearch(operators=operator_dict["set2"], max_no_improvement=10, max_iterations=30, first_improvement=False)
    local_searches.append(ls_03_naive_10_30_best)
    
    ls_04_naive_50_200_best = NaiveLocalSearch(operators=operator_dict["set2"], max_no_improvement=50, max_iterations=200, first_improvement=False)
    local_searches.append(ls_04_naive_50_200_best)
    
    ls_05_random_10_30 = RandomLocalSearch(operators=operator_dict["set2"], max_no_improvement=10, max_iterations=30)
    local_searches.append(ls_05_random_10_30)
    
    ls_06_rl_set2_one_shot = RLLocalSearch.load_from_checkpoint("models/rl_local_search_dqn_400_greedy_binary_100_set2_400_final.pt")
    local_searches.append(ls_06_rl_set2_one_shot)
    
    ls_07_rl_set2_ranking = RLLocalSearch.load_from_checkpoint("models/rl_local_search_dqn_400_greedy_binary_100_set2_400_final.pt")
    ls_07_rl_set2_ranking.type = "Ranking"
    local_searches.append(ls_07_rl_set2_ranking)
    
    ls_06_rl_set2_50_50_one_shot = RLLocalSearch.load_from_checkpoint("models/rl_local_search_dqn_200_greedy_binary_100_set2_200_final.pt")
    ls_06_rl_set2_50_50_one_shot.max_iterations = 50
    ls_06_rl_set2_50_50_one_shot.max_no_improvement = 50
    local_searches.append(ls_06_rl_set2_50_50_one_shot)
    
    ls_07_rl_set2_50_50_ranking = RLLocalSearch.load_from_checkpoint("models/rl_local_search_dqn_200_greedy_binary_100_set2_200_final.pt")
    ls_07_rl_set2_50_50_ranking.max_iterations = 50
    ls_07_rl_set2_50_50_ranking.max_no_improvement = 50
    ls_07_rl_set2_50_50_ranking.type = "Ranking"
    local_searches.append(ls_07_rl_set2_50_50_ranking)
    
    return local_searches

# ============================================================================
# COMBINATIONS TO TEST
# ============================================================================

# Each combination specifies indices into the component lists
# Format: {'name': str, 'selection': idx, 'crossover': idx, 'mutation': idx, 'local_search': idx}
COMBINATIONS = [
    # {'name': 'Baseline_Short', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 1},
    # {'name': 'Baseline_Long', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 2},
    # {'name': 'No_LocalSearch', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 0},
    # {'name': 'LS_Set2_OneShot', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 6},
    # {'name': 'LS_Set2_Ranking', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 7},
    {'name': 'LS_Set2_50_50_OneShot', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 8},
    {'name': 'LS_Set2_50_50_Ranking', 'selection': 1, 'crossover': 1, 'mutation': 0, 'local_search': 9},
]

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_experiment():
    """Run memetic component-wise performance experiment.

    Tests different combinations of memetic algorithm components across all instances.
    Results are saved incrementally to allow for interrupted runs.
    """
    print("=" * 80)
    print("MEMETIC COMPONENT-WISE PERFORMANCE EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Max time per instance: {MAX_TIME_SECONDS} seconds")
    print(f"Total combinations: {len(COMBINATIONS)}")
    print("=" * 80)

    # Create component instances
    operator_dict = create_operators()
    selection_instances = create_selection_instances()
    crossover_instances = create_crossover_instances()
    mutation_instances = create_mutation_instances(operator_dict)
    local_search_instances = create_local_search_instances(operator_dict)

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

    # Store all results
    all_results = {}

    # For each combination
    for combo_idx, combination in enumerate(COMBINATIONS):
        combo_id = f"combo_{combo_idx}"
        combo_name = combination['name']
        print(f"\n[{combo_idx+1}/{len(COMBINATIONS)}] Starting combination {combo_idx}: {combo_name}")
        print(f"  Selection: {combination['selection']}, Crossover: {combination['crossover']}, "
              f"Mutation: {combination['mutation']}, LocalSearch: {combination['local_search']}")
        print("-" * 80)

        combo_start_time = time.time()

        # Create solver with this combination
        solver = MemeticSolver(
            population_size=POPULATION_SIZE,
            max_time_seconds=MAX_TIME_SECONDS,
            selection_operator=selection_instances[combination['selection']],
            crossover_operator=crossover_instances[combination['crossover']],
            mutation_operator=mutation_instances[combination['mutation']],
            local_search_operator=local_search_instances[combination['local_search']],
            evaluation_interval=EVALUATION_INTERVAL,
            verbose=False,
            track_convergence=True
        )

        combination_results = {
            'combination_id': combo_id,
            'combination_name': combo_name,
            'selection_idx': combination['selection'],
            'crossover_idx': combination['crossover'],
            'mutation_idx': combination['mutation'],
            'local_search_idx': combination['local_search'],
            'instances': {}
        }

        # For each instance
        for inst_idx, instance in enumerate(all_instances):
            instance_name = instance.name
            print(f"\n  [{inst_idx+1}/{len(all_instances)}] Running on {instance_name}...")

            # Set deterministic seed based on instance
            seed = inst_idx
            random.seed(seed)
            np.random.seed(seed)

            # Get best known solution
            # Note: instance.name may contain full path, extract just the filename
            clean_instance_name = Path(instance_name).stem if ('/' in instance_name or '\\' in instance_name) else instance_name

            # Temporarily update instance name for BKS lookup
            original_name = instance.name
            instance.name = clean_instance_name

            try:
                # Note: get_bks_as_tuple returns (num_vehicles, total_distance)
                bks_num_vehicles, bks_total_distance = best_known_solutions.get_bks_as_tuple(instance)

                # Calculate BKS fitness using the same formula as solver
                # Assuming BKS solutions are feasible (penalty = 0)
                bks_fitness = bks_total_distance * (1 + bks_num_vehicles / instance.num_vehicles)
            except Exception as e:
                print(f"    Warning: Could not retrieve BKS for {clean_instance_name}: {e}")
                bks_fitness = None
                bks_num_vehicles = None
                bks_total_distance = None
            finally:
                # Restore original name
                instance.name = original_name

            best_results = None
            
            for i in range(NUM_RUNS):
                print(f"    Run {i+1}/{NUM_RUNS}...")
                
                # Run solver
                run_start_time = time.time()
                best_solution = solver.solve(instance)
                run_elapsed_time = time.time() - run_start_time

                # Collect results 
                instance_results = {
                    'instance_name': instance_name,
                    'bks_fitness': bks_fitness,
                    'bks_num_vehicles': bks_num_vehicles,
                    'bks_total_distance': bks_total_distance,
                    'best_fitness': solver.fitness_function(instance, best_solution),
                    'best_num_vehicles': best_solution.num_vehicles_used,
                    'best_total_distance': best_solution.total_distance,
                    'is_feasible': best_solution.is_feasible,
                    'total_time': run_elapsed_time,
                    #'convergence': copy.deepcopy(solver.convergence),
                    #'evaluations': copy.deepcopy(solver.evaluations),
                    'final_evaluation': copy.deepcopy(solver.final_evaluation)
                }
                
                if best_results is None or instance_results['best_fitness'] < best_results['best_fitness']:
                    best_results = copy.deepcopy(instance_results)

            combination_results['instances'][instance_name] = best_results

            print(f"    Best fitness: {instance_results['best_fitness']:.2f}, "
                  f"Vehicles: {instance_results['best_num_vehicles']}, "
                  f"Time: {run_elapsed_time:.2f}s",
                  f"Time to best: {instance_results['final_evaluation'].get('time_to_final_fitness', None)}s"
                  )

        combo_elapsed = time.time() - combo_start_time
        print(f"\n  Combination completed in {combo_elapsed:.2f} seconds")

        # Store results
        all_results[combo_id] = combination_results

        # Save individual combo file
        combo_file = f"{RESULTS_BASE_DIR}/combo_{combo_idx}_{combo_name}_{PROBLEM_SIZES[0]}.json"
        with open(combo_file, 'w') as f:
            json.dump(combination_results, f, indent=2)
        print(f"  Saved to {combo_file}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_CSV_FILE}")
    print("=" * 80)

    return all_results

def save_results(results):
    """Save results to JSON file.

    Args:
        results: Dictionary of results to save
    """
    # Ensure results directory exists
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)

    with open(RESULTS_OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def save_summary_csv(results):
    """Create and save CSV summary of results.

    Args:
        results: Dictionary containing all experiment results
    """
    # Ensure results directory exists
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)

    # Prepare CSV data
    csv_rows = []

    for combo_id, combo_data in results.items():
        for instance_name, instance_data in combo_data['instances'].items():
            row = {
                'Combination_ID': combo_id,
                'Combination_Name': combo_data['combination_name'],
                'Selection_Idx': combo_data['selection_idx'],
                'Crossover_Idx': combo_data['crossover_idx'],
                'Mutation_Idx': combo_data['mutation_idx'],
                'LocalSearch_Idx': combo_data['local_search_idx'],
                'Instance': instance_name,
                'BKS_Fitness': instance_data['bks_fitness'],
                'BKS_Vehicles': instance_data['bks_num_vehicles'],
                'BKS_Distance': instance_data['bks_total_distance'],
                'Best_Fitness': instance_data['best_fitness'],
                'Best_Vehicles': instance_data['best_num_vehicles'],
                'Best_Distance': instance_data['best_total_distance'],
                'Is_Feasible': instance_data['is_feasible'],
                'Total_Time': instance_data['total_time'],
                'Time_To_Best': instance_data['final_evaluation'].get('time_to_final_fitness', None) if instance_data['final_evaluation'] else None,
                'Num_Improvements': instance_data['final_evaluation'].get('num_improvements', None) if instance_data['final_evaluation'] else None,
                'Longest_Stagnation': instance_data['final_evaluation'].get('longest_stagnation', None) if instance_data['final_evaluation'] else None,
            }
            csv_rows.append(row)

    # Write CSV
    if csv_rows:
        fieldnames = csv_rows[0].keys()
        with open(SUMMARY_CSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

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
        print(f"Partial results saved to {RESULTS_OUTPUT_FILE}")

    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be saved to {RESULTS_OUTPUT_FILE}")
