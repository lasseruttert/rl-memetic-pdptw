"""
RL Local Search Feature Ablation Experiment

Loads a trained RL local search model and systematically mutes each state feature
(sets to zero), then runs on all benchmark instances to measure the impact on
performance. Compares average, min, and max fitness across all features to
identify which state features the model relies on most.
"""

from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager

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
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Problem sizes to test
PROBLEM_SIZES = [100]

# Number of runs per instance (averaged for stability)
NUM_RUNS = 3

# Initialization type: "random" or "greedy"
INIT_TYPE = "random"

# Path to trained RL local search model
MODEL_PATH = "models/rl_local_search_dqn_100_greedy_binary_100_set2_final.pt"

# Results directory
RESULTS_DIR = "results/feature_ablation"

# Random seed for reproducibility
SEED = 42

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Solution feature names (indices 0-13 in the state vector)
SOLUTION_FEATURE_NAMES = [
    "num_requests_norm",        # 0
    "vehicle_capacity_norm",    # 1
    "num_vehicles_norm",        # 2
    "route_utilization",        # 3  (num_routes / num_vehicles)
    "customer_service_ratio",   # 4  (served / total)
    "total_distance_norm",      # 5
    "is_feasible",              # 6
    "avg_route_length_norm",    # 7
    "max_route_length_norm",    # 8
    "min_route_length_norm",    # 9
    "std_route_length_norm",    # 10
    "avg_route_distance_norm",  # 11
    "max_route_distance_norm",  # 12
    "std_route_distance_norm",  # 13
]

NUM_SOLUTION_FEATURES = len(SOLUTION_FEATURE_NAMES)

# Operator feature names (3 per operator, starting at index 14)
OPERATOR_FEATURE_SUFFIXES = [
    "application_rate",
    "improvement_rate",
    "acceptance_rate",
]

# Feature groups for group-level ablation
FEATURE_GROUPS = {
    "problem_features": [0, 1, 2],
    "solution_efficiency": [3, 4, 5, 6],
    "route_length_stats": [7, 8, 9, 10],
    "route_distance_stats": [11, 12, 13],
    "all_route_features": [7, 8, 9, 10, 11, 12, 13],
}


# ============================================================================
# FEATURE MUTING
# ============================================================================

def build_feature_name_map(num_operators: int, operator_names: List[str]) -> Dict[str, List[int]]:
    """Build a mapping from human-readable feature names to state vector indices.

    Args:
        num_operators: Number of operators in the model
        operator_names: List of operator names

    Returns:
        Dict mapping feature/group name -> list of indices to mute
    """
    feature_map = {}

    # Individual solution features
    for idx, name in enumerate(SOLUTION_FEATURE_NAMES):
        feature_map[name] = [idx]

    # Individual operator features and per-operator groups
    for op_idx in range(num_operators):
        op_name = operator_names[op_idx] if op_idx < len(operator_names) else f"op_{op_idx}"
        base = NUM_SOLUTION_FEATURES + op_idx * 3

        # Per-operator group (all 3 features for this operator)
        feature_map[f"operator_{op_name}_all"] = [base, base + 1, base + 2]

        # Individual operator features
        for feat_idx, suffix in enumerate(OPERATOR_FEATURE_SUFFIXES):
            feature_map[f"operator_{op_name}_{suffix}"] = [base + feat_idx]

    # Feature groups
    for group_name, indices in FEATURE_GROUPS.items():
        feature_map[f"group_{group_name}"] = indices

    # All operator features
    all_op_indices = list(range(NUM_SOLUTION_FEATURES,
                                NUM_SOLUTION_FEATURES + num_operators * 3))
    feature_map["group_all_operator_features"] = all_op_indices

    # All solution features
    feature_map["group_all_solution_features"] = list(range(NUM_SOLUTION_FEATURES))

    return feature_map


def create_muted_get_state(original_get_state, mute_indices: List[int]):
    """Create a wrapper around _get_state that zeros out specified feature indices.

    Args:
        original_get_state: The original _get_state method
        mute_indices: List of feature indices to set to zero

    Returns:
        Wrapped function that mutes the specified features
    """
    def muted_get_state(self_env):
        state = original_get_state()
        state[mute_indices] = 0.0
        return state
    return muted_get_state


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_single_instance(
    rl_local_search: RLLocalSearch,
    instance,
    generator,
    num_runs: int,
    seed: int,
    inst_idx: int,
) -> Dict:
    """Run RL local search on a single instance multiple times.

    Args:
        rl_local_search: RL local search instance
        instance: Problem instance
        generator: Solution generator
        num_runs: Number of runs
        seed: Random seed
        inst_idx: Instance index for seeding

    Returns:
        Dict with fitness statistics
    """
    final_fitnesses = []

    for run_idx in range(num_runs):
        # Generate initial solution with deterministic seed
        random.seed(seed + inst_idx * 1000 + run_idx)
        np.random.seed(seed + inst_idx * 1000 + run_idx)
        initial_solution = generator.generate(instance, 1)[0]

        # Apply RL local search
        result = rl_local_search.search(
            problem=instance,
            solution=initial_solution,
            epsilon=0.0,
            deterministic_rng=True,
            base_seed=seed + inst_idx * 1000 + run_idx
        )

        # Unpack (tracking returns 3 values, non-tracking returns 2)
        if rl_local_search.tracking:
            best_solution, best_fitness, _ = result
        else:
            best_solution, best_fitness = result

        final_fitnesses.append(best_fitness)

    return {
        'mean_fitness': float(np.mean(final_fitnesses)),
        'std_fitness': float(np.std(final_fitnesses)),
        'min_fitness': float(np.min(final_fitnesses)),
        'max_fitness': float(np.max(final_fitnesses)),
        'all_fitnesses': final_fitnesses,
    }


def run_ablation_condition(
    rl_local_search: RLLocalSearch,
    all_instances: list,
    generator,
    mute_indices: Optional[List[int]],
    condition_name: str,
) -> Dict:
    """Run one ablation condition (baseline or muted feature) across all instances.

    Args:
        rl_local_search: RL local search instance
        all_instances: List of problem instances
        generator: Solution generator
        mute_indices: Feature indices to mute (None for baseline)
        condition_name: Name of this condition

    Returns:
        Dict with per-instance and aggregate results
    """
    # Monkey-patch _get_state if muting
    original_get_state = None
    if mute_indices is not None:
        original_get_state = rl_local_search.env._get_state

        def muted_get_state():
            state = original_get_state()
            state[mute_indices] = 0.0
            return state

        rl_local_search.env._get_state = muted_get_state

    instance_results = {}
    all_mean_fitnesses = []

    for inst_idx, instance in enumerate(all_instances):
        instance_name = Path(instance.name).stem if ('/' in instance.name or '\\' in instance.name) else instance.name

        result = run_single_instance(
            rl_local_search=rl_local_search,
            instance=instance,
            generator=generator,
            num_runs=NUM_RUNS,
            seed=SEED,
            inst_idx=inst_idx,
        )

        instance_results[instance_name] = result
        all_mean_fitnesses.append(result['mean_fitness'])

    # Restore original _get_state
    if original_get_state is not None:
        rl_local_search.env._get_state = original_get_state

    # Aggregate across all instances
    all_fitnesses_flat = []
    for res in instance_results.values():
        all_fitnesses_flat.extend(res['all_fitnesses'])

    aggregate = {
        'condition_name': condition_name,
        'muted_indices': mute_indices,
        'num_instances': len(all_instances),
        'num_runs_per_instance': NUM_RUNS,
        'avg_fitness': float(np.mean(all_mean_fitnesses)),
        'std_fitness': float(np.std(all_mean_fitnesses)),
        'min_fitness': float(np.min(all_mean_fitnesses)),
        'max_fitness': float(np.max(all_mean_fitnesses)),
        'avg_fitness_all_runs': float(np.mean(all_fitnesses_flat)),
        'min_fitness_all_runs': float(np.min(all_fitnesses_flat)),
        'max_fitness_all_runs': float(np.max(all_fitnesses_flat)),
        'instance_results': instance_results,
    }

    return aggregate


def run_experiment():
    """Run the full feature ablation experiment."""
    print("=" * 80)
    print("RL LOCAL SEARCH FEATURE ABLATION EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Runs per instance: {NUM_RUNS}")
    print(f"Initialization: {INIT_TYPE}")
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
    rl_local_search.tracking = False  # Disable tracking for speed

    # Get operator info
    num_operators = len(rl_local_search.operators)
    operator_names = [op.__class__.__name__ for op in rl_local_search.operators]
    state_dim = rl_local_search.env.observation_space.shape[0]

    print(f"\nState dimension: {state_dim}")
    print(f"Number of operators: {num_operators}")
    print(f"Operator names: {operator_names}")
    print(f"Solution features: {NUM_SOLUTION_FEATURES}")
    print(f"Operator features: {num_operators * 3}")

    # Build feature name map
    feature_map = build_feature_name_map(num_operators, operator_names)

    print(f"\nTotal ablation conditions: {len(feature_map)} + 1 baseline")
    print(f"Features to ablate:")
    for name, indices in feature_map.items():
        print(f"  {name}: indices {indices}")

    # Initialize instance managers
    li_lim_manager = LiLimInstanceManager()
    mendeley_manager = MendeleyInstanceManager()

    # Get all instances
    all_instances = []
    for size in PROBLEM_SIZES:
        all_instances.extend(li_lim_manager.get_all(size=size))
        all_instances.extend(mendeley_manager.get_all(size=size))

    print(f"\nTotal instances: {len(all_instances)}")

    # Initialize solution generator
    if INIT_TYPE == "random":
        generator = RandomGenerator()
    elif INIT_TYPE == "greedy":
        generator = GreedyGenerator()
    else:
        raise ValueError(f"Unknown initialization type: {INIT_TYPE}")

    # ---- Run baseline (no muting) ----
    print("\n" + "=" * 80)
    print("Running BASELINE (no features muted)...")
    print("=" * 80)

    start_time = time.time()
    baseline_results = run_ablation_condition(
        rl_local_search=rl_local_search,
        all_instances=all_instances,
        generator=generator,
        mute_indices=None,
        condition_name="baseline",
    )
    baseline_time = time.time() - start_time

    print(f"  Baseline avg fitness: {baseline_results['avg_fitness']:.2f}")
    print(f"  Baseline min fitness: {baseline_results['min_fitness']:.2f}")
    print(f"  Baseline max fitness: {baseline_results['max_fitness']:.2f}")
    print(f"  Time: {baseline_time:.1f}s")

    # ---- Run ablation for each feature / group ----
    ablation_results = {}

    for feat_idx, (feat_name, mute_indices) in enumerate(feature_map.items()):
        print(f"\n[{feat_idx+1}/{len(feature_map)}] Muting: {feat_name} (indices {mute_indices})")

        start_time = time.time()
        result = run_ablation_condition(
            rl_local_search=rl_local_search,
            all_instances=all_instances,
            generator=generator,
            mute_indices=mute_indices,
            condition_name=feat_name,
        )
        elapsed = time.time() - start_time

        # Compute degradation relative to baseline
        degradation = result['avg_fitness'] - baseline_results['avg_fitness']
        degradation_pct = (degradation / baseline_results['avg_fitness'] * 100
                          if baseline_results['avg_fitness'] != 0 else 0.0)

        result['degradation'] = degradation
        result['degradation_pct'] = degradation_pct

        ablation_results[feat_name] = result

        print(f"  Avg fitness: {result['avg_fitness']:.2f} "
              f"(delta: {degradation:+.2f}, {degradation_pct:+.2f}%)")
        print(f"  Min: {result['min_fitness']:.2f}  Max: {result['max_fitness']:.2f}")
        print(f"  Time: {elapsed:.1f}s")

    # ---- Save results ----
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Full results JSON
    full_results = {
        'experiment': 'Feature Ablation',
        'model_path': MODEL_PATH,
        'init_type': INIT_TYPE,
        'num_runs': NUM_RUNS,
        'problem_sizes': PROBLEM_SIZES,
        'seed': SEED,
        'num_instances': len(all_instances),
        'state_dim': state_dim,
        'num_operators': num_operators,
        'operator_names': operator_names,
        'baseline': baseline_results,
        'ablations': {name: {k: v for k, v in res.items() if k != 'instance_results'}
                      for name, res in ablation_results.items()},
    }

    full_path = Path(RESULTS_DIR) / "feature_ablation_full_results.json"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full results to: {full_path}")

    # Per-instance details JSON
    detail_data = {
        'baseline': baseline_results,
        'ablations': ablation_results,
    }
    detail_path = Path(RESULTS_DIR) / "feature_ablation_per_instance_details.json"
    with open(detail_path, 'w') as f:
        json.dump(detail_data, f, indent=2, default=str)
    print(f"Saved per-instance details to: {detail_path}")

    # CSV summary
    create_csv_summary(baseline_results, ablation_results)

    # Print final summary table
    print_summary_table(baseline_results, ablation_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)

    return baseline_results, ablation_results


def create_csv_summary(baseline: Dict, ablations: Dict):
    """Create a CSV summary sorted by degradation (most impactful first).

    Args:
        baseline: Baseline results
        ablations: Dict of ablation results keyed by feature name
    """
    rows = []

    # Baseline row
    rows.append({
        'Feature': 'BASELINE (no muting)',
        'Muted_Indices': '',
        'Avg_Fitness': baseline['avg_fitness'],
        'Std_Fitness': baseline['std_fitness'],
        'Min_Fitness': baseline['min_fitness'],
        'Max_Fitness': baseline['max_fitness'],
        'Degradation': 0.0,
        'Degradation_%': 0.0,
    })

    # Ablation rows
    for feat_name, result in ablations.items():
        rows.append({
            'Feature': feat_name,
            'Muted_Indices': str(result['muted_indices']),
            'Avg_Fitness': result['avg_fitness'],
            'Std_Fitness': result['std_fitness'],
            'Min_Fitness': result['min_fitness'],
            'Max_Fitness': result['max_fitness'],
            'Degradation': result['degradation'],
            'Degradation_%': result['degradation_pct'],
        })

    # Sort by degradation (largest first = most important feature)
    rows.sort(key=lambda r: r['Degradation'], reverse=True)

    csv_path = Path(RESULTS_DIR) / "feature_ablation_summary.csv"
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV summary to: {csv_path}")


def print_summary_table(baseline: Dict, ablations: Dict):
    """Print a formatted summary table to console.

    Args:
        baseline: Baseline results
        ablations: Dict of ablation results keyed by feature name
    """
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE RANKING (sorted by performance degradation)")
    print("=" * 100)
    print(f"{'Rank':<5} {'Feature':<40} {'Avg Fitness':>12} {'Degradation':>12} {'Deg %':>8} "
          f"{'Min':>10} {'Max':>10}")
    print("-" * 100)

    # Baseline
    print(f"{'---':<5} {'BASELINE':<40} {baseline['avg_fitness']:>12.2f} "
          f"{'0.00':>12} {'0.00':>8} "
          f"{baseline['min_fitness']:>10.2f} {baseline['max_fitness']:>10.2f}")
    print("-" * 100)

    # Sort by degradation (largest = most important)
    sorted_features = sorted(
        ablations.items(),
        key=lambda x: x[1]['degradation'],
        reverse=True
    )

    for rank, (feat_name, result) in enumerate(sorted_features, 1):
        print(f"{rank:<5} {feat_name:<40} {result['avg_fitness']:>12.2f} "
              f"{result['degradation']:>+12.2f} {result['degradation_pct']:>+8.2f} "
              f"{result['min_fitness']:>10.2f} {result['max_fitness']:>10.2f}")

    print("=" * 100)
    print("Positive degradation = performance got WORSE when feature was muted (feature is important)")
    print("Negative degradation = performance got BETTER when feature was muted (feature may be harmful)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        baseline, ablations = run_experiment()
        print("\nAll results successfully saved!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to {RESULTS_DIR}")

    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be saved to {RESULTS_DIR}")
