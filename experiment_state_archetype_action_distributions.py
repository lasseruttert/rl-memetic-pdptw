"""
State Archetype Action Distribution Experiment

Shows that the trained RL local search agent adapts its action probabilities
based on the solution state, rather than using a fixed operator preference.

Defines 3 state archetypes based on solution features:
  1. Infeasible — feature[6] == 0
  2. Feasible, High Distance — feature[6] == 1 AND feature[5] > median
  3. Feasible, Compact — feature[6] == 1 AND feature[5] <= median

Collects real states from running the model on benchmark instances, computes
softmax action probability distributions from Q-values, and compares them
across archetypes with statistical tests.

Complements experiment_feature_ablation.py (which shows *which* features matter)
by showing *how* the agent uses them to make different decisions.
"""

from utils.li_lim_instance_manager import LiLimInstanceManager
from utils.mendeley_instance_manager import MendeleyInstanceManager

from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch
from memetic.solution_generators.random_generator import RandomGenerator
from memetic.solution_generators.greedy_generator import GreedyGenerator

import time
import json
import csv
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import mannwhitneyu, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Statistical tests will be skipped.")

# Solution feature names (indices 0-13 in the state vector)
SOLUTION_FEATURE_NAMES = [
    "num_requests_norm",        # 0
    "vehicle_capacity_norm",    # 1
    "num_vehicles_norm",        # 2
    "route_utilization",        # 3
    "customer_service_ratio",   # 4
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

# ============================================================================
# CONFIGURATION
# ============================================================================

PROBLEM_SIZES = [100]
NUM_RUNS = 3
INIT_TYPE = "random"
MODEL_PATH = "models/rl_local_search_dqn_100_greedy_binary_100_set2_final.pt"
RESULTS_DIR = "results/state_archetype_distributions"
SEED = 42

# Softmax temperature for converting Q-values to probabilities
SOFTMAX_TEMPERATURE = 1.0

# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_state_data(
    rl_local_search: RLLocalSearch,
    all_instances: list,
    generator,
    num_runs: int,
    seed: int,
) -> List[Dict]:
    """Run search with tracking and extract per-iteration (state, action) records.

    Args:
        rl_local_search: RL local search instance with tracking enabled
        all_instances: List of problem instances
        generator: Solution generator
        num_runs: Number of runs per instance
        seed: Random seed

    Returns:
        List of dicts, each with 'state_features' (list) and 'action' (int)
    """
    records = []

    for inst_idx, instance in enumerate(all_instances):
        instance_name = (Path(instance.name).stem
                         if ('/' in instance.name or '\\' in instance.name)
                         else instance.name)

        for run_idx in range(num_runs):
            # Deterministic seeding
            random.seed(seed + inst_idx * 1000 + run_idx)
            np.random.seed(seed + inst_idx * 1000 + run_idx)

            initial_solution = generator.generate(instance, 1)[0]

            best_solution, best_fitness, run_history = rl_local_search.search(
                problem=instance,
                solution=initial_solution,
                epsilon=0.0,
                deterministic_rng=True,
                base_seed=seed + inst_idx * 1000 + run_idx,
            )

            # Extract per-iteration records
            for iteration_key, iteration_data in run_history.items():
                records.append({
                    'state_features': iteration_data['state_features'],
                    'action': iteration_data['action'],
                    'instance': instance_name,
                    'run': run_idx,
                    'iteration': iteration_key,
                })

        print(f"  [{inst_idx+1}/{len(all_instances)}] {instance_name}: "
              f"{len(records)} total records so far")

    return records


# ============================================================================
# Q-VALUE / SOFTMAX COMPUTATION
# ============================================================================

def compute_q_values_for_records(
    agent,
    records: List[Dict],
    temperature: float = 1.0,
) -> None:
    """Feed states through agent to get Q-values and softmax probabilities.

    Modifies records in-place, adding 'q_values' and 'action_probs' fields.

    Args:
        agent: DQN agent with get_q_values() method
        records: List of record dicts (must have 'state_features')
        temperature: Softmax temperature
    """
    for record in records:
        state = np.array(record['state_features'], dtype=np.float32)
        q_values = agent.get_q_values(state, update_stats=False)

        # Softmax with temperature
        q_shifted = q_values - np.max(q_values)  # numerical stability
        exp_q = np.exp(q_shifted / temperature)
        probs = exp_q / np.sum(exp_q)

        record['q_values'] = q_values.tolist()
        record['action_probs'] = probs.tolist()


# ============================================================================
# ARCHETYPE CLASSIFICATION
# ============================================================================

def classify_archetypes(records: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Classify records into multiple archetype groupings.

    Grouping 1 — Feasibility & Distance:
      - Infeasible: feature[6] == 0
      - Feasible, High Distance: feature[6] == 1 AND feature[5] > median
      - Feasible, Compact: feature[6] == 1 AND feature[5] <= median

    Grouping 2 — Route Count (feasible only):
      - Many Routes: feature[3] (route_utilization) > median
      - Few Routes: feature[3] <= median

    Grouping 3 — Route Balance (feasible only):
      - Imbalanced Routes: feature[10] (std_route_length_norm) > median
      - Balanced Routes: feature[10] <= median

    Args:
        records: List of record dicts with 'state_features'

    Returns:
        Dict mapping grouping name -> Dict mapping archetype name -> list of records
    """
    # Separate feasible and infeasible
    feasible_records = []
    infeasible_records = []

    for record in records:
        is_feasible = record['state_features'][6]
        if is_feasible > 0.5:
            feasible_records.append(record)
        else:
            infeasible_records.append(record)

    # ---- Grouping 1: Feasibility & Distance ----
    if feasible_records:
        feasible_distances = [r['state_features'][5] for r in feasible_records]
        median_distance = float(np.median(feasible_distances))
    else:
        median_distance = 0.0

    high_distance_records = []
    compact_records = []
    for record in feasible_records:
        if record['state_features'][5] > median_distance:
            high_distance_records.append(record)
        else:
            compact_records.append(record)

    grouping_feasibility = {
        'Infeasible': infeasible_records,
        'Feasible, High Distance': high_distance_records,
        'Feasible, Compact': compact_records,
    }

    # ---- Grouping 2: Route Count (feasible only) ----
    if feasible_records:
        route_utils = [r['state_features'][3] for r in feasible_records]
        median_route_util = float(np.median(route_utils))
    else:
        median_route_util = 0.0

    many_routes = []
    few_routes = []
    for record in feasible_records:
        if record['state_features'][3] > median_route_util:
            many_routes.append(record)
        else:
            few_routes.append(record)

    grouping_route_count = {
        'Many Routes': many_routes,
        'Few Routes': few_routes,
    }

    # ---- Grouping 3: Route Balance (feasible only) ----
    if feasible_records:
        route_stds = [r['state_features'][10] for r in feasible_records]
        median_route_std = float(np.median(route_stds))
    else:
        median_route_std = 0.0

    imbalanced = []
    balanced = []
    for record in feasible_records:
        if record['state_features'][10] > median_route_std:
            imbalanced.append(record)
        else:
            balanced.append(record)

    grouping_route_balance = {
        'Imbalanced Routes': imbalanced,
        'Balanced Routes': balanced,
    }

    # ---- Print summary ----
    print(f"\n  Archetype classification:")
    print(f"\n  Grouping 1: Feasibility & Distance")
    print(f"    Median distance (feasible): {median_distance:.4f}")
    print(f"    Infeasible:              {len(infeasible_records):>6} records")
    print(f"    Feasible, High Distance: {len(high_distance_records):>6} records")
    print(f"    Feasible, Compact:       {len(compact_records):>6} records")

    print(f"\n  Grouping 2: Route Count (feasible only)")
    print(f"    Median route utilization: {median_route_util:.4f}")
    print(f"    Many Routes:             {len(many_routes):>6} records")
    print(f"    Few Routes:              {len(few_routes):>6} records")

    print(f"\n  Grouping 3: Route Balance (feasible only)")
    print(f"    Median route length std:  {median_route_std:.4f}")
    print(f"    Imbalanced Routes:       {len(imbalanced):>6} records")
    print(f"    Balanced Routes:         {len(balanced):>6} records")

    all_groupings = {
        'feasibility': grouping_feasibility,
        'route_count': grouping_route_count,
        'route_balance': grouping_route_balance,
    }

    return all_groupings


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def compute_archetype_distributions(
    archetypes: Dict[str, List[Dict]],
    num_actions: int,
    operator_names: List[str],
) -> Dict[str, Dict]:
    """Compute per-archetype mean/std of probability distributions and empirical counts.

    Args:
        archetypes: Dict mapping archetype name -> list of records
        num_actions: Number of actions/operators
        operator_names: List of operator names

    Returns:
        Dict mapping archetype name -> analysis dict
    """
    results = {}

    for archetype_name, records in archetypes.items():
        if not records:
            results[archetype_name] = {
                'count': 0,
                'mean_probs': [0.0] * num_actions,
                'std_probs': [0.0] * num_actions,
                'empirical_counts': [0] * num_actions,
                'empirical_fractions': [0.0] * num_actions,
                'mean_features': [0.0] * len(SOLUTION_FEATURE_NAMES),
            }
            continue

        # Action probability distributions
        all_probs = np.array([r['action_probs'] for r in records])
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)

        # Empirical action counts
        empirical_counts = np.zeros(num_actions, dtype=int)
        for record in records:
            empirical_counts[record['action']] += 1
        empirical_fractions = empirical_counts / len(records)

        # Mean solution features (first 14 only)
        all_features = np.array([r['state_features'][:len(SOLUTION_FEATURE_NAMES)]
                                 for r in records])
        mean_features = np.mean(all_features, axis=0)

        results[archetype_name] = {
            'count': len(records),
            'mean_probs': mean_probs.tolist(),
            'std_probs': std_probs.tolist(),
            'empirical_counts': empirical_counts.tolist(),
            'empirical_fractions': empirical_fractions.tolist(),
            'mean_features': mean_features.tolist(),
        }

    return results


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def compute_statistical_tests(
    archetypes: Dict[str, List[Dict]],
    distributions: Dict[str, Dict],
    num_actions: int,
    operator_names: List[str],
) -> Dict[str, Dict]:
    """Compute pairwise statistical tests between archetype distributions.

    Tests:
      - Jensen-Shannon Divergence between mean probability distributions
      - Per-operator Mann-Whitney U test on probability values
      - Chi-squared test on empirical action counts

    Args:
        archetypes: Dict mapping archetype name -> list of records
        distributions: Output of compute_archetype_distributions()
        num_actions: Number of actions
        operator_names: List of operator names

    Returns:
        Dict of pairwise comparison results
    """
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}

    archetype_names = [name for name in archetypes if archetypes[name]]
    comparisons = {}

    for i in range(len(archetype_names)):
        for j in range(i + 1, len(archetype_names)):
            name_a = archetype_names[i]
            name_b = archetype_names[j]
            key = f"{name_a} vs {name_b}"

            dist_a = distributions[name_a]
            dist_b = distributions[name_b]

            comparison = {}

            # Jensen-Shannon Divergence
            mean_probs_a = np.array(dist_a['mean_probs'])
            mean_probs_b = np.array(dist_b['mean_probs'])

            # Ensure valid probability distributions (no zeros for JSD)
            eps = 1e-10
            mean_probs_a_safe = np.clip(mean_probs_a, eps, None)
            mean_probs_a_safe /= mean_probs_a_safe.sum()
            mean_probs_b_safe = np.clip(mean_probs_b, eps, None)
            mean_probs_b_safe /= mean_probs_b_safe.sum()

            jsd = float(jensenshannon(mean_probs_a_safe, mean_probs_b_safe))
            comparison['jensen_shannon_divergence'] = jsd

            # Per-operator Mann-Whitney U test on probability values
            records_a = archetypes[name_a]
            records_b = archetypes[name_b]
            probs_a = np.array([r['action_probs'] for r in records_a])
            probs_b = np.array([r['action_probs'] for r in records_b])

            mann_whitney_results = {}
            for op_idx in range(num_actions):
                op_name = operator_names[op_idx]
                try:
                    stat, pval = mannwhitneyu(
                        probs_a[:, op_idx],
                        probs_b[:, op_idx],
                        alternative='two-sided',
                    )
                    mann_whitney_results[op_name] = {
                        'U_statistic': float(stat),
                        'p_value': float(pval),
                        'significant_0.05': bool(pval < 0.05),
                        'significant_0.01': bool(pval < 0.01),
                    }
                except Exception as e:
                    mann_whitney_results[op_name] = {'error': str(e)}

            comparison['mann_whitney_per_operator'] = mann_whitney_results

            # Chi-squared test on empirical action counts
            counts_a = np.array(dist_a['empirical_counts'])
            counts_b = np.array(dist_b['empirical_counts'])

            # Only include operators with nonzero counts in at least one archetype
            mask = (counts_a + counts_b) > 0
            if mask.sum() >= 2:
                contingency = np.array([counts_a[mask], counts_b[mask]])
                try:
                    chi2, chi2_p, dof, expected = chi2_contingency(contingency)
                    comparison['chi_squared'] = {
                        'chi2_statistic': float(chi2),
                        'p_value': float(chi2_p),
                        'degrees_of_freedom': int(dof),
                        'significant_0.05': bool(chi2_p < 0.05),
                        'significant_0.01': bool(chi2_p < 0.01),
                    }
                except Exception as e:
                    comparison['chi_squared'] = {'error': str(e)}
            else:
                comparison['chi_squared'] = {'error': 'insufficient non-zero categories'}

            comparisons[key] = comparison

    return comparisons


# ============================================================================
# OUTPUT
# ============================================================================

def print_summary_table(
    distributions: Dict[str, Dict],
    operator_names: List[str],
    num_actions: int,
    statistical_tests: Dict[str, Dict],
) -> None:
    """Print formatted console summary showing probability shifts across archetypes.

    Args:
        distributions: Output of compute_archetype_distributions()
        operator_names: List of operator names
        num_actions: Number of actions
        statistical_tests: Output of compute_statistical_tests()
    """
    archetype_names = [name for name in distributions if distributions[name]['count'] > 0]

    print("\n" + "=" * 100)
    print("STATE ARCHETYPE ACTION PROBABILITY DISTRIBUTIONS")
    print("=" * 100)

    # Header
    header = f"{'Operator':<25}"
    for name in archetype_names:
        count = distributions[name]['count']
        header += f" | {name} (n={count})"
    print(header)
    print("-" * 100)

    # Probability rows
    print("--- Mean Action Probabilities (from softmax of Q-values) ---")
    for op_idx in range(num_actions):
        op_name = operator_names[op_idx]
        row = f"{op_name:<25}"
        for name in archetype_names:
            mean_p = distributions[name]['mean_probs'][op_idx]
            std_p = distributions[name]['std_probs'][op_idx]
            row += f" | {mean_p:>8.4f} +/- {std_p:.4f}    "
        print(row)

    print()
    print("--- Empirical Action Fractions (what the agent actually picked) ---")
    for op_idx in range(num_actions):
        op_name = operator_names[op_idx]
        row = f"{op_name:<25}"
        for name in archetype_names:
            frac = distributions[name]['empirical_fractions'][op_idx]
            count = distributions[name]['empirical_counts'][op_idx]
            row += f" | {frac:>8.4f} ({count:>5d})       "
        print(row)

    # Mean features per archetype
    print()
    print("--- Mean Solution Features per Archetype ---")
    for feat_idx, feat_name in enumerate(SOLUTION_FEATURE_NAMES):
        row = f"{feat_name:<25}"
        for name in archetype_names:
            val = distributions[name]['mean_features'][feat_idx]
            row += f" | {val:>8.4f}                  "
        print(row)

    # Statistical tests
    if statistical_tests and 'error' not in statistical_tests:
        print()
        print("--- Statistical Tests (Pairwise) ---")
        for comparison_name, results in statistical_tests.items():
            print(f"\n  {comparison_name}:")
            print(f"    Jensen-Shannon Divergence: {results['jensen_shannon_divergence']:.6f}")

            if 'chi_squared' in results and 'error' not in results['chi_squared']:
                chi2 = results['chi_squared']
                print(f"    Chi-squared: chi2={chi2['chi2_statistic']:.2f}, "
                      f"p={chi2['p_value']:.2e}, dof={chi2['degrees_of_freedom']}, "
                      f"sig@0.05={chi2['significant_0.05']}")

            if 'mann_whitney_per_operator' in results:
                sig_ops = []
                for op_name, mw in results['mann_whitney_per_operator'].items():
                    if 'error' not in mw and mw['significant_0.05']:
                        sig_ops.append(op_name)
                print(f"    Mann-Whitney significant operators (p<0.05): "
                      f"{sig_ops if sig_ops else 'none'}")

    print("\n" + "=" * 100)


def save_csv_table(
    distributions: Dict[str, Dict],
    operator_names: List[str],
    num_actions: int,
    results_dir: str,
    grouping_name: str = "",
) -> None:
    """Save probability table as CSV.

    Args:
        distributions: Output of compute_archetype_distributions()
        operator_names: List of operator names
        num_actions: Number of actions
        results_dir: Output directory
        grouping_name: Name suffix for the file
    """
    rows = []
    archetype_names = [name for name in distributions if distributions[name]['count'] > 0]

    for op_idx in range(num_actions):
        row = {'Operator': operator_names[op_idx]}
        for name in archetype_names:
            row[f'{name}_mean_prob'] = distributions[name]['mean_probs'][op_idx]
            row[f'{name}_std_prob'] = distributions[name]['std_probs'][op_idx]
            row[f'{name}_empirical_frac'] = distributions[name]['empirical_fractions'][op_idx]
            row[f'{name}_empirical_count'] = distributions[name]['empirical_counts'][op_idx]
        rows.append(row)

    suffix = f"_{grouping_name}" if grouping_name else ""
    csv_path = Path(results_dir) / f"state_archetype_probability_table{suffix}.csv"
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV table to: {csv_path}")


def save_full_results(
    all_distributions: Dict[str, Dict[str, Dict]],
    all_statistical_tests: Dict[str, Dict[str, Dict]],
    operator_names: List[str],
    config: Dict,
    results_dir: str,
) -> None:
    """Save full results as JSON.

    Args:
        all_distributions: Dict mapping grouping name -> distributions
        all_statistical_tests: Dict mapping grouping name -> statistical tests
        operator_names: List of operator names
        config: Experiment configuration
        results_dir: Output directory
    """
    full_results = {
        'experiment': 'State Archetype Action Distributions',
        'config': config,
        'operator_names': operator_names,
        'groupings': {},
    }

    for grouping_name in all_distributions:
        full_results['groupings'][grouping_name] = {
            'distributions': all_distributions[grouping_name],
            'statistical_tests': all_statistical_tests.get(grouping_name, {}),
        }

    json_path = Path(results_dir) / "state_archetype_full_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full results to: {json_path}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_experiment():
    """Main orchestration for the state archetype action distribution experiment."""
    print("=" * 80)
    print("STATE ARCHETYPE ACTION DISTRIBUTION EXPERIMENT")
    print("=" * 80)
    print(f"Problem sizes: {PROBLEM_SIZES}")
    print(f"Runs per instance: {NUM_RUNS}")
    print(f"Initialization: {INIT_TYPE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Softmax temperature: {SOFTMAX_TEMPERATURE}")
    print("=" * 80)

    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)

    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load trained model
    print(f"\nLoading trained model from {MODEL_PATH}...")
    rl_local_search = RLLocalSearch.load_from_checkpoint(MODEL_PATH, verbose=True)
    rl_local_search.tracking = True

    # Get operator info
    num_operators = len(rl_local_search.operators)
    operator_names = [
        op.name if hasattr(op, 'name') else type(op).__name__
        for op in rl_local_search.operators
    ]

    print(f"\nNumber of operators: {num_operators}")
    print(f"Operator names: {operator_names}")

    # Initialize instance managers
    li_lim_manager = LiLimInstanceManager()
    mendeley_manager = MendeleyInstanceManager()

    # Get all instances
    all_instances = []
    for size in PROBLEM_SIZES:
        all_instances.extend(li_lim_manager.get_all(size=size))
        all_instances.extend(mendeley_manager.get_all(size=size))

    print(f"Total instances: {len(all_instances)}")

    # Initialize solution generator
    if INIT_TYPE == "random":
        generator = RandomGenerator()
    elif INIT_TYPE == "greedy":
        generator = GreedyGenerator()
    else:
        raise ValueError(f"Unknown initialization type: {INIT_TYPE}")

    # ---- Phase 1: Data Collection ----
    print("\n" + "=" * 80)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 80)

    start_time = time.time()
    records = collect_state_data(
        rl_local_search=rl_local_search,
        all_instances=all_instances,
        generator=generator,
        num_runs=NUM_RUNS,
        seed=SEED,
    )
    collection_time = time.time() - start_time

    print(f"\nCollected {len(records)} state-action records in {collection_time:.1f}s")

    # Compute Q-values and softmax probabilities
    print("\nComputing Q-values and softmax probabilities...")
    start_time = time.time()
    compute_q_values_for_records(
        agent=rl_local_search.agent,
        records=records,
        temperature=SOFTMAX_TEMPERATURE,
    )
    qval_time = time.time() - start_time
    print(f"Q-value computation done in {qval_time:.1f}s")

    # ---- Phase 2: Classification & Analysis ----
    print("\n" + "=" * 80)
    print("PHASE 2: CLASSIFICATION & ANALYSIS")
    print("=" * 80)

    all_groupings = classify_archetypes(records)

    all_distributions = {}
    all_statistical_tests = {}

    for grouping_name, archetypes in all_groupings.items():
        print(f"\n--- Analyzing grouping: {grouping_name} ---")

        distributions = compute_archetype_distributions(
            archetypes=archetypes,
            num_actions=num_operators,
            operator_names=operator_names,
        )
        all_distributions[grouping_name] = distributions

        print(f"  Running statistical tests for {grouping_name}...")
        statistical_tests = compute_statistical_tests(
            archetypes=archetypes,
            distributions=distributions,
            num_actions=num_operators,
            operator_names=operator_names,
        )
        all_statistical_tests[grouping_name] = statistical_tests

    # ---- Phase 3: Output ----
    print("\n" + "=" * 80)
    print("PHASE 3: OUTPUT")
    print("=" * 80)

    for grouping_name, archetypes in all_groupings.items():
        distributions = all_distributions[grouping_name]
        statistical_tests = all_statistical_tests[grouping_name]

        # Console summary
        print(f"\n{'='*80}")
        print(f"  GROUPING: {grouping_name}")
        print(f"{'='*80}")

        print_summary_table(
            distributions=distributions,
            operator_names=operator_names,
            num_actions=num_operators,
            statistical_tests=statistical_tests,
        )

        # CSV per grouping
        save_csv_table(
            distributions=distributions,
            operator_names=operator_names,
            num_actions=num_operators,
            results_dir=RESULTS_DIR,
            grouping_name=grouping_name,
        )

    # Full JSON with all groupings
    config = {
        'problem_sizes': PROBLEM_SIZES,
        'num_runs': NUM_RUNS,
        'init_type': INIT_TYPE,
        'model_path': MODEL_PATH,
        'seed': SEED,
        'softmax_temperature': SOFTMAX_TEMPERATURE,
        'total_records': len(records),
        'total_instances': len(all_instances),
    }

    save_full_results(
        all_distributions=all_distributions,
        all_statistical_tests=all_statistical_tests,
        operator_names=operator_names,
        config=config,
        results_dir=RESULTS_DIR,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)

    return all_distributions, all_statistical_tests


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        distributions, tests = run_experiment()
        print("\nAll results successfully saved!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to {RESULTS_DIR}")

    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be saved to {RESULTS_DIR}")
