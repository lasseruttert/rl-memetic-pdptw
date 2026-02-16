"""Script for testing stability of RL training across different random seeds."""

import sys
import os
import argparse
import random
import time
import csv
import numpy as np


class TeeLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, filename, mode='w'):
        """Initialize TeeLogger.

        Args:
            filename: Path to log file
            mode: File open mode ('w' for overwrite, 'a' for append)
        """
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')

    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file

    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close log file."""
        self.log.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_instance_manager import LiLimInstanceManager
from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch
from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.solution_generators.random_generator import RandomGenerator

# Import config system
from config.config_loader import load_config
from config.operator_factory import create_operators_from_config

# fitness function for initial solution evaluation
from memetic.fitness.fitness import fitness


def create_problem_generator(size: int = 100, categories: list[str] = None, instance_subset: dict = None, return_name: bool = False):
    """Create a function that generates problem instances using InstanceManager.

    Args:
        size: Problem size (100, 200, 400, 600, 1000)
        categories: List of categories to sample from (default: all)
        instance_subset: Dict mapping category -> list of instance names to use (default: all instances in categories)
        return_name: If True, generator returns (problem, instance_name) tuple instead of just problem

    Returns:
        Generator function that returns random problem instances (or (problem, name) if return_name=True)
    """
    instance_manager = LiLimInstanceManager()

    if categories is None:
        categories = list(instance_manager._get_categories(size).keys())

    # If instance_subset is provided, use only those instances
    if instance_subset is None:
        available_instances = {cat: instance_manager._get_categories(size)[cat] for cat in categories}
    else:
        available_instances = instance_subset

    print(available_instances)

    def generator():
        # Randomly select category and instance from available subset
        category = random.choice(list(available_instances.keys()))
        instance_name = random.choice(available_instances[category])
        problem = instance_manager.load(instance_name, size)

        if return_name:
            return problem, instance_name
        else:
            return problem

    return generator


def split_instances_by_ratio(size: int, categories: list[str], train_ratio: float) -> tuple[dict, dict]:
    """Split instances within each category by ratio.

    Args:
        categories: List of category names to split
        train_ratio: Ratio for training (e.g., 0.7 = 70% train, 30% test)

    Returns:
        (train_instances, test_instances) where each is a dict {category: [instance_names]}
    """
    instance_manager = LiLimInstanceManager()
    train_instances = {}
    test_instances = {}

    for category in categories:
        all_instances = instance_manager._get_categories(size)[category]

        if train_ratio >= 1.0:
            # Use all instances for both training and testing
            train_instances[category] = all_instances
            test_instances[category] = all_instances
        else:
            split_index = int(len(all_instances) * train_ratio)

            # Ensure at least 1 instance for training and testing if possible
            if len(all_instances) > 1:
                split_index = max(1, min(split_index, len(all_instances) - 1))
            else:
                split_index = 1

            train_instances[category] = all_instances[:split_index]
            test_instances[category] = all_instances[split_index:]

    return train_instances, test_instances


def create_solution_generator(problem: PDPTWProblem) -> PDPTWSolution:
    """Generate initial solution for a problem instance."""
    generator = RandomGenerator()
    solution = generator.generate(problem, num_solutions=1)[0]
    return solution

def create_solution_generator_with_ls(problem: PDPTWProblem) -> PDPTWSolution:
    """Generate initial solution for a problem instance."""
    generator = RandomGenerator()
    solution = generator.generate(problem, num_solutions=1)[0]
    num_iterations = random.randint(0, 30)
    from memetic.solution_operators.reinsert import ReinsertOperator
    ls = NaiveLocalSearch(
        operators = [ReinsertOperator()],
        max_iterations = num_iterations,
        max_no_improvement=num_iterations,
    )
    solution, _ = ls.search(problem, solution)
    return solution


def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seeds
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main():
    """Main script for testing RL training stability across seeds."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test RL training stability across multiple random seeds")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--rl_algorithm", type=str, choices=["dqn", "ppo"],
                        help="RL algorithm (overrides config)")
    parser.add_argument("--use_operator_attention", action="store_true",
                        help="Enable operator attention (overrides config)")
    parser.add_argument("--ls_initial_solution", action="store_true",
                        help="Use initial solution with local search")
    parser.add_argument("--num_seed_runs", type=int, default=30,
                        help="Number of different training seeds to test (default: 30)")
    parser.add_argument("--base_seed", type=int, default=100,
                        help="Base seed for generating training seeds (default: 100)")

    args = parser.parse_args()

    # Load configuration from YAML
    print(f"Loading configuration from: {args.config}")
    cli_overrides = {}
    if args.use_operator_attention:
        cli_overrides['use_operator_attention'] = True

    config = load_config(args.config, cli_overrides=cli_overrides)

    # Extract configuration values
    PROBLEM_SIZE = config.problem['size']
    CATEGORIES = config.problem['categories']
    RL_ALGORITHM = args.rl_algorithm if args.rl_algorithm else config.rl_algorithm
    ACCEPTANCE_STRATEGY = config.algorithm['acceptance_strategy']
    REWARD_STRATEGY = config.algorithm['reward_strategy']
    USE_OPERATOR_ATTENTION = args.use_operator_attention or config.get('use_operator_attention', False)

    # Generate list of training seeds
    BASE_SEED = args.base_seed
    NUM_SEED_RUNS = args.num_seed_runs
    TRAINING_SEEDS = [BASE_SEED + i for i in range(NUM_SEED_RUNS)]

    # Build run name
    attention_suffix = "_attention" if USE_OPERATOR_ATTENTION else ""
    SUFFIX = config.paths.get('suffix', '')
    suffix_str = f"_{SUFFIX}" if SUFFIX else ""
    RUN_NAME = f"rl_seed_stability_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{attention_suffix}_base{BASE_SEED}_n{NUM_SEED_RUNS}{suffix_str}_{int(time.time())}"

    # Setup logging to file
    log_dir = config.paths.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"seed_stability_{RUN_NAME}.log")

    # Redirect stdout and stderr to both console and log file
    tee_logger = TeeLogger(log_filename)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"Logging to: {log_filename}")
    print("=" * 80)
    print("SEED STABILITY TESTING")
    print("=" * 80)
    print(f"Testing {NUM_SEED_RUNS} different training seeds: {TRAINING_SEEDS[0]} to {TRAINING_SEEDS[-1]}")
    print(f"Algorithm: {RL_ALGORITHM.upper()}")
    print(f"Problem Size: {PROBLEM_SIZE}")
    print(f"Categories: {CATEGORIES}")
    print(f"Operator Attention: {'ENABLED' if USE_OPERATOR_ATTENTION else 'DISABLED'}")
    print("=" * 80)
    print()

    # One-time setup: Create train/test split (same for all seed runs)
    train_instances, test_instances = split_instances_by_ratio(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        train_ratio=config.problem['train_ratio']
    )

    # Log the train/test split
    print(f"Train/Test Split (ratio={config.problem['train_ratio']}):")
    for category in CATEGORIES:
        print(f"  {category}:")
        print(f"    Train ({len(train_instances[category])}): {train_instances[category]}")
        print(f"    Test  ({len(test_instances[category])}): {test_instances[category]}")
    print()

    # Create problem generators (same for all seed runs)
    problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=train_instances
    )

    test_problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=test_instances,
        return_name=True
    )

    # No baseline methods needed - only testing RL OneShot stability

    # Results structure: training_seed -> list of fitness values
    seed_results = {}

    # Also track per-instance results for detailed analysis
    instance_results = {}

    # Configure testing parameters
    TEST_SEEDS = config.testing['test_seeds']

    print(f"Testing Configuration:")
    print(f"  Test seeds per training run: {TEST_SEEDS}")
    print(f"  Test problems per seed: {config.testing['num_test_problems']}")
    print(f"  Runs per problem: {config.testing['runs_per_problem']}")
    print(f"  Deterministic RNG: {config.testing['deterministic_test_rng']}")
    print()

    # Main seed loop
    for seed_idx, training_seed in enumerate(TRAINING_SEEDS):
        print("\n" + "=" * 80)
        print(f"SEED RUN {seed_idx + 1}/{NUM_SEED_RUNS}: Training Seed = {training_seed}")
        print("=" * 80)
        print()

        # Set seed for this training run
        set_seed(training_seed)

        # Create fresh RL model for this seed
        operators = create_operators_from_config(config.operators)
        batch_size = config.dqn['batch_size'] if RL_ALGORITHM == "dqn" else config.ppo['batch_size']

        rl_local_search = RLLocalSearch(
            operators=operators,
            rl_algorithm=RL_ALGORITHM,
            hidden_dims=config.network['hidden_dims'],
            learning_rate=config.network['learning_rate'],
            gamma=config.network['gamma'],
            epsilon_start=config.dqn['epsilon_start'],
            epsilon_end=config.dqn['epsilon_end'],
            epsilon_decay=config.dqn['epsilon_decay'],
            target_update_interval=config.dqn['target_update_interval'],
            replay_buffer_capacity=config.dqn['replay_buffer_capacity'],
            batch_size=batch_size,
            n_step=config.dqn['n_step'],
            use_prioritized_replay=config.dqn['use_prioritized_replay'],
            per_alpha=config.dqn['per_alpha'],
            per_beta_start=config.dqn['per_beta_start'],
            ppo_clip_epsilon=config.ppo['clip_epsilon'],
            ppo_entropy_coef=config.ppo['entropy_coef'],
            ppo_num_epochs=config.ppo['num_epochs'],
            ppo_num_minibatches=config.ppo['num_minibatches'],
            alpha=config.training['alpha'],
            beta=config.training['beta'],
            acceptance_strategy=ACCEPTANCE_STRATEGY,
            reward_strategy=REWARD_STRATEGY,
            max_iterations=config.training['max_iterations'],
            max_no_improvement=config.training['max_no_improvement'],
            use_operator_attention=USE_OPERATOR_ATTENTION,
            type="OneShot",  # Use OneShot for testing
            verbose=False  # Reduced output for speed
        )

        # Train the RL agent (no validation, no saving)
        print(f"Training RL model with seed {training_seed}...")
        training_history = rl_local_search.train(
            problem_generator=problem_generator,
            initial_solution_generator=create_solution_generator if not args.ls_initial_solution else create_solution_generator_with_ls,
            num_episodes=config.training['num_episodes'],
            new_instance_interval=config.training['new_instance_interval'],
            new_solution_interval=config.training['new_solution_interval'],
            update_interval=config.training['update_interval'],
            warmup_episodes=config.training['warmup_episodes'],
            save_interval=None,  # Don't save model
            save_path=None,  # Don't save model
            tensorboard_dir=None,  # Skip tensorboard
            seed=training_seed,
            validation_set=None,  # Skip validation
            validation_interval=None,
            validation_seeds=None,
            validation_runs_per_seed=None
        )

        print(f"\nTraining completed for seed {training_seed}!")
        print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
        print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")

        # Test the trained model immediately (no save/load)
        print(f"\nTesting trained model (seed {training_seed})...")

        # Initialize seed results
        seed_results[training_seed] = {
            'fitness_values': [],
            'times': [],
            'training_time': training_history.get('total_training_time', 0) if hasattr(training_history, 'get') else 0
        }

        test_start_time = time.time()
        total_tests = 0

        # Test on all test seeds
        for test_seed_idx, test_seed in enumerate(TEST_SEEDS):
            # Generate test cases for this test seed
            for case_idx in range(config.testing['num_test_problems']):
                set_seed(test_seed + case_idx)
                test_problem, instance_name = test_problem_generator()
                initial_solution = create_solution_generator(test_problem)
                initial_fitness = fitness(test_problem, initial_solution)

                # Initialize instance tracking if first time
                if instance_name not in instance_results:
                    instance_results[instance_name] = {}
                if training_seed not in instance_results[instance_name]:
                    instance_results[instance_name][training_seed] = []

                for run_idx in range(config.testing['runs_per_problem']):
                    # Different RNG seed for each run
                    run_seed = test_seed + case_idx * 1000 + run_idx

                    # Test RL model (OneShot only)
                    set_seed(run_seed)
                    rl_solution = initial_solution.clone()
                    t0 = time.time()
                    try:
                        rl_best_solution, rl_best_fitness = rl_local_search.search(
                            problem=test_problem,
                            solution=rl_solution,
                            epsilon=0.0,
                            deterministic_rng=config.testing['deterministic_test_rng'],
                            base_seed=run_seed
                        )
                    except Exception as e:
                        print(f"    RL model failed on {instance_name}: {e}")
                        rl_best_fitness = float('inf')
                    rl_time = time.time() - t0

                    # Store results
                    seed_results[training_seed]['fitness_values'].append(rl_best_fitness)
                    seed_results[training_seed]['times'].append(rl_time)
                    instance_results[instance_name][training_seed].append(
                        (initial_fitness, rl_best_fitness, rl_time)
                    )
                    total_tests += 1

        test_total_time = time.time() - test_start_time
        seed_results[training_seed]['test_time'] = test_total_time
        print(f"  Completed {total_tests} test evaluations in {test_total_time:.2f}s")

        # Print summary for this seed
        fitness_vals = seed_results[training_seed]['fitness_values']
        time_vals = seed_results[training_seed]['times']

        print(f"\nSeed {training_seed} - Fitness: {np.mean(fitness_vals):.2f} ± {np.std(fitness_vals):.2f}, "
              f"Avg Time: {np.mean(time_vals):.3f}s\n")

    # ========================================================================
    # FINAL STABILITY ANALYSIS
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL SEED STABILITY ANALYSIS - RL ONESHOT")
    print("=" * 80)
    print()

    # Calculate statistics across all seeds
    all_seed_means = []
    all_seed_stds = []
    all_seed_times = []

    for training_seed in TRAINING_SEEDS:
        fitness_vals = seed_results[training_seed]['fitness_values']
        time_vals = seed_results[training_seed]['times']
        all_seed_means.append(np.mean(fitness_vals))
        all_seed_stds.append(np.std(fitness_vals))
        all_seed_times.append(np.mean(time_vals))

    # Overall statistics
    overall_mean = np.mean(all_seed_means)
    overall_std = np.std(all_seed_means)
    cv = (overall_std / overall_mean * 100) if overall_mean != 0 else 0
    min_fitness = np.min(all_seed_means)
    max_fitness = np.max(all_seed_means)
    avg_time = np.mean(all_seed_times)

    best_seed_idx = np.argmin(all_seed_means)
    worst_seed_idx = np.argmax(all_seed_means)
    best_seed = TRAINING_SEEDS[best_seed_idx]
    worst_seed = TRAINING_SEEDS[worst_seed_idx]

    print(f"Analyzed {NUM_SEED_RUNS} training seeds: {TRAINING_SEEDS[0]} to {TRAINING_SEEDS[-1]}")
    print(f"Total test evaluations per seed: {len(seed_results[TRAINING_SEEDS[0]]['fitness_values'])}")
    print()
    print(f"Overall Statistics:")
    print(f"  Mean Fitness:      {overall_mean:.2f}")
    print(f"  Std Across Seeds:  {overall_std:.2f}")
    print(f"  CV:                {cv:.2f}%")
    print(f"  Min (best seed):   {min_fitness:.2f} (seed {best_seed})")
    print(f"  Max (worst seed):  {max_fitness:.2f} (seed {worst_seed})")
    print(f"  Range:             {max_fitness - min_fitness:.2f}")
    print(f"  Avg Time per Test: {avg_time:.3f}s")
    print()

    # Save detailed results to CSV
    csv_filename = os.path.join(log_dir, f"seed_stability_{RUN_NAME}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Training_Seed', 'Mean_Fitness', 'Std_Fitness', 'Avg_Time_s', 'Num_Tests'])

        for training_seed in TRAINING_SEEDS:
            fitness_vals = seed_results[training_seed]['fitness_values']
            time_vals = seed_results[training_seed]['times']
            writer.writerow([
                training_seed,
                f"{np.mean(fitness_vals):.4f}",
                f"{np.std(fitness_vals):.4f}",
                f"{np.mean(time_vals):.4f}",
                len(fitness_vals)
            ])

        # Add summary row
        writer.writerow([])
        writer.writerow(['SUMMARY', '', '', '', ''])
        writer.writerow(['Overall_Mean', f"{overall_mean:.4f}", '', '', ''])
        writer.writerow(['Overall_Std', f"{overall_std:.4f}", '', '', ''])
        writer.writerow(['CV_%', f"{cv:.4f}", '', '', ''])
        writer.writerow(['Min_Fitness', f"{min_fitness:.4f}", '', '', ''])
        writer.writerow(['Max_Fitness', f"{max_fitness:.4f}", '', '', ''])
        writer.writerow(['Best_Seed', best_seed, '', '', ''])
        writer.writerow(['Worst_Seed', worst_seed, '', '', ''])

    print(f"Results saved to: {csv_filename}")
    print()
    print("=" * 80)
    print("SEED STABILITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTesting failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original stdout/stderr and close log file
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
