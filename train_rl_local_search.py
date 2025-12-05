"""Example script for training and using RL-based local search."""

import sys
import os
import argparse
import random
import time
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
from memetic.local_search.rl_local_search.dqn_network import detect_attention_from_checkpoint
from memetic.local_search.rl_local_search.ppo_network import detect_ppo_from_checkpoint
from memetic.local_search.adaptive_local_search import AdaptiveLocalSearch
from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.local_search.random_local_search import RandomLocalSearch
from memetic.solution_generators.random_generator import RandomGenerator

# Import config system
from config.config_loader import load_config
from config.operator_factory import create_operators_from_config

# fitness function for initial solution evaluation
from memetic.fitness.fitness import fitness

# Import operators
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.shift import ShiftOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
from memetic.solution_operators.two_opt import TwoOptOperator


# Operator creation moved to config system
# Use create_operators_from_config() with YAML configuration


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
        categories = list(instance_manager.CATEGORIES.keys())

    # If instance_subset is provided, use only those instances
    if instance_subset is None:
        available_instances = {cat: instance_manager.CATEGORIES[cat] for cat in categories}
    else:
        available_instances = instance_subset

    def generator():
        # Randomly select category and instance from available subset
        category = random.choice(list(available_instances.keys()))
        instance_name = random.choice(available_instances[category])
        # print(f"Loading instance: {instance_name} from category: {category}")
        problem = instance_manager.load(instance_name, size)

        if return_name:
            return problem, instance_name
        else:
            return problem

    return generator


def split_instances_by_ratio(categories: list[str], train_ratio: float) -> tuple[dict, dict]:
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
        all_instances = instance_manager.CATEGORIES[category]

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


def get_validation_instance_names(size: int, num_instances: int) -> list[str]:
    """Get fixed list of validation instance names for reproducible evaluation.

    Args:
        size: Problem size (100, 200, 400, 600, 1000)
        num_instances: Number of validation instances to return

    Returns:
        List of instance names from different categories
    """
    # Fixed validation instances per category (diverse selection)
    validation_instances = {
        'lc1': ['lc101', 'lc102', 'lc103', 'lc104', 'lc105'],
        'lc2': ['lc201', 'lc202', 'lc203', 'lc204', 'lc205'],
        'lr1': ['lr101', 'lr102', 'lr103', 'lr104', 'lr105'],
        'lr2': ['lr201', 'lr202', 'lr203', 'lr204', 'lr205'],
        'lrc1': ['lrc101', 'lrc102', 'lrc103', 'lrc104', 'lrc105'],
        'lrc2': ['lrc201', 'lrc202', 'lrc203', 'lrc204', 'lrc205'],
    }

    # Collect instances in round-robin fashion for diversity
    selected = []
    categories = list(validation_instances.keys())
    category_idx = 0
    instance_idx = 0

    while len(selected) < num_instances:
        category = categories[category_idx]
        instances = validation_instances[category]

        if instance_idx < len(instances):
            selected.append(instances[instance_idx])

        category_idx = (category_idx + 1) % len(categories)
        if category_idx == 0:
            instance_idx += 1

    return selected[:num_instances]


def create_validation_set(
    instance_manager: LiLimInstanceManager,
    solution_generator: callable,
    size: int,
    mode: str = "fixed_benchmark",
    num_instances: int = 10,
    seed: int = 42,
    problem_generator: callable = None
):
    """Create validation set for evaluating RL local search.

    Args:
        instance_manager: LiLimInstanceManager for loading problems
        solution_generator: Function that generates initial solutions
        size: Problem size (100, 200, 400, 600, 1000)
        mode: "fixed_benchmark" (recommended) or "random_sampled"
        num_instances: Number of validation instances
        seed: Random seed for reproducibility
        problem_generator: Required if mode="random_sampled"

    Returns:
        List of (problem, initial_solution) tuples
    """
    validation_set = []

    if mode == "fixed_benchmark":
        # Use fixed benchmark instances (best for reproducibility)
        instance_names = get_validation_instance_names(size, num_instances)

        for name in instance_names:
            try:
                problem = instance_manager.load(name, size)
                # Use fixed seed for consistent initial solutions
                set_seed(seed + hash(name) % 1000)
                solution = solution_generator(problem)
                validation_set.append((problem, solution))
            except Exception as e:
                print(f"Warning: Could not load validation instance {name}: {e}")

    elif mode == "random_sampled":
        # Random sampling with fixed seed
        if problem_generator is None:
            raise ValueError("problem_generator required for random_sampled mode")

        set_seed(seed)
        for i in range(num_instances):
            problem = problem_generator()
            solution = solution_generator(problem)
            validation_set.append((problem, solution))

    else:
        raise ValueError(f"Unknown validation mode: {mode}")

    return validation_set


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


def format_path(template: str, **kwargs) -> str:
    """Format path template with runtime values.

    Args:
        template: Path template string with {placeholders}
        **kwargs: Key-value pairs for placeholder replacement

    Returns:
        Formatted path string
    """
    return template.format(**kwargs)


def main():
    """Main training script for RL local search."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RL-based local search using YAML configuration")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--rl_algorithm", type=str, choices=["dqn", "ppo"],
                        help="RL algorithm (overrides config)")
    parser.add_argument("--seed", type=int,
                        help="Random seed (overrides config)")
    parser.add_argument("--use_operator_attention", action="store_true",
                        help="Enable operator attention (overrides config)")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip testing after training")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Alias for --train_only (deprecated)")

    args = parser.parse_args()

    # Load configuration from YAML
    print(f"Loading configuration from: {args.config}")
    cli_overrides = {}
    if args.seed is not None:
        cli_overrides['seed'] = args.seed
    if args.use_operator_attention:
        cli_overrides['use_operator_attention'] = True

    config = load_config(args.config, cli_overrides=cli_overrides)

    # Extract configuration values
    PROBLEM_SIZE = config.problem['size']
    CATEGORIES = config.problem['categories']
    RL_ALGORITHM = args.rl_algorithm if args.rl_algorithm else config.rl_algorithm
    ACCEPTANCE_STRATEGY = config.algorithm['acceptance_strategy']
    REWARD_STRATEGY = config.algorithm['reward_strategy']
    SEED = args.seed if args.seed is not None else 100
    USE_OPERATOR_ATTENTION = args.use_operator_attention or config.get('use_operator_attention', False)

    # Set random seed if provided
    if SEED is not None:
        print(f"Setting random seed to {SEED}")
        set_seed(SEED)

    # Build run name
    seed_suffix = f"_seed{SEED}" if SEED is not None else ""
    attention_suffix = "_attention" if USE_OPERATOR_ATTENTION else ""
    SUFFIX = config.paths.get('suffix', '')
    suffix_str = f"_{SUFFIX}" if SUFFIX else ""
    RUN_NAME = f"rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{attention_suffix}{seed_suffix}{suffix_str}_{int(time.time())}"

    # Setup logging to file
    log_dir = config.paths.get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_{RUN_NAME}.log")

    # Redirect stdout and stderr to both console and log file
    tee_logger = TeeLogger(log_filename)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"Logging to: {log_filename}")
    print("=" * 80)

    # Validate argument combinations and handle test/train flags
    skip_testing = args.train_only or args.skip_testing
    test_only = False  # No longer supported, use train_only instead

    if skip_testing:
        print("\n*** Training only mode - Testing will be skipped ***\n")

    if not test_only:
        # Create operators from configuration
        operators = create_operators_from_config(config.operators)
        print(f"Created {len(operators)} operators from config (preset: {config.operators.get('preset', 'custom')})")

        # Initialize RL local search with config values
        # Use different batch sizes for DQN vs PPO
        batch_size = config.dqn['batch_size'] if RL_ALGORITHM == "dqn" else config.ppo['batch_size']

        rl_local_search = RLLocalSearch(
            operators=operators,
            rl_algorithm=RL_ALGORITHM,
            hidden_dims=config.network['hidden_dims'],
            learning_rate=config.network['learning_rate'],
            gamma=config.network['gamma'],
            # DQN-specific parameters
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
            # PPO-specific parameters
            ppo_clip_epsilon=config.ppo['clip_epsilon'],
            ppo_entropy_coef=config.ppo['entropy_coef'],
            ppo_num_epochs=config.ppo['num_epochs'],
            ppo_num_minibatches=config.ppo['num_minibatches'],
            # Common parameters
            alpha=config.training['alpha'],
            beta=config.training['beta'],
            acceptance_strategy=ACCEPTANCE_STRATEGY,
            reward_strategy=REWARD_STRATEGY,
            max_iterations=config.training['max_iterations'],
            max_no_improvement=config.training['max_no_improvement'],
            use_operator_attention=USE_OPERATOR_ATTENTION,
            device="cuda",
            verbose=True
        )

    # Create train/test split
    train_instances, test_instances = split_instances_by_ratio(
        categories=CATEGORIES,
        train_ratio=config.problem['train_ratio']
    )

    # Log the train/test split
    print(f"\nTrain/Test Split (ratio={config.problem['train_ratio']}):")
    for category in CATEGORIES:
        print(f"  {category}:")
        print(f"    Train ({len(train_instances[category])}): {train_instances[category]}")
        print(f"    Test  ({len(test_instances[category])}): {test_instances[category]}")
    print()

    # Create problem and solution generators
    # Training uses only train_instances
    problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=train_instances
    )

    # Testing uses only test_instances and returns instance names for tracking
    test_problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=test_instances,
        return_name=True
    )

    if not test_only:
        # Create validation set (unless disabled)
        if not config.validation['skip_validation']:
            instance_manager = LiLimInstanceManager()
            validation_set = create_validation_set(
                instance_manager=instance_manager,
                solution_generator=create_solution_generator,
                size=PROBLEM_SIZE,
                mode=config.validation['mode'],
                num_instances=config.validation['num_instances'],
                seed=SEED if SEED is not None else 42,
                problem_generator=problem_generator
            )

            total_validation_runs = len(config.validation['seeds']) * config.validation['runs_per_seed']
            print(f"\nValidation Set:")
            print(f"  Mode: {config.validation['mode']}")
            print(f"  Instances: {len(validation_set)}")
            print(f"  Seeds: {config.validation['seeds']}")
            print(f"  Runs per seed: {config.validation['runs_per_seed']}")
            print(f"  Total runs per instance: {total_validation_runs}")
            print(f"  Interval: {config.validation['interval']} episodes\n")
        else:
            validation_set = None
            print(f"\nValidation: DISABLED\n")

        # Train the RL agent
        print(f"Starting RL Local Search Training on size {PROBLEM_SIZE} instances...")
        print(f"Algorithm: {RL_ALGORITHM.upper()}")
        print(f"Categories: {CATEGORIES}")
        print(f"Operator Attention: {'ENABLED' if USE_OPERATOR_ATTENTION else 'DISABLED'}")

        # Format paths with runtime values
        save_path = format_path(
            config.paths['save_path'],
            algorithm=RL_ALGORITHM,
            problem_size=PROBLEM_SIZE,
            acceptance=ACCEPTANCE_STRATEGY,
            reward=REWARD_STRATEGY,
            attention=attention_suffix,
            seed=SEED,
            suffix=SUFFIX
        )
        tensorboard_dir = format_path(
            config.paths['tensorboard_dir'],
            run_name=RUN_NAME
        )

        training_history = rl_local_search.train(
            problem_generator=problem_generator,
            initial_solution_generator=create_solution_generator,
            num_episodes=config.training['num_episodes'],
            new_instance_interval=config.training['new_instance_interval'],
            new_solution_interval=config.training['new_solution_interval'],
            update_interval=config.training['update_interval'],
            warmup_episodes=config.training['warmup_episodes'],
            save_interval=config.training['save_interval'],
            save_path=save_path,
            tensorboard_dir=tensorboard_dir,
            seed=SEED,
            validation_set=validation_set,
            validation_interval=config.validation['interval'],
            validation_seeds=config.validation['seeds'],
            validation_runs_per_seed=config.validation['runs_per_seed']
        )

        print("\nTraining completed!")
        print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
        print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")
    else:
        print("\nStarting evaluation on test instances...")

    # Skip testing if requested
    if skip_testing:
        print("\n" + "="*80)
        print("TESTING PHASE: SKIPPED")
        print("="*80)
        print("\nTraining completed. Exiting without testing/evaluation.")
        return

    # Create baseline methods with independent operator instances from config
    baseline_operators = create_operators_from_config(config.operators)
    max_iterations = config.training['max_iterations']
    max_no_improvement = config.training['max_no_improvement']

    adaptive_local_search = AdaptiveLocalSearch(
        operators=create_operators_from_config(config.operators),
        max_no_improvement=max_no_improvement,
        max_iterations=max_iterations
    )

    naive_local_search = NaiveLocalSearch(
        operators=create_operators_from_config(config.operators),
        max_no_improvement=max_no_improvement,
        max_iterations=max_iterations,
        first_improvement=True,
        random_operator_order=True
    )

    naive_with_best_local_search = NaiveLocalSearch(
        operators=create_operators_from_config(config.operators),
        max_no_improvement=max_no_improvement,
        max_iterations=max_iterations,
        first_improvement=False
    )

    random_local_search = RandomLocalSearch(
        operators=create_operators_from_config(config.operators),
        max_no_improvement=max_no_improvement,
        max_iterations=max_iterations
    )

    # You can provide up to 6 different RL model paths to evaluate here
    # Use formatted save_path from config
    RL_MODEL_PATHS = [
        f"{save_path}_final.pt" if not test_only else format_path(
            config.paths['save_path'],
            algorithm=RL_ALGORITHM,
            problem_size=PROBLEM_SIZE,
            acceptance=ACCEPTANCE_STRATEGY,
            reward=REWARD_STRATEGY,
            attention=attention_suffix,
            seed=SEED,
            suffix=SUFFIX
        ) + "_final.pt",
    ]

    # Configure seeds for testing from config
    TEST_SEEDS = config.testing['test_seeds']
    # Total number of test evaluations = num_test_problems * runs_per_problem per seed

    # Build RLLocalSearch instances for each provided path and try to load them.
    rl_models = []
    model_names = []
    for path in RL_MODEL_PATHS:
        if not path:
            continue

        # Auto-detect architecture from checkpoint
        try:
            # Detect algorithm type (DQN or PPO)
            model_is_ppo = detect_ppo_from_checkpoint(path)
            model_algorithm = "ppo" if model_is_ppo else "dqn"

            # Detect attention mechanism (only for DQN models)
            if not model_is_ppo:
                model_uses_attention = detect_attention_from_checkpoint(path)
            else:
                # For PPO, we need to check the network architecture
                # For now, assume same as training configuration
                model_uses_attention = USE_OPERATOR_ATTENTION

            arch_str = f"{model_algorithm.upper()}, {'with' if model_uses_attention else 'without'} attention"
            print(f"Detected model architecture ({arch_str}): {path}")
        except Exception as e:
            print(f"Warning: could not detect architecture for {path}: {e}")
            continue

        # Create models with config values for testing
        # OneShot model (same architecture as saved model)
        model_oneshot = RLLocalSearch(
            operators=create_operators_from_config(config.operators),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network['hidden_dims'],
            learning_rate=config.network['learning_rate'],
            gamma=config.network['gamma'],
            epsilon_start=config.dqn['epsilon_start'],
            epsilon_end=config.dqn['epsilon_end'],
            epsilon_decay=config.dqn['epsilon_decay'],
            target_update_interval=config.dqn['target_update_interval'],
            alpha=config.training['alpha'],
            beta=config.training['beta'],
            acceptance_strategy="greedy",
            type="OneShot",
            max_iterations=config.training['max_iterations'],
            max_no_improvement=config.training['max_no_improvement'],
            replay_buffer_capacity=config.dqn['replay_buffer_capacity'],
            batch_size=config.dqn['batch_size'],
            n_step=config.dqn['n_step'],
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            device="cuda",
            verbose=False
        )
        # Roulette model (probabilistic sampling based on Q-values/policy logits)
        model_roulette = RLLocalSearch(
            operators=create_operators_from_config(config.operators),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network['hidden_dims'],
            learning_rate=config.network['learning_rate'],
            gamma=config.network['gamma'],
            epsilon_start=config.dqn['epsilon_start'],
            epsilon_end=config.dqn['epsilon_end'],
            epsilon_decay=config.dqn['epsilon_decay'],
            target_update_interval=config.dqn['target_update_interval'],
            alpha=config.training['alpha'],
            beta=config.training['beta'],
            acceptance_strategy="greedy",
            type="Roulette",
            max_iterations=config.training['max_iterations'],
            max_no_improvement=config.training['max_no_improvement'],
            replay_buffer_capacity=config.dqn['replay_buffer_capacity'],
            batch_size=config.dqn['batch_size'],
            n_step=config.dqn['n_step'],
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            device="cuda",
            verbose=False
        )
        # Ranking model (strict Q-value/policy order, best first)
        model_ranking = RLLocalSearch(
            operators=create_operators_from_config(config.operators),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network['hidden_dims'],
            learning_rate=config.network['learning_rate'],
            gamma=config.network['gamma'],
            epsilon_start=config.dqn['epsilon_start'],
            epsilon_end=config.dqn['epsilon_end'],
            epsilon_decay=config.dqn['epsilon_decay'],
            target_update_interval=config.dqn['target_update_interval'],
            alpha=config.training['alpha'],
            beta=config.training['beta'],
            acceptance_strategy="greedy",
            type="Ranking",
            max_iterations=config.training['max_iterations'],
            max_no_improvement=config.training['max_no_improvement'],
            replay_buffer_capacity=config.dqn['replay_buffer_capacity'],
            batch_size=config.dqn['batch_size'],
            n_step=config.dqn['n_step'],
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            device="cuda",
            verbose=False
        )
        try:
            model_oneshot.load(path)
            model_roulette.load(path)
            model_ranking.load(path)
            print(f"Loaded RL models (OneShot + Roulette + Ranking) from {path}")
            rl_models.append(model_oneshot)
            rl_models.append(model_roulette)
            rl_models.append(model_ranking)
            base_name = os.path.basename(path)
            model_names.append(f"{base_name} (OneShot)")
            model_names.append(f"{base_name} (Roulette)")
            model_names.append(f"{base_name} (Ranking)")
        except Exception as e:
            print(f"Warning: could not load RL model from {path}: {e}")

    if not rl_models:
        print("No RL models loaded. Aborting evaluation.")
        return

    # Compare RL local search models vs Naive local search and Random local search
    # Results structure: per instance -> per method -> list of (initial, best, time, case_idx, run_idx) tuples
    instance_results = {}

    print("\n--- Comparing RL models vs Naive/Random Local Search ---")
    print(f"Testing Configuration:")
    print(f"  Seeds: {TEST_SEEDS}")
    print(f"  Test problems per seed: {config.testing['num_test_problems']}")
    print(f"  Runs per problem: {config.testing['runs_per_problem']}")
    print(f"  Total evaluations per seed: {config.testing['num_test_problems'] * config.testing['runs_per_problem']}")
    print(f"  Deterministic RNG: {config.testing['deterministic_test_rng']}")
    print()

    for seed_idx, test_seed in enumerate(TEST_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")
        print(f"{'='*80}")

        # Generate test cases (problem + initial solution pairs) upfront for this seed
        print(f"\nGenerating {config.testing['num_test_problems']} test cases...")
        test_cases = []
        for case_idx in range(config.testing['num_test_problems']):
            set_seed(test_seed + case_idx)
            test_problem, instance_name = test_problem_generator()
            initial_solution = create_solution_generator(test_problem)
            initial_fitness = fitness(test_problem, initial_solution)
            test_cases.append((test_problem, initial_solution, initial_fitness, instance_name))

        # Run each test case multiple times
        for case_idx, (test_problem, initial_solution, initial_fitness, instance_name) in enumerate(test_cases):
            print(f"\nTest Case {case_idx+1}/{config.testing['num_test_problems']} - Instance: {instance_name}")
            print(f"  Initial fitness: {initial_fitness:.2f}")

            # Initialize instance tracking if first time seeing this instance
            if instance_name not in instance_results:
                instance_results[instance_name] = {
                    'rl_models': [[] for _ in range(len(rl_models))],
                    'adaptive': [],
                    'naive': [],
                    'naive_with_best': [],
                    'random': []
                }

            for run_idx in range(config.testing['runs_per_problem']):
                if config.testing['runs_per_problem'] > 1:
                    print(f"  Run {run_idx+1}/{config.testing['runs_per_problem']}")

                # Different RNG seed for each run
                run_seed = test_seed + case_idx * 1000 + run_idx

                # Run each RL model
                for midx, model in enumerate(rl_models):
                    set_seed(run_seed)  # Same seed for all methods in this run
                    rl_solution = initial_solution.clone()
                    t0 = time.time()
                    try:
                        rl_best_solution, rl_best_fitness = model.search(
                            problem=test_problem,
                            solution=rl_solution,
                            epsilon=0.0,
                            deterministic_rng=config.testing['deterministic_test_rng'],
                            base_seed=run_seed
                        )
                    except Exception as e:
                        print(f"    Model {model_names[midx]} failed: {e}")
                        rl_best_fitness = float('inf')
                    rl_time = time.time() - t0
                    if config.testing['runs_per_problem'] > 1:
                        print(f"    {model_names[midx]}: {rl_best_fitness:.2f} (time: {rl_time:.2f}s)")
                    else:
                        print(f"  {model_names[midx]}: {rl_best_fitness:.2f} (time: {rl_time:.2f}s)")
                    instance_results[instance_name]['rl_models'][midx].append(
                        (initial_fitness, rl_best_fitness, rl_time, case_idx, run_idx)
                    )

                # Adaptive local search
                set_seed(run_seed)
                adaptive_solution = initial_solution.clone()
                t0 = time.time()
                adaptive_best_solution, adaptive_best_fitness = adaptive_local_search.search(
                    problem=test_problem,
                    solution=adaptive_solution,
                    deterministic_rng=config.testing['deterministic_test_rng'],
                    base_seed=run_seed
                )
                adaptive_time = time.time() - t0
                if config.testing['runs_per_problem'] > 1:
                    print(f"    Adaptive: {adaptive_best_fitness:.2f} (time: {adaptive_time:.2f}s)")
                else:
                    print(f"  Adaptive: {adaptive_best_fitness:.2f} (time: {adaptive_time:.2f}s)")
                instance_results[instance_name]['adaptive'].append(
                    (initial_fitness, adaptive_best_fitness, adaptive_time, case_idx, run_idx)
                )

                # Naive local search
                set_seed(run_seed)
                naive_solution = initial_solution.clone()
                t0 = time.time()
                naive_best_solution, naive_best_fitness = naive_local_search.search(
                    problem=test_problem,
                    solution=naive_solution,
                    deterministic_rng=config.testing['deterministic_test_rng'],
                    base_seed=run_seed
                )
                naive_time = time.time() - t0
                if config.testing['runs_per_problem'] > 1:
                    print(f"    Naive: {naive_best_fitness:.2f} (time: {naive_time:.2f}s)")
                else:
                    print(f"  Naive: {naive_best_fitness:.2f} (time: {naive_time:.2f}s)")
                instance_results[instance_name]['naive'].append(
                    (initial_fitness, naive_best_fitness, naive_time, case_idx, run_idx)
                )

                # Naive local search with best improvement
                set_seed(run_seed)
                naive_with_best_solution = initial_solution.clone()
                t0 = time.time()
                naive_with_best_best_solution, naive_with_best_best_fitness = naive_with_best_local_search.search(
                    problem=test_problem,
                    solution=naive_with_best_solution,
                    deterministic_rng=config.testing['deterministic_test_rng'],
                    base_seed=run_seed
                )
                naive_with_best_time = time.time() - t0
                if config.testing['runs_per_problem'] > 1:
                    print(f"    Naive (best): {naive_with_best_best_fitness:.2f} (time: {naive_with_best_time:.2f}s)")
                else:
                    print(f"  Naive (best): {naive_with_best_best_fitness:.2f} (time: {naive_with_best_time:.2f}s)")
                instance_results[instance_name]['naive_with_best'].append(
                    (initial_fitness, naive_with_best_best_fitness, naive_with_best_time, case_idx, run_idx)
                )

                # Random local search
                set_seed(run_seed)
                random_solution = initial_solution.clone()
                t0 = time.time()
                random_best_solution, random_best_fitness = random_local_search.search(
                    problem=test_problem,
                    solution=random_solution,
                    deterministic_rng=config.testing['deterministic_test_rng'],
                    base_seed=run_seed
                )
                random_time = time.time() - t0
                if config.testing['runs_per_problem'] > 1:
                    print(f"    Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                else:
                    print(f"  Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                instance_results[instance_name]['random'].append(
                    (initial_fitness, random_best_fitness, random_time, case_idx, run_idx)
                )

        # Print brief summary for this seed
        total_evals = config.testing['num_test_problems'] * config.testing['runs_per_problem']
        print(f"\n--- Completed Seed {test_seed}: {total_evals} evaluations across {len(test_cases)} test cases ---")

    # Per-instance summary
    print(f"\n{'='*80}")
    print("PER-INSTANCE PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")

    for instance_name in sorted(instance_results.keys()):
        results = instance_results[instance_name]

        # Calculate number of unique test cases and total tests for this instance
        if results['adaptive']:
            num_tests = len(results['adaptive'])
            # Count unique case indices to get number of unique test cases
            unique_cases = len(set(r[3] for r in results['adaptive']))
            runs_per_case = config.testing['runs_per_problem']
        else:
            num_tests = 0
            unique_cases = 0
            runs_per_case = 0

        if num_tests == 0:
            continue

        print(f"--- {instance_name} ---")
        print(f"  Test cases: {unique_cases}, Runs per case: {runs_per_case}, Total tests: {num_tests}")

        # Average initial fitness
        avg_initial = np.mean([r[0] for r in results['adaptive']])
        std_initial = np.std([r[0] for r in results['adaptive']])
        print(f"  Avg Initial: {avg_initial:.2f} ± {std_initial:.2f}")

        if runs_per_case > 1:
            # Show within-solution variance
            print(f"\n  Method Performance (mean ± overall_std, within-solution std):")

            # RL models
            for midx, name in enumerate(model_names):
                all_results = results['rl_models'][midx]
                avg_best = np.mean([r[1] for r in all_results])
                std_best = np.std([r[1] for r in all_results])

                # Within-solution variance (how much method varies on same initial solution)
                case_variances = []
                for case_idx in range(unique_cases):
                    case_results = [r[1] for r in all_results if r[3] == case_idx]
                    if len(case_results) > 1:
                        case_variances.append(np.std(case_results))

                avg_within_std = np.mean(case_variances) if case_variances else 0.0
                avg_improvement = np.mean([r[0] - r[1] for r in all_results])

                print(f"    {name:45s}: {avg_best:.2f} ± {std_best:.2f} (within: {avg_within_std:.2f}, Δ={avg_improvement:+.2f})")

            # Baselines
            for method_name, method_key in [('Adaptive', 'adaptive'), ('Naive', 'naive'),
                                             ('Naive (best)', 'naive_with_best'), ('Random', 'random')]:
                all_results = results[method_key]
                avg_best = np.mean([r[1] for r in all_results])
                std_best = np.std([r[1] for r in all_results])

                case_variances = []
                for case_idx in range(unique_cases):
                    case_results = [r[1] for r in all_results if r[3] == case_idx]
                    if len(case_results) > 1:
                        case_variances.append(np.std(case_results))

                avg_within_std = np.mean(case_variances) if case_variances else 0.0
                avg_improvement = np.mean([r[0] - r[1] for r in all_results])

                print(f"    {method_name:45s}: {avg_best:.2f} ± {std_best:.2f} (within: {avg_within_std:.2f}, Δ={avg_improvement:+.2f})")
        else:
            # Simpler output for single runs
            print(f"\n  Method Performance (mean ± std):")

            for midx, name in enumerate(model_names):
                all_results = results['rl_models'][midx]
                avg_best = np.mean([r[1] for r in all_results])
                std_best = np.std([r[1] for r in all_results])
                avg_improvement = np.mean([r[0] - r[1] for r in all_results])
                print(f"    {name:45s}: {avg_best:.2f} ± {std_best:.2f} (Δ={avg_improvement:+.2f})")

            for method_name, method_key in [('Adaptive', 'adaptive'), ('Naive', 'naive'),
                                             ('Naive (best)', 'naive_with_best'), ('Random', 'random')]:
                all_results = results[method_key]
                avg_best = np.mean([r[1] for r in all_results])
                std_best = np.std([r[1] for r in all_results])
                avg_improvement = np.mean([r[0] - r[1] for r in all_results])
                print(f"    {method_name:45s}: {avg_best:.2f} ± {std_best:.2f} (Δ={avg_improvement:+.2f})")

        print()

    # Overall summary across all instances
    print(f"{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL INSTANCES")
    print(f"{'='*80}")

    # Aggregate all results
    all_rl_results = [[] for _ in range(len(rl_models))]
    all_adaptive_results = []
    all_naive_results = []
    all_naive_with_best_results = []
    all_random_results = []

    for instance_name, results in instance_results.items():
        for midx in range(len(rl_models)):
            all_rl_results[midx].extend(results['rl_models'][midx])
        all_adaptive_results.extend(results['adaptive'])
        all_naive_results.extend(results['naive'])
        all_naive_with_best_results.extend(results['naive_with_best'])
        all_random_results.extend(results['random'])

    total_tests = len(all_adaptive_results) if all_adaptive_results else 0
    print(f"Total evaluations: {total_tests}")
    print(f"Unique instances tested: {len(instance_results)}")

    if total_tests > 0:
        # Print overall averages
        avg_initial = np.mean([r[0] for r in all_adaptive_results])
        print(f"\nAverage initial fitness: {avg_initial:.2f}")

        print("\nRL Models:")
        for midx, name in enumerate(model_names):
            avg_best = np.mean([r[1] for r in all_rl_results[midx]])
            avg_time = np.mean([r[2] for r in all_rl_results[midx]])
            std_best = np.std([r[1] for r in all_rl_results[midx]])
            avg_improvement = np.mean([r[0] - r[1] for r in all_rl_results[midx]])
            print(f"  {name}: {avg_best:.2f} ± {std_best:.2f} (Δ={avg_improvement:+.2f}, time: {avg_time:.2f}s)")

        print("\nBaseline Methods:")
        for method_name, method_results in [('Adaptive', all_adaptive_results), ('Naive', all_naive_results),
                                              ('Naive (best)', all_naive_with_best_results), ('Random', all_random_results)]:
            avg_best = np.mean([r[1] for r in method_results])
            avg_time = np.mean([r[2] for r in method_results])
            std_best = np.std([r[1] for r in method_results])
            avg_improvement = np.mean([r[0] - r[1] for r in method_results])
            print(f"  {method_name}: {avg_best:.2f} ± {std_best:.2f} (Δ={avg_improvement:+.2f}, time: {avg_time:.2f}s)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTraining failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original stdout/stderr and close log file
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
