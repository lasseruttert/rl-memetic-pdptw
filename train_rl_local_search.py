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


def create_operators():
    """Create a fresh set of operators with independent state.

    Returns:
        List of operator instances
    """
    return [
        ReinsertOperator(),
        ReinsertOperator(max_attempts=5,clustered=True),
        ReinsertOperator(force_same_vehicle=True),
        ReinsertOperator(allow_same_vehicle=False),
        ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        
        RouteEliminationOperator(),
        
        # FlipOperator(),
        # FlipOperator(max_attempts=5),
        # FlipOperator(single_route=True),
        
        SwapWithinOperator(),
        # SwapWithinOperator(max_attempts=5),
        # SwapWithinOperator(single_route=True),
        # SwapWithinOperator(single_route=True, type="best"),
        # SwapWithinOperator(single_route=False, type="best"),

        SwapBetweenOperator(),
        # SwapBetweenOperator(type="best"),
        
        TransferOperator(),
        # TransferOperator(single_route=True),
        # TransferOperator(max_attempts=5,single_route=True),

        # ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5),
        # ShiftOperator(type="random", segment_length=2, max_shift_distance=4, max_attempts=5),
        # ShiftOperator(type="random", segment_length=4, max_shift_distance=2, max_attempts=3),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=5, max_attempts=3),
        # ShiftOperator(type="best", segment_length=2, max_shift_distance=3),
        # ShiftOperator(type="best", segment_length=3, max_shift_distance=2),
        # ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5, single_route=True),

        TwoOptOperator(),
        
        # CLSM1Operator(),
        # CLSM2Operator(),
        # CLSM3Operator(),
        # CLSM4Operator()
    ]


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


def main():
    """Main training script for RL local search."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RL-based local search with configurable strategies")
    parser.add_argument("--rl_algorithm", type=str, default="dqn",
                        choices=["dqn", "ppo"],
                        help="RL algorithm to use: dqn or ppo (default: dqn)")
    parser.add_argument("--acceptance_strategy", type=str, default="greedy",
                        choices=["greedy", "always", "epsilon_greedy",
                                 "simulated_annealing", "late_acceptance", "rising_epsilon_greedy"],
                        help="Acceptance strategy for local search")
    parser.add_argument("--reward_strategy", type=str, default="binary",
                        choices=["binary", "initial_improvement", "old_improvement", "hybrid_improvement",
                                 "distance_baseline", "log_improvement", "tanh", "component"],
                        help="Reward strategy for RL agent")
    parser.add_argument("--problem_size", type=int, default=100,
                        choices=[100, 200, 400, 600, 1000],
                        help="Problem size")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use_operator_attention", action="store_true",
                        help="Enable operator attention mechanism for operator selection")
    parser.add_argument("--validation_mode", type=str, default="fixed_benchmark",
                        choices=["fixed_benchmark", "random_sampled"],
                        help="Validation set mode: fixed_benchmark (reproducible) or random_sampled (faster)")
    parser.add_argument("--num_validation_instances", type=int, default=10,
                        help="Number of validation instances (default: 10)")
    parser.add_argument("--validation_interval", type=int, default=50,
                        help="Evaluate on validation set every N episodes (default: 50)")
    parser.add_argument("--validation_seeds", type=int, nargs='+', default=[42, 111, 222, 333, 444],
                        help="List of base seeds for validation runs (default: [42, 111, 222, 333, 444])")
    parser.add_argument("--validation_runs_per_seed", type=int, default=1,
                        help="Number of runs per seed for each validation instance (default: 1)")

    # Train/Test split arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of instances to use for training (0.0-1.0, default: 1.0 = use all for both train/test)")
    parser.add_argument("--num_test_problems", type=int, default=50,
                        help="Number of unique test cases (problem + initial solution) per seed (default: 50)")
    parser.add_argument("--runs_per_problem", type=int, default=3,
                        help="Number of runs per test case with different RNG seeds (default: 5)")
    parser.add_argument("--deterministic_test_rng", action="store_true",
                        help="Use deterministic RNG during testing for reproducible results")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run testing/evaluation on existing models")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip validation during training (disables validation set creation)")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Skip testing/evaluation phase after training")

    # PPO-specific arguments
    parser.add_argument("--ppo_batch_size", type=int, default=2048,
                        help="PPO batch size - steps to accumulate before update (default: 2048)")
    parser.add_argument("--ppo_clip_epsilon", type=float, default=0.2,
                        help="PPO clipping parameter (default: 0.2)")
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01,
                        help="PPO entropy coefficient (default: 0.01)")
    parser.add_argument("--ppo_num_epochs", type=int, default=2,
                        help="PPO number of update epochs per trajectory (default: 2)")
    parser.add_argument("--ppo_num_minibatches", type=int, default=2,
                        help="PPO number of minibatches per epoch (default: 2)")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # Configuration
    PROBLEM_SIZE = args.problem_size
    CATEGORIES = ['lc1', 'lc2', 'lr1', 'lr2']
    RL_ALGORITHM = args.rl_algorithm
    ACCEPTANCE_STRATEGY = args.acceptance_strategy
    REWARD_STRATEGY = args.reward_strategy
    SEED = args.seed
    USE_OPERATOR_ATTENTION = args.use_operator_attention
    seed_suffix = f"_seed{SEED}" if SEED is not None else ""
    attention_suffix = "_attention" if USE_OPERATOR_ATTENTION else ""
    RUN_NAME = f"rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{attention_suffix}{seed_suffix}_{int(time.time())}"

    # Setup logging to file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_{RUN_NAME}.log")

    # Redirect stdout and stderr to both console and log file
    tee_logger = TeeLogger(log_filename)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"Logging to: {log_filename}")
    print("=" * 80)

    # Validate argument combinations
    if args.test_only and args.skip_testing:
        print("Error: Cannot use --test_only and --skip_testing together")
        print("  --test_only means 'skip training, run testing only'")
        print("  --skip_testing means 'skip testing phase'")
        print("  These options are mutually exclusive.")
        sys.exit(1)

    if args.test_only:
        print("\n*** TEST ONLY MODE - Skipping training ***\n")
    else:
        # Initialize RL local search with fresh operators
        # Use different batch sizes for DQN (64) vs PPO (2048)
        batch_size = 64 if RL_ALGORITHM == "dqn" else args.ppo_batch_size

        rl_local_search = RLLocalSearch(
            operators=create_operators(),
            rl_algorithm=RL_ALGORITHM,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-4,
            gamma=0.90,
            # DQN-specific parameters
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9975,
            target_update_interval=100,
            replay_buffer_capacity=100000,
            batch_size=batch_size,
            n_step=3,
            use_prioritized_replay=True,
            per_alpha=0.6,
            per_beta_start=0.4,
            # PPO-specific parameters
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            ppo_entropy_coef=args.ppo_entropy_coef,
            ppo_num_epochs=args.ppo_num_epochs,
            ppo_num_minibatches=args.ppo_num_minibatches,
            # Common parameters
            alpha=10.0,
            beta=0.0,
            acceptance_strategy=ACCEPTANCE_STRATEGY,
            reward_strategy=REWARD_STRATEGY,
            max_iterations=200,
            max_no_improvement=50,
            use_operator_attention=USE_OPERATOR_ATTENTION,
            device="cuda",
            verbose=True
        )

    # Create train/test split
    train_instances, test_instances = split_instances_by_ratio(
        categories=CATEGORIES,
        train_ratio=args.train_ratio
    )

    # Log the train/test split
    print(f"\nTrain/Test Split (ratio={args.train_ratio}):")
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

    if not args.test_only:
        # Create validation set (unless disabled)
        if not args.skip_validation:
            instance_manager = LiLimInstanceManager()
            validation_set = create_validation_set(
                instance_manager=instance_manager,
                solution_generator=create_solution_generator,
                size=PROBLEM_SIZE,
                mode=args.validation_mode,
                num_instances=args.num_validation_instances,
                seed=SEED if SEED is not None else 42,
                problem_generator=problem_generator
            )

            total_validation_runs = len(args.validation_seeds) * args.validation_runs_per_seed
            print(f"\nValidation Set:")
            print(f"  Mode: {args.validation_mode}")
            print(f"  Instances: {len(validation_set)}")
            print(f"  Seeds: {args.validation_seeds}")
            print(f"  Runs per seed: {args.validation_runs_per_seed}")
            print(f"  Total runs per instance: {total_validation_runs}")
            print(f"  Interval: {args.validation_interval} episodes\n")
        else:
            validation_set = None
            print(f"\nValidation: DISABLED (--skip_validation flag set)\n")

        # Train the RL agent
        print(f"Starting RL Local Search Training on size {PROBLEM_SIZE} instances...")
        print(f"Algorithm: {RL_ALGORITHM.upper()}")
        print(f"Categories: {CATEGORIES}")
        print(f"Operator Attention: {'ENABLED' if USE_OPERATOR_ATTENTION else 'DISABLED'}")

        training_history = rl_local_search.train(
            problem_generator=problem_generator,
            initial_solution_generator=create_solution_generator,
            num_episodes=args.num_episodes,
            new_instance_interval=5,
            new_solution_interval=1,
            update_interval=1,
            warmup_episodes=10,
            save_interval=1000,
            save_path=f"models/rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{attention_suffix}_{SEED}_o",
            tensorboard_dir=f"runs/{RUN_NAME}",
            seed=SEED,
            validation_set=validation_set,
            validation_interval=args.validation_interval,
            validation_seeds=args.validation_seeds,
            validation_runs_per_seed=args.validation_runs_per_seed
        )

        print("\nTraining completed!")
        print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
        print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")
    else:
        print("\nStarting evaluation on test instances...")

    # Skip testing if requested
    if args.skip_testing:
        print("\n" + "="*80)
        print("TESTING PHASE: SKIPPED (--skip_testing flag set)")
        print("="*80)
        print("\nTraining completed. Exiting without testing/evaluation.")
        return

    # Create baseline methods with independent operator instances
    adaptive_local_search = AdaptiveLocalSearch(operators=create_operators(), max_no_improvement=50, max_iterations=200)

    naive_local_search = NaiveLocalSearch(operators=create_operators(), max_no_improvement=50, max_iterations=200, first_improvement=True, random_operator_order=True)

    naive_with_best_local_search = NaiveLocalSearch(operators=create_operators(), max_no_improvement=50, max_iterations=200, first_improvement=False)

    random_local_search = RandomLocalSearch(operators=create_operators(), max_no_improvement=50, max_iterations=200)

    # You can provide up to 6 different RL model paths to evaluate here.
    RL_MODEL_PATHS = [
        # f"models/rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}_final_422.pt",
        # f"models/rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}_final_42.pt",
        # f"models/rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}_final_100.pt",
        f"models/rl_local_search_{RL_ALGORITHM}_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{attention_suffix}_{SEED}_o_final.pt",
    ]

    # Configure seeds for testing
    TEST_SEEDS = [42, 422, 100, 200, 300]
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

        # OneShot model (same architecture as saved model)
        model_oneshot = RLLocalSearch(
            operators=create_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=10.0,
            beta=0.0,
            acceptance_strategy="greedy",
            type="OneShot",
            max_iterations=200,
            max_no_improvement=50,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            device="cuda",
            verbose=False
        )
        # Roulette model (probabilistic sampling based on Q-values/policy logits)
        model_roulette = RLLocalSearch(
            operators=create_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=10.0,
            beta=0.0,
            acceptance_strategy="greedy",
            type="Roulette",
            max_iterations=200,
            max_no_improvement=50,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            device="cuda",
            verbose=False
        )
        # Ranking model (strict Q-value/policy order, best first)
        model_ranking = RLLocalSearch(
            operators=create_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=10.0,
            beta=0.0,
            acceptance_strategy="greedy",
            type="Ranking",
            max_iterations=200,
            max_no_improvement=50,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
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
    print(f"  Test problems per seed: {args.num_test_problems}")
    print(f"  Runs per problem: {args.runs_per_problem}")
    print(f"  Total evaluations per seed: {args.num_test_problems * args.runs_per_problem}")
    print(f"  Deterministic RNG: {args.deterministic_test_rng}")
    print()

    for seed_idx, test_seed in enumerate(TEST_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")
        print(f"{'='*80}")

        # Generate test cases (problem + initial solution pairs) upfront for this seed
        print(f"\nGenerating {args.num_test_problems} test cases...")
        test_cases = []
        for case_idx in range(args.num_test_problems):
            set_seed(test_seed + case_idx)
            test_problem, instance_name = test_problem_generator()
            initial_solution = create_solution_generator(test_problem)
            initial_fitness = fitness(test_problem, initial_solution)
            test_cases.append((test_problem, initial_solution, initial_fitness, instance_name))

        # Run each test case multiple times
        for case_idx, (test_problem, initial_solution, initial_fitness, instance_name) in enumerate(test_cases):
            print(f"\nTest Case {case_idx+1}/{args.num_test_problems} - Instance: {instance_name}")
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

            for run_idx in range(args.runs_per_problem):
                if args.runs_per_problem > 1:
                    print(f"  Run {run_idx+1}/{args.runs_per_problem}")

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
                            deterministic_rng=args.deterministic_test_rng,
                            base_seed=run_seed
                        )
                    except Exception as e:
                        print(f"    Model {model_names[midx]} failed: {e}")
                        rl_best_fitness = float('inf')
                    rl_time = time.time() - t0
                    if args.runs_per_problem > 1:
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
                    deterministic_rng=args.deterministic_test_rng,
                    base_seed=run_seed
                )
                adaptive_time = time.time() - t0
                if args.runs_per_problem > 1:
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
                    deterministic_rng=args.deterministic_test_rng,
                    base_seed=run_seed
                )
                naive_time = time.time() - t0
                if args.runs_per_problem > 1:
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
                    deterministic_rng=args.deterministic_test_rng,
                    base_seed=run_seed
                )
                naive_with_best_time = time.time() - t0
                if args.runs_per_problem > 1:
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
                    deterministic_rng=args.deterministic_test_rng,
                    base_seed=run_seed
                )
                random_time = time.time() - t0
                if args.runs_per_problem > 1:
                    print(f"    Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                else:
                    print(f"  Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                instance_results[instance_name]['random'].append(
                    (initial_fitness, random_best_fitness, random_time, case_idx, run_idx)
                )

        # Print brief summary for this seed
        total_evals = args.num_test_problems * args.runs_per_problem
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
            runs_per_case = args.runs_per_problem
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
