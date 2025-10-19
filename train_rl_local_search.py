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
from utils.instance_manager import InstanceManager
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
        # ReinsertOperator(max_attempts=5,clustered=True),
        # ReinsertOperator(force_same_vehicle=True),
        # ReinsertOperator(allow_same_vehicle=False),
        # ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        
        RouteEliminationOperator(),
        
        # FlipOperator(),
        # FlipOperator(max_attempts=5),
        FlipOperator(single_route=True),
        
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


def create_problem_generator(size: int = 100, categories: list[str] = None):
    """Create a function that generates problem instances using InstanceManager.

    Args:
        size: Problem size (100, 200, 400, 600, 1000)
        categories: List of categories to sample from (default: all)

    Returns:
        Generator function that returns random problem instances
    """
    instance_manager = InstanceManager()

    if categories is None:
        categories = list(instance_manager.CATEGORIES.keys())

    def generator() -> PDPTWProblem:
        # Randomly select category and instance
        category = random.choice(categories)
        instance_name = random.choice(instance_manager.CATEGORIES[category])
        # print(f"Loading instance: {instance_name} from category: {category}")
        return instance_manager.load(instance_name, size)

    return generator


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
    instance_manager: InstanceManager,
    solution_generator: callable,
    size: int,
    mode: str = "fixed_benchmark",
    num_instances: int = 10,
    seed: int = 42,
    problem_generator: callable = None
):
    """Create validation set for evaluating RL local search.

    Args:
        instance_manager: InstanceManager for loading problems
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
                        choices=[    "tanh", "distance_baseline_tanh",
    "distance_baseline_normalized",
    "pure_normalized",
    "distance_baseline_asymmetric_tanh",
    "initial_improvement",
    "old_improvement",
    "hybrid_improvement",
    "distance_baseline",
    "log_improvement",
    "binary",
    "distance_baseline_clipped",
    "another_one"],
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

    # # Create problem and solution generators
    problem_generator = create_problem_generator(size=PROBLEM_SIZE, categories=CATEGORIES)

    # Create validation set
    instance_manager = InstanceManager()
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
    NUM_TESTS_PER_SEED = 50  # Number of tests per seed

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
    # Results structure: per seed -> per model -> list of (initial, best, time) tuples
    all_seeds_results = {}

    print("\n--- Comparing RL models vs Naive/Random Local Search ---")
    print(f"Testing across {len(TEST_SEEDS)} seeds with {NUM_TESTS_PER_SEED} tests per seed")
    print(f"Seeds: {TEST_SEEDS}")

    for seed_idx, test_seed in enumerate(TEST_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")
        print(f"{'='*80}")

        rl_results = [[] for _ in range(len(rl_models))]  # per-model collected tuples (initial, best, time)
        adaptive_results = []
        naive_results = []
        naive_with_best_results = []
        random_results = []

        for i in range(NUM_TESTS_PER_SEED):
            print(f"\nTest {i+1}/{NUM_TESTS_PER_SEED}")
            # Base seed for this test (ensures all methods see same randomness)
            base_seed = test_seed + i
            set_seed(base_seed)
            test_problem = problem_generator()
            initial_solution = create_solution_generator(test_problem)

            # Evaluate initial solution
            initial_fitness = fitness(test_problem, initial_solution)
            print(f"Initial fitness: {initial_fitness:.2f}")

            # Run each RL model
            for midx, model in enumerate(rl_models):
                set_seed(base_seed)  # Reset to same seed for fair comparison
                rl_solution = initial_solution.clone()
                t0 = time.time()
                try:
                    rl_best_solution, rl_best_fitness = model.search(
                        problem=test_problem,
                        solution=rl_solution,
                        epsilon=0.0,
                        deterministic_rng=True,
                        base_seed=base_seed
                    )
                except Exception as e:
                    print(f"Model {model_names[midx]} failed on test {i+1}: {e}")
                    rl_best_fitness = float('inf')
                rl_time = time.time() - t0
                print(f"{model_names[midx]}: best fitness: {rl_best_fitness:.2f} (time: {rl_time:.2f}s)")
                rl_results[midx].append((initial_fitness, rl_best_fitness, rl_time))

            # Adaptive local search
            set_seed(base_seed)
            adaptive_solution = initial_solution.clone()
            t0 = time.time()
            adaptive_best_solution, adaptive_best_fitness = adaptive_local_search.search(
                problem=test_problem,
                solution=adaptive_solution,
                deterministic_rng=True,
                base_seed=base_seed
            )
            adaptive_time = time.time() - t0
            print(f"Adaptive: best fitness: {adaptive_best_fitness:.2f} (time: {adaptive_time:.2f}s)")
            adaptive_results.append((initial_fitness, adaptive_best_fitness, adaptive_time))

            # Naive local search
            set_seed(base_seed)
            naive_solution = initial_solution.clone()
            t0 = time.time()
            naive_best_solution, naive_best_fitness = naive_local_search.search(
                problem=test_problem,
                solution=naive_solution,
                deterministic_rng=True,
                base_seed=base_seed
            )
            naive_time = time.time() - t0
            print(f"Naive: best fitness: {naive_best_fitness:.2f} (time: {naive_time:.2f}s)")
            naive_results.append((initial_fitness, naive_best_fitness, naive_time))

            # Naive local search with best improvement
            set_seed(base_seed)
            naive_with_best_solution = initial_solution.clone()
            t0 = time.time()
            naive_with_best_best_solution, naive_with_best_best_fitness = naive_with_best_local_search.search(
                problem=test_problem,
                solution=naive_with_best_solution,
                deterministic_rng=True,
                base_seed=base_seed
            )
            naive_with_best_time = time.time() - t0
            print(f"Naive (best improvement): best fitness: {naive_with_best_best_fitness:.2f} (time: {naive_with_best_time:.2f}s)")
            naive_with_best_results.append((initial_fitness, naive_with_best_best_fitness, naive_with_best_time))

            # Random local search
            set_seed(base_seed)
            random_solution = initial_solution.clone()
            t0 = time.time()
            random_best_solution, random_best_fitness = random_local_search.search(
                problem=test_problem,
                solution=random_solution,
                deterministic_rng=True,
                base_seed=base_seed
            )
            random_time = time.time() - t0
            print(f"Random: best fitness: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
            random_results.append((initial_fitness, random_best_fitness, random_time))

        # Store results for this seed
        all_seeds_results[test_seed] = {
            'rl_models': rl_results,
            'adaptive': adaptive_results,
            'naive': naive_results,
            'naive_with_best': naive_with_best_results,
            'random': random_results
        }

        # Print summary for this seed
        print(f"\n--- Summary for Seed {test_seed} ---")
        avg_initial = sum(r[0] for r in rl_results[0]) / NUM_TESTS_PER_SEED if rl_results[0] else 0.0
        print(f"Average initial fitness: {avg_initial:.2f}")
        for midx, name in enumerate(model_names):
            avg_best = sum(r[1] for r in rl_results[midx]) / NUM_TESTS_PER_SEED
            avg_time = sum(r[2] for r in rl_results[midx]) / NUM_TESTS_PER_SEED
            print(f"Model {name}: Avg best fitness: {avg_best:.2f} (avg time: {avg_time:.2f}s)")

        avg_adaptive = sum(r[1] for r in adaptive_results) / NUM_TESTS_PER_SEED
        avg_time_adaptive = sum(r[2] for r in adaptive_results) / NUM_TESTS_PER_SEED
        avg_naive = sum(r[1] for r in naive_results) / NUM_TESTS_PER_SEED
        avg_time_naive = sum(r[2] for r in naive_results) / NUM_TESTS_PER_SEED
        avg_naive_with_best = sum(r[1] for r in naive_with_best_results) / NUM_TESTS_PER_SEED
        avg_time_naive_with_best = sum(r[2] for r in naive_with_best_results) / NUM_TESTS_PER_SEED
        avg_random = sum(r[1] for r in random_results) / NUM_TESTS_PER_SEED
        avg_time_random = sum(r[2] for r in random_results) / NUM_TESTS_PER_SEED
        print(f"Adaptive: Avg best fitness: {avg_adaptive:.2f} (avg time: {avg_time_adaptive:.2f}s)")
        print(f"Naive: Avg best fitness: {avg_naive:.2f} (avg time: {avg_time_naive:.2f}s)")
        print(f"Naive (best improvement): Avg best fitness: {avg_naive_with_best:.2f} (avg time: {avg_time_naive_with_best:.2f}s)")
        print(f"Random: Avg best fitness: {avg_random:.2f} (avg time: {avg_time_random:.2f}s)")

    # Overall summary across all seeds
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*80}")
    total_tests = len(TEST_SEEDS) * NUM_TESTS_PER_SEED
    print(f"Total tests: {total_tests} ({len(TEST_SEEDS)} seeds × {NUM_TESTS_PER_SEED} tests per seed)")

    # Aggregate all results across seeds
    all_rl_results = [[] for _ in range(len(rl_models))]
    all_adaptive_results = []
    all_naive_results = []
    all_naive_with_best_results = []
    all_random_results = []

    for seed, results in all_seeds_results.items():
        for midx in range(len(rl_models)):
            all_rl_results[midx].extend(results['rl_models'][midx])
        all_adaptive_results.extend(results['adaptive'])
        all_naive_results.extend(results['naive'])
        all_naive_with_best_results.extend(results['naive_with_best'])
        all_random_results.extend(results['random'])

    # Print overall averages
    avg_initial = sum(r[0] for r in all_rl_results[0]) / total_tests if all_rl_results[0] else 0.0
    print(f"\nAverage initial fitness: {avg_initial:.2f}")

    print("\nRL Models:")
    for midx, name in enumerate(model_names):
        avg_best = sum(r[1] for r in all_rl_results[midx]) / total_tests
        avg_time = sum(r[2] for r in all_rl_results[midx]) / total_tests
        std_best = np.std([r[1] for r in all_rl_results[midx]])
        print(f"  {name}: Avg best: {avg_best:.2f} ± {std_best:.2f} (avg time: {avg_time:.2f}s)")

    print("\nBaseline Methods:")
    avg_adaptive = sum(r[1] for r in all_adaptive_results) / total_tests
    avg_time_adaptive = sum(r[2] for r in all_adaptive_results) / total_tests
    std_adaptive = np.std([r[1] for r in all_adaptive_results])
    print(f"  Adaptive: Avg best: {avg_adaptive:.2f} ± {std_adaptive:.2f} (avg time: {avg_time_adaptive:.2f}s)")

    avg_naive = sum(r[1] for r in all_naive_results) / total_tests
    avg_time_naive = sum(r[2] for r in all_naive_results) / total_tests
    std_naive = np.std([r[1] for r in all_naive_results])
    print(f"  Naive: Avg best: {avg_naive:.2f} ± {std_naive:.2f} (avg time: {avg_time_naive:.2f}s)")

    avg_naive_with_best = sum(r[1] for r in all_naive_with_best_results) / total_tests
    avg_time_naive_with_best = sum(r[2] for r in all_naive_with_best_results) / total_tests
    std_naive_with_best = np.std([r[1] for r in all_naive_with_best_results])
    print(f"  Naive (best improvement): Avg best: {avg_naive_with_best:.2f} ± {std_naive_with_best:.2f} (avg time: {avg_time_naive_with_best:.2f}s)")

    avg_random = sum(r[1] for r in all_random_results) / total_tests
    avg_time_random = sum(r[2] for r in all_random_results) / total_tests
    std_random = np.std([r[1] for r in all_random_results])
    print(f"  Random: Avg best: {avg_random:.2f} ± {std_random:.2f} (avg time: {avg_time_random:.2f}s)")

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
