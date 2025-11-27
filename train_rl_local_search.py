import sys
import os
import argparse
import random
import time
import numpy as np

class TeeLogger:
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

from memetic.fitness.fitness import fitness

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

from config import load_config, create_operators_from_config, TrainingConfiguration


def create_preset_operators():
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
        
        FlipOperator(),
        FlipOperator(max_attempts=5),
        FlipOperator(single_route=True),
        
        MergeOperator(type="random", num_routes=2),
        MergeOperator(type="random", num_routes=2, reorder=False),
        
        MergeOperator(type="min", num_routes=2),
        MergeOperator(type="min", num_routes=2, reorder=False),
        
        SwapWithinOperator(),
        SwapWithinOperator(max_attempts=5),
        SwapWithinOperator(single_route=True),
        SwapWithinOperator(single_route=True, type="best"),
        SwapWithinOperator(single_route=False, type="best"),

        SwapBetweenOperator(),
        SwapBetweenOperator(type="best"),
        
        TransferOperator(),
        TransferOperator(single_route=True),
        TransferOperator(max_attempts=5,single_route=True),

        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5),
        ShiftOperator(type="random", segment_length=2, max_shift_distance=4, max_attempts=5),
        ShiftOperator(type="random", segment_length=4, max_shift_distance=2, max_attempts=3),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=5, max_attempts=3),
        ShiftOperator(type="best", segment_length=2, max_shift_distance=3),
        ShiftOperator(type="best", segment_length=3, max_shift_distance=2),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5, single_route=True),

        TwoOptOperator(),
        
        CLSM1Operator(),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator(),
        
        RequestShiftWithinOperator(),
        NodeSwapWithinOperator(check_precedence=True),
        NodeSwapWithinOperator(check_precedence=False),
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

    if instance_subset is None:
        available_instances = {cat: instance_manager.CATEGORIES[cat] for cat in categories}
    else:
        available_instances = instance_subset

    def generator():
        category = random.choice(list(available_instances.keys()))
        instance_name = random.choice(available_instances[category])
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
    random.seed(seed)
    np.random.seed(seed)

    # Set PyTorch seeds
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_configuration(args) -> TrainingConfiguration:
    if args.config is not None:
        # CONFIG MODE: Load from YAML
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        print("Configuration loaded successfully (CLI arguments ignored)")
        return config
    else:
        # CLI MODE: Build config from arguments (backward compatible)
        print("Using CLI arguments (no config file provided)")

        config = TrainingConfiguration()

        # Map CLI args to config structure
        config.general.seed = args.seed
        config.general.problem_size = args.problem_size
        config.general.categories = ['lc1', 'lc2', 'lr1', 'lr2']  

        config.algorithm.type = args.rl_algorithm
        config.algorithm.acceptance_strategy = "greedy"  
        config.algorithm.reward_strategy = "binary"  
        config.algorithm.search_type = "OneShot" 

        config.network.learning_rate = 1e-4  
        config.network.gamma = 0.90  
        config.network.hidden_dims = [128, 128, 64]  
        config.network.use_operator_attention = args.use_operator_attention
        config.network.disable_operator_features = False  
        config.network.feature_weights = None  
        # DQN parameters 
        config.dqn.epsilon_start = 1.0
        config.dqn.epsilon_end = 0.05
        config.dqn.epsilon_decay = 0.9975
        config.dqn.target_update_interval = 100
        config.dqn.replay_buffer_capacity = 100000
        config.dqn.batch_size = 64 if args.rl_algorithm == "dqn" else 2048
        config.dqn.n_step = 3
        config.dqn.use_prioritized_replay = True
        config.dqn.per_alpha = 0.6
        config.dqn.per_beta_start = 0.4

        # PPO parameters 
        config.ppo.batch_size = 2048  
        config.ppo.clip_epsilon = 0.2  
        config.ppo.entropy_coef = 0.01  
        config.ppo.num_epochs = 2  
        config.ppo.num_minibatches = 2  
        # PPO parameters not in CLI 
        config.ppo.value_coef = 0.5
        config.ppo.gae_lambda = 0.95
        config.ppo.max_grad_norm = 0.5
        config.ppo.normalize_advantages = True

        # Training parameters
        config.training.num_episodes = args.num_episodes
        config.training.max_iterations = 200  
        config.training.max_no_improvement = 50  
        config.training.new_instance_interval = 5  
        config.training.new_solution_interval = 1  
        config.training.update_interval = 1  
        config.training.warmup_episodes = 10  
        config.training.save_interval = 1000 
        config.training.save_path = "models/rl_local_search"
        config.training.tensorboard_dir = None  
        config.training.log_interval = 10  
        # Environment parameters
        config.environment.alpha = 10.0  
        config.environment.beta = 0.0  

        # Validation
        config.validation.skip_validation = False  
        config.validation.mode = "fixed_benchmark"  
        config.validation.num_instances = 10  
        config.validation.interval = 50  
        config.validation.seeds = [42, 111, 222, 333, 444]  
        config.validation.runs_per_seed = 1  

        # Testing
        config.testing.skip_testing = args.skip_testing
        config.testing.test_only = args.test_only
        config.testing.train_ratio = 0.8  
        config.testing.num_test_problems = 50  
        config.testing.runs_per_problem = 3  
        config.testing.deterministic_test_rng = False  
        config.testing.test_seeds = [42, 422, 100, 200, 300]  
        config.testing.model_paths = [] 

        # Operators: always use preset in CLI mode
        config.operators.mode = "preset"

        # Validate and return
        config.validate()
        return config


def print_config_summary(config: TrainingConfiguration):
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)

    print(f"\nGeneral:")
    print(f"  Seed: {config.general.seed}")
    print(f"  Problem size: {config.general.problem_size}")
    print(f"  Categories: {', '.join(config.general.categories)}")

    print(f"\nAlgorithm:")
    print(f"  Type: {config.algorithm.type.upper()}")
    print(f"  Acceptance strategy: {config.algorithm.acceptance_strategy}")
    print(f"  Reward strategy: {config.algorithm.reward_strategy}")
    print(f"  Search type: {config.algorithm.search_type}")

    print(f"\nNetwork:")
    print(f"  Hidden dims: {config.network.hidden_dims}")
    print(f"  Learning rate: {config.network.learning_rate}")
    print(f"  Gamma: {config.network.gamma}")
    print(f"  Operator attention: {config.network.use_operator_attention}")
    print(f"  Disable operator features: {config.network.disable_operator_features}")
    if config.network.feature_weights is not None:
        num_enabled = sum(config.network.feature_weights)
        total = len(config.network.feature_weights)
        print(f"  Feature weights: {num_enabled}/{total} features enabled")
    else:
        print(f"  Feature weights: None (all enabled)")

    if config.algorithm.type == "dqn":
        print(f"\nDQN Parameters:")
        print(f"  Epsilon: {config.dqn.epsilon_start} -> {config.dqn.epsilon_end} (decay: {config.dqn.epsilon_decay})")
        print(f"  Batch size: {config.dqn.batch_size}")
        print(f"  N-step: {config.dqn.n_step}")
        print(f"  Prioritized replay: {config.dqn.use_prioritized_replay}")
        if config.dqn.use_prioritized_replay:
            print(f"  PER alpha: {config.dqn.per_alpha}, beta start: {config.dqn.per_beta_start}")
    else:
        print(f"\nPPO Parameters:")
        print(f"  Batch size: {config.ppo.batch_size}")
        print(f"  Clip epsilon: {config.ppo.clip_epsilon}")
        print(f"  Entropy coef: {config.ppo.entropy_coef}")
        print(f"  Num epochs: {config.ppo.num_epochs}")
        print(f"  Num minibatches: {config.ppo.num_minibatches}")

    print(f"\nTraining:")
    print(f"  Episodes: {config.training.num_episodes}")
    print(f"  Max iterations/episode: {config.training.max_iterations}")
    print(f"  Max no improvement: {config.training.max_no_improvement}")
    print(f"  New instance interval: {config.training.new_instance_interval}")
    print(f"  Save interval: {config.training.save_interval}")

    print(f"\nValidation:")
    if config.validation.skip_validation:
        print(f"  DISABLED")
    else:
        print(f"  Mode: {config.validation.mode}")
        print(f"  Instances: {config.validation.num_instances}")
        print(f"  Interval: {config.validation.interval}")
        print(f"  Seeds: {config.validation.seeds}")

    print(f"\nTesting:")
    if config.testing.skip_testing:
        print(f"  DISABLED")
    elif config.testing.test_only:
        print(f"  TEST ONLY MODE")
    else:
        print(f"  Train ratio: {config.testing.train_ratio}")
        print(f"  Test problems: {config.testing.num_test_problems}")
        print(f"  Runs per problem: {config.testing.runs_per_problem}")

    print(f"\nOperators:")
    print(f"  Mode: {config.operators.mode}")
    if config.operators.mode == "custom":
        print(f"  Custom operators: {len(config.operators.custom_list)}")
        for i, op_spec in enumerate(config.operators.custom_list):
            print(f"    [{i}] {op_spec.type}")

    print("="*80 + "\n")


def main():
    """Main training script for RL local search."""

    parser = argparse.ArgumentParser(
        description="Train RL-based local search with configurable strategies",
        epilog="Use --config to load all settings from YAML file (overrides all other arguments)"
    )

    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file (when provided, all other args are ignored)")

    parser.add_argument("--rl_algorithm", type=str, default="dqn",
                        choices=["dqn", "ppo"],
                        help="RL algorithm to use: dqn or ppo (default: dqn)")
    parser.add_argument("--problem_size", type=int, default=100,
                        choices=[100, 200, 400, 600, 1000],
                        help="Problem size")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use_operator_attention", action="store_true",
                        help="Enable operator attention mechanism for operator selection")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run testing/evaluation on existing models")
    parser.add_argument("--skip_testing", action="store_true",
                        help="Skip testing/evaluation phase after training")

    args = parser.parse_args()

    config = resolve_configuration(args)

    if config.general.seed is not None:
        print(f"Setting random seed to {config.general.seed}")
        set_seed(config.general.seed)

    # Generate run name from config
    attention_suffix = "_attention" if config.network.use_operator_attention else ""
    seed_suffix = f"_seed{config.general.seed}" if config.general.seed is not None else ""
    custom_suffix = f"_{config.general.run_name_suffix}" if config.general.run_name_suffix else ""
    RUN_NAME = (f"rl_local_search_{config.algorithm.type}_{config.general.problem_size}_"
                f"{config.algorithm.acceptance_strategy}_{config.algorithm.reward_strategy}"
                f"{attention_suffix}{seed_suffix}{custom_suffix}_{int(time.time())}")

    # Setup logging to file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_{RUN_NAME}.log")

    tee_logger = TeeLogger(log_filename)
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    print(f"Logging to: {log_filename}")
    print("=" * 80)

    print_config_summary(config)

    if config.testing.test_only and config.testing.skip_testing:
        print("Error: Cannot use --test_only and --skip_testing together")
        sys.exit(1)

    # Create operators
    if config.operators.mode == "preset":
        operators = create_preset_operators()
        print(f"Using preset operators ({len(operators)} operators)")
    else:
        operators = create_operators_from_config(config.operators.custom_list)
        print(f"Using custom operators ({len(operators)} operators)")
        for i, op in enumerate(operators):
            print(f"  [{i}] {op.__class__.__name__}")

    if config.testing.test_only:
        print("\n*** TEST ONLY MODE - Skipping training ***\n")
    else:
        batch_size = config.dqn.batch_size if config.algorithm.type == "dqn" else config.ppo.batch_size

        rl_local_search = RLLocalSearch(
            operators=operators,
            rl_algorithm=config.algorithm.type,
            hidden_dims=config.network.hidden_dims,
            learning_rate=config.network.learning_rate,
            gamma=config.network.gamma,
            # DQN-specific parameters
            epsilon_start=config.dqn.epsilon_start,
            epsilon_end=config.dqn.epsilon_end,
            epsilon_decay=config.dqn.epsilon_decay,
            target_update_interval=config.dqn.target_update_interval,
            replay_buffer_capacity=config.dqn.replay_buffer_capacity,
            batch_size=batch_size,
            n_step=config.dqn.n_step,
            use_prioritized_replay=config.dqn.use_prioritized_replay,
            per_alpha=config.dqn.per_alpha,
            per_beta_start=config.dqn.per_beta_start,
            # PPO-specific parameters
            ppo_clip_epsilon=config.ppo.clip_epsilon,
            ppo_entropy_coef=config.ppo.entropy_coef,
            ppo_num_epochs=config.ppo.num_epochs,
            ppo_num_minibatches=config.ppo.num_minibatches,
            # Common parameters
            alpha=config.environment.alpha,
            beta=config.environment.beta,
            acceptance_strategy=config.algorithm.acceptance_strategy,
            reward_strategy=config.algorithm.reward_strategy,
            max_iterations=config.training.max_iterations,
            max_no_improvement=config.training.max_no_improvement,
            use_operator_attention=config.network.use_operator_attention,
            disable_operator_features=config.network.disable_operator_features,
            feature_weights=np.array(config.network.feature_weights) if config.network.feature_weights is not None else None,
            device="cuda",
            verbose=True
        )

    # Create train/test split
    train_instances, test_instances = split_instances_by_ratio(
        categories=config.general.categories,
        train_ratio=config.testing.train_ratio
    )

    # Log the train/test split
    print(f"\nTrain/Test Split (ratio={config.testing.train_ratio}):")
    for category in config.general.categories:
        print(f"  {category}:")
        print(f"    Train ({len(train_instances[category])}): {train_instances[category]}")
        print(f"    Test  ({len(test_instances[category])}): {test_instances[category]}")
    print()

    problem_generator = create_problem_generator(
        size=config.general.problem_size,
        categories=config.general.categories,
        instance_subset=train_instances
    )

    test_problem_generator = create_problem_generator(
        size=config.general.problem_size,
        categories=config.general.categories,
        instance_subset=test_instances,
        return_name=True
    )

    if not config.testing.test_only:
        if not config.validation.skip_validation:
            instance_manager = LiLimInstanceManager()
            validation_set = create_validation_set(
                instance_manager=instance_manager,
                solution_generator=create_solution_generator,
                size=config.general.problem_size,
                mode=config.validation.mode,
                num_instances=config.validation.num_instances,
                seed=config.general.seed if config.general.seed is not None else 42,
                problem_generator=problem_generator
            )

            total_validation_runs = len(config.validation.seeds) * config.validation.runs_per_seed
        else:
            validation_set = None

        # Train the RL agent
        print(f"Starting RL Local Search Training on size {config.general.problem_size} instances...")

        # Auto-generate save path and tensorboard dir
        save_path = (f"{config.training.save_path}_{config.algorithm.type}_{config.general.problem_size}_"
                    f"{config.algorithm.acceptance_strategy}_{config.algorithm.reward_strategy}"
                    f"{attention_suffix}_{config.general.seed}_{config.general.run_name_suffix}")

        # Auto-generate tensorboard subdirectory with run name
        if config.training.tensorboard_dir:
            base_dir = config.training.tensorboard_dir.rstrip('/')
            tensorboard_dir = f"{base_dir}/{RUN_NAME}"
        else:
            # Default: use runs/ as base directory
            tensorboard_dir = f"runs/{RUN_NAME}"

        training_history = rl_local_search.train(
            problem_generator=problem_generator,
            initial_solution_generator=create_solution_generator,
            num_episodes=config.training.num_episodes,
            new_instance_interval=config.training.new_instance_interval,
            new_solution_interval=config.training.new_solution_interval,
            update_interval=config.training.update_interval,
            warmup_episodes=config.training.warmup_episodes,
            save_interval=config.training.save_interval,
            save_path=save_path,
            tensorboard_dir=tensorboard_dir,
            seed=config.general.seed,
            validation_set=validation_set,
            validation_interval=config.validation.interval,
            validation_seeds=config.validation.seeds,
            validation_runs_per_seed=config.validation.runs_per_seed
        )

        print("\nTraining completed!")
        print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
        print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")
    else:
        print("\nStarting evaluation on test instances...")

    if config.testing.skip_testing:
        return

    # Create baseline methods with independent operator instances
    adaptive_local_search = AdaptiveLocalSearch(
        operators=create_preset_operators(),
        max_no_improvement=config.training.max_no_improvement,
        max_iterations=config.training.max_iterations
    )

    naive_local_search = NaiveLocalSearch(
        operators=create_preset_operators(),
        max_no_improvement=config.training.max_no_improvement,
        max_iterations=config.training.max_iterations,
        first_improvement=True,
        random_operator_order=True
    )

    naive_with_best_local_search = NaiveLocalSearch(
        operators=create_preset_operators(),
        max_no_improvement=config.training.max_no_improvement,
        max_iterations=config.training.max_iterations,
        first_improvement=False
    )

    random_local_search = RandomLocalSearch(
        operators=create_preset_operators(),
        max_no_improvement=config.training.max_no_improvement,
        max_iterations=config.training.max_iterations
    )

    # Configure RL model paths for evaluation
    if config.testing.model_paths:
        # Use paths from config
        RL_MODEL_PATHS = config.testing.model_paths
    else:
        # Auto-generate path for just-trained model
        RL_MODEL_PATHS = [
            f"models/rl_local_search_{config.algorithm.type}_{config.general.problem_size}_"
            f"{config.algorithm.acceptance_strategy}_{config.algorithm.reward_strategy}"
            f"{attention_suffix}_{config.general.seed}_o_final.pt",
        ]

    TEST_SEEDS = config.testing.test_seeds

    rl_models = []
    model_names = []
    for path in RL_MODEL_PATHS:
        if not path:
            continue

        # Auto-detect architecture from checkpoint
        try:
            model_is_ppo = detect_ppo_from_checkpoint(path)
            model_algorithm = "ppo" if model_is_ppo else "dqn"

            if not model_is_ppo:
                model_uses_attention = detect_attention_from_checkpoint(path)
            else:
                model_uses_attention = config.network.use_operator_attention

            arch_str = f"{model_algorithm.upper()}, {'with' if model_uses_attention else 'without'} attention"
            print(f"Detected model architecture ({arch_str}): {path}")
        except Exception as e:
            print(f"Warning: could not detect architecture for {path}: {e}")
            continue

        # OneShot model (same architecture as saved model)
        model_oneshot = RLLocalSearch(
            operators=create_preset_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network.hidden_dims,
            learning_rate=config.network.learning_rate,
            gamma=config.network.gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=config.environment.alpha,
            beta=config.environment.beta,
            acceptance_strategy="greedy",
            type="OneShot",
            max_iterations=config.training.max_iterations,
            max_no_improvement=config.training.max_no_improvement,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            disable_operator_features=config.network.disable_operator_features,
            feature_weights=np.array(config.network.feature_weights) if config.network.feature_weights is not None else None,
            device="cuda",
            verbose=False
        )
        # Roulette model (probabilistic sampling based on Q-values/policy logits)
        model_roulette = RLLocalSearch(
            operators=create_preset_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network.hidden_dims,
            learning_rate=config.network.learning_rate,
            gamma=config.network.gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=config.environment.alpha,
            beta=config.environment.beta,
            acceptance_strategy="greedy",
            type="Roulette",
            max_iterations=config.training.max_iterations,
            max_no_improvement=config.training.max_no_improvement,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            disable_operator_features=config.network.disable_operator_features,
            feature_weights=np.array(config.network.feature_weights) if config.network.feature_weights is not None else None,
            device="cuda",
            verbose=False
        )
        # Ranking model (strict Q-value/policy order, best first)
        model_ranking = RLLocalSearch(
            operators=create_preset_operators(),
            rl_algorithm=model_algorithm,
            hidden_dims=config.network.hidden_dims,
            learning_rate=config.network.learning_rate,
            gamma=config.network.gamma,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_interval=100,
            alpha=config.environment.alpha,
            beta=config.environment.beta,
            acceptance_strategy="greedy",
            type="Ranking",
            max_iterations=config.training.max_iterations,
            max_no_improvement=config.training.max_no_improvement,
            replay_buffer_capacity=100000,
            batch_size=64,
            n_step=3,
            use_prioritized_replay=False,
            use_operator_attention=model_uses_attention,
            disable_operator_features=config.network.disable_operator_features,
            feature_weights=np.array(config.network.feature_weights) if config.network.feature_weights is not None else None,
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
    print(f"  Test problems per seed: {config.testing.num_test_problems}")
    print(f"  Runs per problem: {config.testing.runs_per_problem}")
    print(f"  Total evaluations per seed: {config.testing.num_test_problems * config.testing.runs_per_problem}")
    print(f"  Deterministic RNG: {config.testing.deterministic_test_rng}")
    print()

    for seed_idx, test_seed in enumerate(TEST_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")
        print(f"{'='*80}")

        # Generate test cases (problem + initial solution pairs) upfront for this seed
        print(f"\nGenerating {config.testing.num_test_problems} test cases...")
        test_cases = []
        for case_idx in range(config.testing.num_test_problems):
            set_seed(test_seed + case_idx)
            test_problem, instance_name = test_problem_generator()
            initial_solution = create_solution_generator(test_problem)
            initial_fitness = fitness(test_problem, initial_solution)
            test_cases.append((test_problem, initial_solution, initial_fitness, instance_name))

        # Run each test case multiple times
        for case_idx, (test_problem, initial_solution, initial_fitness, instance_name) in enumerate(test_cases):
            print(f"\nTest Case {case_idx+1}/{config.testing.num_test_problems} - Instance: {instance_name}")
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

            for run_idx in range(config.testing.runs_per_problem):
                if config.testing.runs_per_problem > 1:
                    print(f"  Run {run_idx+1}/{config.testing.runs_per_problem}")

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
                            deterministic_rng=config.testing.deterministic_test_rng,
                            base_seed=run_seed
                        )
                    except Exception as e:
                        print(f"    Model {model_names[midx]} failed: {e}")
                        rl_best_fitness = float('inf')
                    rl_time = time.time() - t0
                    if config.testing.runs_per_problem > 1:
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
                    deterministic_rng=config.testing.deterministic_test_rng,
                    base_seed=run_seed
                )
                adaptive_time = time.time() - t0
                if config.testing.runs_per_problem > 1:
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
                    deterministic_rng=config.testing.deterministic_test_rng,
                    base_seed=run_seed
                )
                naive_time = time.time() - t0
                if config.testing.runs_per_problem > 1:
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
                    deterministic_rng=config.testing.deterministic_test_rng,
                    base_seed=run_seed
                )
                naive_with_best_time = time.time() - t0
                if config.testing.runs_per_problem > 1:
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
                    deterministic_rng=config.testing.deterministic_test_rng,
                    base_seed=run_seed
                )
                random_time = time.time() - t0
                if config.testing.runs_per_problem > 1:
                    print(f"    Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                else:
                    print(f"  Random: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
                instance_results[instance_name]['random'].append(
                    (initial_fitness, random_best_fitness, random_time, case_idx, run_idx)
                )

        total_evals = config.testing.num_test_problems * config.testing.runs_per_problem
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
            runs_per_case = config.testing.runs_per_problem
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
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
