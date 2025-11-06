"""Example script for training and using RL-based mutation."""

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
from memetic.mutation.rl_mutation.rl_mutation import RLMutation
from memetic.mutation.naive_mutation import NaiveMutation
from memetic.mutation.rl_mutation.ppo_mutation_network import detect_ppo_from_checkpoint
from memetic.solution_generators.random_generator import RandomGenerator
from memetic.solution_generators.hybrid_generator import HybridGenerator

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
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.no_op import NoOpOperator


def create_operators(include_noop: bool = True):
    """Create a fresh set of mutation operators with independent state.

    Args:
        include_noop: Whether to include NoOp operator

    Returns:
        List of operator instances
    """
    operators = [
        ReinsertOperator(),
        RouteEliminationOperator(),
        FlipOperator(single_route=True),
        SwapWithinOperator(),
        SwapBetweenOperator(),
        TransferOperator(),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=5, max_attempts=3),
        TwoOptOperator(),
    ]

    # operators.append(NoOpOperator())

    return operators


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


def create_population_generator(population_size: int = 10):
    """Create a function that generates populations for a given problem.

    Args:
        population_size: Number of solutions in population

    Returns:
        Generator function that creates populations
    """
    def generator(problem: PDPTWProblem) -> list[PDPTWSolution]:
        gen = RandomGenerator()
        population = gen.generate(problem, num_solutions=population_size)
        return population

    return generator


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
    """Main training script for RL mutation."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train RL-based mutation with configurable strategies")
    parser.add_argument("--rl_algorithm", type=str, default="ppo",
                        choices=["dqn", "ppo"],
                        help="RL algorithm to use: dqn or ppo (default: ppo)")
    parser.add_argument("--acceptance_strategy", type=str, default="greedy",
                        choices=["greedy", "always"],
                        help="Acceptance strategy for mutation")
    parser.add_argument("--reward_strategy", type=str, default="hybrid_sparse",
                        help="Reward strategy for RL agent")
    parser.add_argument("--problem_size", type=int, default=100,
                        choices=[100, 200, 400, 600, 1000],
                        help="Problem size")
    parser.add_argument("--population_size", type=int, default=10,
                        help="Population size for mutation (default: 10)")
    parser.add_argument("--num_episodes", type=int, default=1500,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--include_noop", action="store_true",
                        help="Include NoOp operator in operator set")

    # Training loop intervals
    parser.add_argument("--new_instance_interval", type=int, default=6,
                        help="Generate new instance every N episodes ")
    parser.add_argument("--new_population_interval", type=int, default=3,
                        help="Generate new population every N episodes ")
    parser.add_argument("--new_solution_interval", type=int, default=1,
                        help="Select new solution from population every N episodes ")

    # Train/Test split arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of instances to use for training (0.0-1.0, default: 0.8)")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only run testing/evaluation on existing models")

    # DQN-specific arguments
    parser.add_argument("--dqn_batch_size", type=int, default=64,
                        help="DQN batch size (default: 64)")
    parser.add_argument("--dqn_n_step", type=int, default=3,
                        help="DQN n-step returns (default: 3)")
    parser.add_argument("--use_prioritized_replay", action="store_true",
                        help="Use prioritized experience replay for DQN")

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
    POPULATION_SIZE = args.population_size

    seed_suffix = f"_seed{SEED}" if SEED is not None else ""
    noop_suffix = "_noop" if args.include_noop else ""
    RUN_NAME = f"rl_mutation_{RL_ALGORITHM}_{PROBLEM_SIZE}_pop{POPULATION_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{noop_suffix}{seed_suffix}_{int(time.time())}"

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

    if args.test_only:
        print("\n*** TEST ONLY MODE - Skipping training ***\n")
    else:
        # Initialize RL mutation with fresh operators
        batch_size = args.dqn_batch_size if RL_ALGORITHM == "dqn" else args.ppo_batch_size

        rl_mutation = RLMutation(
            operators=create_operators(include_noop=args.include_noop),
            rl_algorithm=RL_ALGORITHM,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-3,
            gamma=0.99,
            # DQN-specific parameters
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_interval=100,
            replay_buffer_capacity=100000,
            batch_size=batch_size,
            n_step=args.dqn_n_step,
            use_prioritized_replay=args.use_prioritized_replay,
            per_alpha=0.6,
            per_beta_start=0.4,
            # PPO-specific parameters
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            ppo_entropy_coef=args.ppo_entropy_coef,
            ppo_num_epochs=args.ppo_num_epochs,
            ppo_num_minibatches=args.ppo_num_minibatches,
            # Common parameters
            alpha=1.0,
            acceptance_strategy=ACCEPTANCE_STRATEGY,
            reward_strategy=REWARD_STRATEGY,
            max_steps=100,
            max_no_improvement=None,
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

    # Create problem and population generators
    problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=train_instances
    )

    population_generator = create_population_generator(population_size=POPULATION_SIZE)

    if not args.test_only:
        # Train the RL agent
        print(f"Starting RL Mutation Training on size {PROBLEM_SIZE} instances...")
        print(f"Algorithm: {RL_ALGORITHM.upper()}")
        print(f"Categories: {CATEGORIES}")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Include NoOp: {args.include_noop}")
        print(f"Operators: {[op.name if hasattr(op, 'name') else type(op).__name__ for op in create_operators(args.include_noop)]}")

        training_history = rl_mutation.train(
            problem_generator=problem_generator,
            population_generator=population_generator,
            num_episodes=args.num_episodes,
            new_instance_interval=args.new_instance_interval,
            new_population_interval=args.new_population_interval,
            new_solution_interval=args.new_solution_interval,
            update_interval=1,
            warmup_episodes=10,
            save_interval=500,
            save_path=f"models/rl_mutation_{RL_ALGORITHM}_{PROBLEM_SIZE}_pop{POPULATION_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{noop_suffix}_{SEED}",
            tensorboard_dir=f"runs/{RUN_NAME}",
            seed=SEED,
            log_interval=10
        )

        print("\nTraining completed!")
        print(f"Final average reward (last 100): {np.mean(training_history['episode_rewards'][-100:]):.2f}")
        print(f"Final average fitness (last 100): {np.mean(training_history['episode_best_fitness'][-100:]):.2f}")

        # Save final model
        final_model_path = f"models/rl_mutation_{RL_ALGORITHM}_{PROBLEM_SIZE}_pop{POPULATION_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{noop_suffix}_{SEED}_final.pt"
        rl_mutation.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")

    else:
        print("\nSkipping training - proceeding to evaluation...")

    # Evaluation section
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST INSTANCES")
    print("=" * 80)

    # Load trained models for evaluation
    MODEL_PATHS = [
        f"models/rl_mutation_{RL_ALGORITHM}_{PROBLEM_SIZE}_pop{POPULATION_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{noop_suffix}_{SEED}_final.pt",
    ]

    # Test by running mutation episodes and comparing best solutions
    test_problem_generator = create_problem_generator(
        size=PROBLEM_SIZE,
        categories=CATEGORIES,
        instance_subset=test_instances,
        return_name=True
    )

    # Evaluation configuration
    NUM_TEST_INSTANCES = 20
    NUM_EPISODES_PER_TEST = 1  # Number of mutation episodes per test instance
    TEST_SEEDS = [42, 111, 222]

    print(f"\nEvaluation Configuration:")
    print(f"  Test instances: {NUM_TEST_INSTANCES}")
    print(f"  Episodes per instance: {NUM_EPISODES_PER_TEST}")
    print(f"  Test seeds: {TEST_SEEDS}")
    print(f"  Total evaluations per model: {NUM_TEST_INSTANCES * NUM_EPISODES_PER_TEST * len(TEST_SEEDS)}")

    # ========================================
    # NAIVE BASELINE EVALUATION (Random Operator Selection)
    # ========================================
    print(f"\n{'='*80}")
    print("NAIVE BASELINE EVALUATION (Random Operator Selection)")
    print(f"{'='*80}")

    # Import distance measure for diversity calculation
    from memetic.utils.distance_measure import DistanceMeasure
    distance_measure = DistanceMeasure()

    # Create naive mutation with single iteration (will call 100 times)
    baseline_operators = create_operators(include_noop=args.include_noop)
    naive_mutation = NaiveMutation(operators=baseline_operators, max_iterations=30)
    baseline_results = []

    for seed_idx, test_seed in enumerate(TEST_SEEDS):
        print(f"\nTest Seed {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")

        for test_idx in range(NUM_TEST_INSTANCES):
            set_seed(test_seed + test_idx)
            problem, instance_name = test_problem_generator()
            population = population_generator(problem)

            initial_fitnesses = [fitness(problem, sol) for sol in population]
            initial_best_fitness = min(initial_fitnesses)
            initial_mean_fitness = np.mean(initial_fitnesses)

            # Run multiple mutation episodes
            episode_best_fitnesses = []
            episode_best_diversities = []

            for episode in range(NUM_EPISODES_PER_TEST):
                # Select random solution from population
                random_idx = random.randint(0, len(population) - 1)
                current_solution = population[random_idx]

                # Track best solution across mutation steps
                best_episode_fitness = fitness(problem, current_solution)
                best_episode_solution = current_solution.clone()

                # Apply 100 mutation steps using NaiveMutation
                current_solution = naive_mutation.mutate(problem, current_solution, population)
                current_fitness = fitness(problem, current_solution)
                # Track best solution
                if current_fitness < best_episode_fitness:
                    best_episode_fitness = current_fitness
                    best_episode_solution = current_solution.clone()

                # Calculate diversity for best solution
                distances = [
                    distance_measure.edge_distance(best_episode_solution, pop_sol)
                    for pop_sol in population
                ]
                diversity_score = np.mean(distances)

                episode_best_fitnesses.append(best_episode_fitness)
                episode_best_diversities.append(diversity_score)

            # Overall best from all episodes
            final_best_fitness = min(episode_best_fitnesses)
            final_mean_fitness = np.mean(episode_best_fitnesses)
            final_best_diversity = max(episode_best_diversities)
            final_mean_diversity = np.mean(episode_best_diversities)

            improvement = initial_best_fitness - final_best_fitness
            improvement_pct = (improvement / initial_best_fitness * 100) if initial_best_fitness > 0 else 0

            baseline_results.append({
                'instance': instance_name,
                'initial_best': initial_best_fitness,
                'initial_mean': initial_mean_fitness,
                'final_best': final_best_fitness,
                'final_mean': final_mean_fitness,
                'final_best_diversity': final_best_diversity,
                'final_mean_diversity': final_mean_diversity,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

            print(f"  {test_idx+1}/{NUM_TEST_INSTANCES} {instance_name:15s} | "
                  f"Fitness: {initial_best_fitness:7.2f} -> {final_best_fitness:7.2f} | "
                  f"Δ={improvement:+7.2f} ({improvement_pct:+.1f}%) | "
                  f"Diversity: {final_mean_diversity:.3f}")

    # Baseline summary statistics
    print(f"\n{'='*80}")
    print("BASELINE SUMMARY")
    print(f"{'='*80}")

    baseline_avg_initial_best = np.mean([r['initial_best'] for r in baseline_results])
    baseline_avg_final_best = np.mean([r['final_best'] for r in baseline_results])
    baseline_avg_improvement = np.mean([r['improvement'] for r in baseline_results])
    baseline_avg_improvement_pct = np.mean([r['improvement_pct'] for r in baseline_results])
    baseline_avg_final_best_diversity = np.mean([r['final_best_diversity'] for r in baseline_results])
    baseline_avg_final_mean_diversity = np.mean([r['final_mean_diversity'] for r in baseline_results])

    print(f"Average initial best fitness: {baseline_avg_initial_best:.2f}")
    print(f"Average final best fitness: {baseline_avg_final_best:.2f}")
    print(f"Average improvement: {baseline_avg_improvement:+.2f} ({baseline_avg_improvement_pct:+.2f}%)")
    print(f"Average final best diversity: {baseline_avg_final_best_diversity:.3f}")
    print(f"Average final mean diversity: {baseline_avg_final_mean_diversity:.3f}")

    # ========================================
    # RL MODEL EVALUATION
    # ========================================
    print(f"\n{'='*80}")
    print("RL MODEL EVALUATION")
    print(f"{'='*80}")

    # Load and evaluate each model
    for model_path in MODEL_PATHS:
        if not os.path.exists(model_path):
            print(f"\nWarning: Model not found: {model_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_path}")
        print(f"{'='*80}")

        # Detect architecture
        try:
            model_is_ppo = detect_ppo_from_checkpoint(model_path)
            model_algorithm = "ppo" if model_is_ppo else "dqn"

            print(f"Detected: {model_algorithm.upper()}")
        except Exception as e:
            print(f"Error detecting architecture: {e}")
            continue

        # Create model instance
        eval_batch_size = args.dqn_batch_size if model_algorithm == "dqn" else args.ppo_batch_size

        eval_model = RLMutation(
            operators=create_operators(include_noop=args.include_noop),
            rl_algorithm=model_algorithm,
            hidden_dims=[128, 128, 64],
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=eval_batch_size,
            alpha=1.0,
            reward_strategy=REWARD_STRATEGY,
            acceptance_strategy=ACCEPTANCE_STRATEGY,
            max_steps=30,
            device="cuda",
            verbose=False
        )

        # Load weights
        try:
            eval_model.load(model_path)
            print(f"Successfully loaded model")
        except Exception as e:
            print(f"Error loading model: {e}")
            continue

        # Run evaluation
        rl_results = []

        for seed_idx, test_seed in enumerate(TEST_SEEDS):
            print(f"\nTest Seed {seed_idx+1}/{len(TEST_SEEDS)}: {test_seed}")

            for test_idx in range(NUM_TEST_INSTANCES):
                set_seed(test_seed + test_idx)
                problem, instance_name = test_problem_generator()
                population = population_generator(problem)

                initial_fitnesses = [fitness(problem, sol) for sol in population]
                initial_best_fitness = min(initial_fitnesses)
                initial_mean_fitness = np.mean(initial_fitnesses)

                # Run multiple mutation episodes
                episode_best_fitnesses = []
                episode_best_diversities = []

                for episode in range(NUM_EPISODES_PER_TEST):
                    # Select random solution from population
                    random_idx = random.randint(0, len(population) - 1)
                    solution = population[random_idx]

                    # Apply RL mutation (handles loop internally, returns best solution)
                    mutated_solution = eval_model.mutate(problem, solution, population)

                    # Calculate fitness and diversity
                    mutated_fitness = fitness(problem, mutated_solution)
                    distances = [
                        distance_measure.edge_distance(mutated_solution, pop_sol)
                        for pop_sol in population
                    ]
                    diversity_score = np.mean(distances)

                    episode_best_fitnesses.append(mutated_fitness)
                    episode_best_diversities.append(diversity_score)

                # Overall best from all episodes
                final_best_fitness = min(episode_best_fitnesses)
                final_mean_fitness = np.mean(episode_best_fitnesses)
                final_best_diversity = max(episode_best_diversities)  # Higher diversity is better
                final_mean_diversity = np.mean(episode_best_diversities)

                improvement = initial_best_fitness - final_best_fitness
                improvement_pct = (improvement / initial_best_fitness * 100) if initial_best_fitness > 0 else 0

                rl_results.append({
                    'instance': instance_name,
                    'initial_best': initial_best_fitness,
                    'initial_mean': initial_mean_fitness,
                    'final_best': final_best_fitness,
                    'final_mean': final_mean_fitness,
                    'final_best_diversity': final_best_diversity,
                    'final_mean_diversity': final_mean_diversity,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                })

                print(f"  {test_idx+1}/{NUM_TEST_INSTANCES} {instance_name:15s} | "
                      f"Fitness: {initial_best_fitness:7.2f} -> {final_best_fitness:7.2f} | "
                      f"Δ={improvement:+7.2f} ({improvement_pct:+.1f}%) | "
                      f"Diversity: {final_mean_diversity:.3f}")

        # Summary statistics
        print(f"\n{'='*80}")
        print("RL MODEL SUMMARY")
        print(f"{'='*80}")

        avg_initial_best = np.mean([r['initial_best'] for r in rl_results])
        avg_final_best = np.mean([r['final_best'] for r in rl_results])
        avg_improvement = np.mean([r['improvement'] for r in rl_results])
        avg_improvement_pct = np.mean([r['improvement_pct'] for r in rl_results])
        avg_final_best_diversity = np.mean([r['final_best_diversity'] for r in rl_results])
        avg_final_mean_diversity = np.mean([r['final_mean_diversity'] for r in rl_results])

        print(f"Average initial best fitness: {avg_initial_best:.2f}")
        print(f"Average final best fitness: {avg_final_best:.2f}")
        print(f"Average improvement: {avg_improvement:+.2f} ({avg_improvement_pct:+.2f}%)")
        print(f"Average final best diversity: {avg_final_best_diversity:.3f}")
        print(f"Average final mean diversity: {avg_final_mean_diversity:.3f}")

        # Per-instance breakdown
        instance_summary = {}
        for r in rl_results:
            inst = r['instance']
            if inst not in instance_summary:
                instance_summary[inst] = []
            instance_summary[inst].append(r)

        print(f"\nPer-Instance Results:")
        for inst in sorted(instance_summary.keys()):
            inst_results = instance_summary[inst]
            inst_avg_improvement = np.mean([r['improvement'] for r in inst_results])
            inst_avg_improvement_pct = np.mean([r['improvement_pct'] for r in inst_results])
            inst_avg_diversity = np.mean([r['final_mean_diversity'] for r in inst_results])
            inst_count = len(inst_results)
            print(f"  {inst:15s} ({inst_count:2d} tests): Δ={inst_avg_improvement:+7.2f} ({inst_avg_improvement_pct:+.1f}%) | Diversity: {inst_avg_diversity:.3f}")

    # ========================================
    # COMPARISON: RL vs BASELINE
    # ========================================
    print(f"\n{'='*80}")
    print("COMPARISON: RL MODEL vs NAIVE BASELINE")
    print(f"{'='*80}")

    # Calculate comparison metrics
    rl_avg_improvement = np.mean([r['improvement'] for r in rl_results])
    rl_avg_improvement_pct = np.mean([r['improvement_pct'] for r in rl_results])
    rl_avg_final_best = np.mean([r['final_best'] for r in rl_results])
    rl_avg_diversity = np.mean([r['final_mean_diversity'] for r in rl_results])

    # Improvement difference
    improvement_diff = rl_avg_improvement - baseline_avg_improvement
    improvement_pct_diff = rl_avg_improvement_pct - baseline_avg_improvement_pct
    fitness_diff = baseline_avg_final_best - rl_avg_final_best  # Positive means RL is better
    diversity_diff = rl_avg_diversity - baseline_avg_final_mean_diversity

    print(f"\nAverage Final Best Fitness:")
    print(f"  Baseline: {baseline_avg_final_best:.2f}")
    print(f"  RL Model: {rl_avg_final_best:.2f}")
    print(f"  Difference: {fitness_diff:+.2f} (RL {'better' if fitness_diff > 0 else 'worse'} by {abs(fitness_diff):.2f})")

    print(f"\nAverage Improvement:")
    print(f"  Baseline: {baseline_avg_improvement:+.2f} ({baseline_avg_improvement_pct:+.2f}%)")
    print(f"  RL Model: {rl_avg_improvement:+.2f} ({rl_avg_improvement_pct:+.2f}%)")
    print(f"  Difference: {improvement_diff:+.2f} ({improvement_pct_diff:+.2f}%)")

    print(f"\nAverage Final Diversity:")
    print(f"  Baseline: {baseline_avg_final_mean_diversity:.3f}")
    print(f"  RL Model: {rl_avg_diversity:.3f}")
    print(f"  Difference: {diversity_diff:+.3f} ({diversity_diff/baseline_avg_final_mean_diversity*100:+.2f}%)")

    # Win rate comparison
    wins = 0
    losses = 0
    ties = 0
    for rl_r, base_r in zip(rl_results, baseline_results):
        if rl_r['final_best'] < base_r['final_best']:
            wins += 1
        elif rl_r['final_best'] > base_r['final_best']:
            losses += 1
        else:
            ties += 1

    win_rate = wins / len(rl_results) * 100
    loss_rate = losses / len(rl_results) * 100
    tie_rate = ties / len(rl_results) * 100

    print(f"\nWin/Loss/Tie (RL vs Baseline):")
    print(f"  Wins:   {wins:3d} ({win_rate:.1f}%)")
    print(f"  Losses: {losses:3d} ({loss_rate:.1f}%)")
    print(f"  Ties:   {ties:3d} ({tie_rate:.1f}%)")


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
