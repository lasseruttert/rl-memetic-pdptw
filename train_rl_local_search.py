"""Example script for training and using RL-based local search."""

import sys
import os
import argparse
import random
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.instance_manager import InstanceManager
from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch
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
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
from memetic.solution_operators.two_opt import TwoOptOperator


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
    parser.add_argument("--seed", type=int, default=422,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # Configuration
    PROBLEM_SIZE = args.problem_size
    CATEGORIES = ['lc1', 'lc2', 'lr1', 'lr2']
    ACCEPTANCE_STRATEGY = args.acceptance_strategy
    REWARD_STRATEGY = args.reward_strategy
    SEED = args.seed
    seed_suffix = f"_seed{SEED}" if SEED is not None else ""
    RUN_NAME = f"rl_local_search_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}{seed_suffix}_{int(time.time())}"

    # Define operators to learn from
    # operators = [
    #     ReinsertOperator(max_attempts=1, clustered=False),
    #     ReinsertOperator(max_attempts=3, clustered=True),
    #     ReinsertOperator(allow_same_vehicle=False),
    #     RouteEliminationOperator(),
    #     SwapWithinOperator(),
    #     SwapBetweenOperator(),
    #     TransferOperator(single_route=True),
    #     CLSM1Operator(),
    #     CLSM2Operator(),
    # ]
    
    operators = [
        SwapWithinOperator(),
        # SwapWithinOperator(single_route=True),
        SwapBetweenOperator(),
        TransferOperator(),
        # TransferOperator(single_route=True),
        FlipOperator(),
        # FlipOperator(single_route=True),
        # TwoOptOperator(),
        ReinsertOperator(),
    ]

    # Initialize RL local search
    rl_local_search = RLLocalSearch(
        operators=operators,
        hidden_dims=[128, 128, 64],
        learning_rate=1e-4,
        gamma=0.90,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9975,
        target_update_interval=100,
        alpha=10.0,
        beta=0.0,
        acceptance_strategy=ACCEPTANCE_STRATEGY,
        reward_strategy = REWARD_STRATEGY,
        max_iterations=200,
        max_no_improvement=50,
        replay_buffer_capacity=100000,
        batch_size=64,
        device="cuda",
        verbose=True
    )

    # # Create problem and solution generators
    problem_generator = create_problem_generator(size=PROBLEM_SIZE, categories=CATEGORIES)

    # # Train the RL agent
    # print(f"Starting RL Local Search Training on size {PROBLEM_SIZE} instances...")
    # print(f"Categories: {CATEGORIES}")
    # print(f"Operators: {len(operators)}")

    # training_history = rl_local_search.train(
    #     problem_generator=problem_generator,
    #     initial_solution_generator=create_solution_generator,
    #     num_episodes=args.num_episodes,
    #     new_instance_interval=5,
    #     new_solution_interval=1,
    #     update_interval=1,
    #     warmup_episodes=10,
    #     save_interval=1000,
    #     save_path=f"models/rl_local_search_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}",
    #     tensorboard_dir=f"runs/{RUN_NAME}",
    #     seed=SEED, 
    # )

    # print("\nTraining completed!")
    # print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
    # print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")
    
    adaptive_local_search = AdaptiveLocalSearch(operators=operators, max_no_improvement=50, max_iterations=200)

    naive_local_search = NaiveLocalSearch(operators=operators, max_no_improvement=50, max_iterations=200, first_improvement=True)
    
    naive_with_best_local_search = NaiveLocalSearch(operators=operators, max_no_improvement=50, max_iterations=200, first_improvement=False)

    random_local_search = RandomLocalSearch(operators=operators, max_no_improvement=50, max_iterations=200)

    # You can provide up to 6 different RL model paths to evaluate here.
    RL_MODEL_PATHS = [
        f"models/rl_local_search_{PROBLEM_SIZE}_{ACCEPTANCE_STRATEGY}_{REWARD_STRATEGY}_final.pt",
    ]

    # Build RLLocalSearch instances for each provided path and try to load them.
    rl_models = []
    model_names = []
    for path in RL_MODEL_PATHS:
        if not path:
            continue
        # OneShot model
        model_oneshot = RLLocalSearch(
            operators=operators,
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
            device="cuda",
            verbose=False
        )
        # Roulette model (probabilistic sampling based on Q-values)
        model_roulette = RLLocalSearch(
            operators=operators,
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
            device="cuda",
            verbose=False
        )
        # Ranking model (strict Q-value order, best first)
        model_ranking = RLLocalSearch(
            operators=operators,
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
    NUM_TESTS = 100
    rl_results = [[] for _ in range(len(rl_models))]  # per-model collected tuples (initial, best, time)
    adaptive_results = []
    naive_results = []
    naive_with_best_results = []
    random_results = []

    print("\n--- Comparing RL models vs Naive/Random Local Search ---")
    for i in range(NUM_TESTS):
        print(f"\nTest {i+1}/{NUM_TESTS}")
        # Base seed for this test (ensures all methods see same randomness)
        base_seed = (SEED + i) if SEED is not None else None
        set_seed(base_seed)
        test_problem = problem_generator()
        initial_solution = create_solution_generator(test_problem)

        # Evaluate initial solution
        initial_fitness = fitness(test_problem, initial_solution)
        print(f"Initial fitness: {initial_fitness:.2f}")


        # Run each RL model
        for midx, model in enumerate(rl_models):
            if base_seed is not None:
                set_seed(base_seed)  # Reset to same seed for fair comparison
            rl_solution = initial_solution.clone()
            t0 = time.time()
            try:
                rl_best_solution, rl_best_fitness = model.search(
                    problem=test_problem,
                    solution=rl_solution,
                    epsilon=0.0
                )
            except Exception as e:
                print(f"Model {model_names[midx]} failed on test {i+1}: {e}")
                rl_best_fitness = float('inf')
            rl_time = time.time() - t0
            print(f"{model_names[midx]}: best fitness: {rl_best_fitness:.2f} (time: {rl_time:.2f}s)")
            rl_results[midx].append((initial_fitness, rl_best_fitness, rl_time))

        # Adaptive local search
        if base_seed is not None:
            set_seed(base_seed)
        adaptive_solution = initial_solution.clone()
        t0 = time.time()
        adaptive_best_solution, adaptive_best_fitness = adaptive_local_search.search(
            problem=test_problem,
            solution=adaptive_solution
        )
        adaptive_time = time.time() - t0
        print(f"Adaptive: best fitness: {adaptive_best_fitness:.2f} (time: {adaptive_time:.2f}s)")
        adaptive_results.append((initial_fitness, adaptive_best_fitness, adaptive_time))

        # Naive local search
        if base_seed is not None:
            set_seed(base_seed)
        naive_solution = initial_solution.clone()
        t0 = time.time()
        naive_best_solution, naive_best_fitness = naive_local_search.search(
            problem=test_problem,
            solution=naive_solution
        )
        naive_time = time.time() - t0
        print(f"Naive: best fitness: {naive_best_fitness:.2f} (time: {naive_time:.2f}s)")
        naive_results.append((initial_fitness, naive_best_fitness, naive_time))

        # Naive local search with best improvement
        if base_seed is not None:
            set_seed(base_seed)
        naive_with_best_solution = initial_solution.clone()
        t0 = time.time()
        naive_with_best_best_solution, naive_with_best_best_fitness = naive_with_best_local_search.search(
            problem=test_problem,
            solution=naive_with_best_solution
        )
        naive_with_best_time = time.time() - t0
        print(f"Naive (best improvement): best fitness: {naive_with_best_best_fitness:.2f} (time: {naive_with_best_time:.2f}s)")
        naive_with_best_results.append((initial_fitness, naive_with_best_best_fitness, naive_with_best_time))

        # Random local search
        if base_seed is not None:
            set_seed(base_seed)
        random_solution = initial_solution.clone()
        t0 = time.time()
        random_best_solution, random_best_fitness = random_local_search.search(
            problem=test_problem,
            solution=random_solution
        )
        random_time = time.time() - t0
        print(f"Random: best fitness: {random_best_fitness:.2f} (time: {random_time:.2f}s)")
        random_results.append((initial_fitness, random_best_fitness, random_time))

    # Summary per RL model
    print("\n--- Summary over tests ---")
    avg_initial = sum(r[0] for r in rl_results[0]) / NUM_TESTS if rl_results[0] else 0.0
    print(f"Average initial fitness: {avg_initial:.2f}")
    for midx, name in enumerate(model_names):
        avg_best = sum(r[1] for r in rl_results[midx]) / NUM_TESTS
        avg_time = sum(r[2] for r in rl_results[midx]) / NUM_TESTS
        print(f"Model {name}: Avg best fitness: {avg_best:.2f} (avg time: {avg_time:.2f}s)")

    # Summary for Adaptive, Naive, and Random
    avg_adaptive = sum(r[1] for r in adaptive_results) / NUM_TESTS
    avg_time_adaptive = sum(r[2] for r in adaptive_results) / NUM_TESTS
    avg_naive = sum(r[1] for r in naive_results) / NUM_TESTS
    avg_time_naive = sum(r[2] for r in naive_results) / NUM_TESTS
    avg_naive_with_best = sum(r[1] for r in naive_with_best_results) / NUM_TESTS
    avg_time_naive_with_best = sum(r[2] for r in naive_with_best_results) / NUM_TESTS
    avg_random = sum(r[1] for r in random_results) / NUM_TESTS
    avg_time_random = sum(r[2] for r in random_results) / NUM_TESTS
    print(f"Adaptive: Avg best fitness: {avg_adaptive:.2f} (avg time: {avg_time_adaptive:.2f}s)")
    print(f"Naive: Avg best fitness: {avg_naive:.2f} (avg time: {avg_time_naive:.2f}s)")
    print(f"Naive (best improvement): Avg best fitness: {avg_naive_with_best:.2f} (avg time: {avg_time_naive_with_best:.2f}s)")
    print(f"Random: Avg best fitness: {avg_random:.2f} (avg time: {avg_time_random:.2f}s)")


if __name__ == "__main__":
    main()
