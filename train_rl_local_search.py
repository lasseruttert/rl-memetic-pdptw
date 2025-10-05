"""Example script for training and using RL-based local search."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.instance_manager import InstanceManager
from memetic.local_search.rl_local_search.rl_local_search import RLLocalSearch
from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.solution_generators.random_generator import RandomGenerator
import random
import time

# fitness function for initial solution evaluation
from memetic.fitness.fitness import fitness

# Import operators
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator


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


def main():
    """Main training script for RL local search."""

    # Configuration
    PROBLEM_SIZE = 100  # Train on 100-request instances
    CATEGORIES = ['lc1', 'lr1']  # Mix of clustered and random instances

    # Define operators to learn from
    operators = [
        ReinsertOperator(max_attempts=1, clustered=False),
        ReinsertOperator(max_attempts=3, clustered=True),
        ReinsertOperator(allow_same_vehicle=False),
        RouteEliminationOperator(),
        SwapWithinOperator(),
        SwapBetweenOperator(),
        TransferOperator(single_route=True),
        CLSM1Operator(),
        CLSM2Operator(),
    ]

    # Initialize RL local search
    rl_local_search = RLLocalSearch(
        operators=operators,
        hidden_dims=[128, 128, 64],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_interval=100,
        alpha=10.0,  # Fitness weight (higher for stronger signal)
        beta=0.0,    # Not used anymore (removed from environment)
        acceptance_strategy="always",  # Important for training!
        max_steps_per_episode=200,
        replay_buffer_capacity=100000,
        batch_size=64,
        device="cuda",  # or "cpu"
        verbose=True
    )

    # # Create problem and solution generators
    problem_generator = create_problem_generator(size=PROBLEM_SIZE, categories=CATEGORIES)

    # # Train the RL agent
    print(f"Starting RL Local Search Training on size {PROBLEM_SIZE} instances...")
    print(f"Categories: {CATEGORIES}")
    print(f"Operators: {len(operators)}")

    training_history = rl_local_search.train(
        problem_generator=problem_generator,
        initial_solution_generator=create_solution_generator,
        num_episodes=2000,          # More episodes for better learning
        new_instance_interval=10,   # NEW: More diverse instances (200 different ones)
        new_solution_interval=3,    # NEW: More diverse starting points
        update_interval=1,          # update policy every step
        warmup_episodes=10,         # start training after 10 episodes
        save_interval=1000,          # save checkpoint every 100 episodes
        save_path=f"models/rl_local_search_{PROBLEM_SIZE}"
    )

    print("\nTraining completed!")
    print(f"Final average reward: {sum(training_history['episode_rewards'][-100:]) / 100:.2f}")
    # print(f"Final average fitness: {sum(training_history['episode_best_fitness'][-100:]) / 100:.2f}")

    naive_local_search = NaiveLocalSearch(operators=operators, max_no_improvement=50, max_iterations=200, first_improvement=True)

    final_rl_local_search = RLLocalSearch(
        operators=operators,
        hidden_dims=[128, 128, 64],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_interval=100,
        alpha=10.0,  # Same as training (not actually used during inference)
        beta=0.0,    # Not used anymore
        acceptance_strategy="greedy",  # Greedy for inference/testing
        max_steps_per_episode=200,
        replay_buffer_capacity=100000,
        batch_size=64,
        device="cuda",  # or "cpu"
        verbose=False  # Less verbose during testing
    )
    
    final_rl_local_search.load(f"models/rl_local_search_{PROBLEM_SIZE}_final.pt")

    # Compare RL local search vs Naive local search on 5 problems
    NUM_TESTS = 5
    rl_results = []  # tuples of (initial, rl_fitness, rl_time)
    naive_results = []

    print("\n--- Comparing RL Local Search vs Naive Local Search ---")
    for i in range(NUM_TESTS):
        print(f"\nTest {i+1}/{NUM_TESTS}")
        test_problem = problem_generator()
        initial_solution = create_solution_generator(test_problem)

        # Evaluate initial solution
        initial_fitness = fitness(test_problem, initial_solution)
        print(f"Initial fitness: {initial_fitness:.2f}")

        # RL local search (start from a fresh clone)
        rl_solution = initial_solution.clone()
        t0 = time.time()
        rl_best_solution, rl_best_fitness = final_rl_local_search.search(
            problem=test_problem,
            solution=rl_solution,
            max_iterations=200,
            epsilon=0.0  # greedy (no exploration)
        )
        rl_time = time.time() - t0
        print(f"RL: best fitness: {rl_best_fitness:.2f} (time: {rl_time:.2f}s)")

        # Naive local search (start from a fresh clone)
        naive_solution = initial_solution.clone()
        t0 = time.time()
        naive_best_solution, naive_best_fitness = naive_local_search.search(
            problem=test_problem,
            solution=naive_solution
        )
        naive_time = time.time() - t0
        print(f"Naive: best fitness: {naive_best_fitness:.2f} (time: {naive_time:.2f}s)")

        rl_results.append((initial_fitness, rl_best_fitness, rl_time))
        naive_results.append((initial_fitness, naive_best_fitness, naive_time))

    # Summary
    avg_initial = sum(r[0] for r in rl_results) / NUM_TESTS
    avg_rl = sum(r[1] for r in rl_results) / NUM_TESTS
    avg_naive = sum(r[1] for r in naive_results) / NUM_TESTS
    avg_time_rl = sum(r[2] for r in rl_results) / NUM_TESTS
    avg_time_naive = sum(r[2] for r in naive_results) / NUM_TESTS

    print("\n--- Summary over tests ---")
    print(f"Average initial fitness: {avg_initial:.2f}")
    print(f"Average RL best fitness: {avg_rl:.2f} (avg time: {avg_time_rl:.2f}s)")
    print(f"Average Naive best fitness: {avg_naive:.2f} (avg time: {avg_time_naive:.2f}s)" )

    # Save final model
    final_path = f"models/rl_local_search_{PROBLEM_SIZE}_final.pt"
    final_rl_local_search.save(final_path)
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":
    main()
