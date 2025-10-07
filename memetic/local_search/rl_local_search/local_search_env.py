"""Gymnasium environment for RL-based local search."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness
from memetic.local_search.rl_local_search.rl_utils import extract_solution_features, calculate_constraint_violations


class LocalSearchEnv(gym.Env):
    """Gymnasium environment for learning local search operator selection.

    The agent selects which operator to apply to the current solution.
    The reward is based on fitness improvement and feasibility changes.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        operators: list[BaseOperator],
        alpha: float = 1.0,
        acceptance_strategy: str = "greedy",
        reward_strategy: str = "initial_improvement",
        max_steps: int = 100
    ):
        """Initialize the local search environment.

        Args:
            operators: List of local search operators to choose from
            alpha: Weight for fitness improvement in reward function
            beta: Weight for feasibility improvement in reward function
            acceptance_strategy: "greedy" (only accept improvements) or "always"
            max_steps: Maximum number of steps per episode
        """
        super().__init__()

        self.operators = operators
        self.alpha = alpha
        self.acceptance_strategy = acceptance_strategy
        self.reward_strategy = reward_strategy
        self.max_steps = max_steps

        # Action space: discrete selection of operators (no no-op)
        self.action_space = spaces.Discrete(len(operators))

        # Observation space: feature vector 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(22,),
            dtype=np.float32
        )

        # Episode state
        self.problem: Optional[PDPTWProblem] = None
        self.current_solution: Optional[PDPTWSolution] = None
        self.current_fitness: Optional[float] = None
        self.current_violations: Optional[dict] = None
        self.step_count: int = 0
        self.initial_fitness: Optional[float] = None
        self.best_solution: Optional[PDPTWSolution] = None
        self.best_fitness: float = float('inf')

        self.epsilon_value = 0.1

        # Simulated annealing parameters
        self.initial_temp = 1000.0
        self.temp_decay = 0.995
        self.temperature = self.initial_temp

    def reset(
        self,
        problem: PDPTWProblem,
        initial_solution: PDPTWSolution,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment with a new problem and initial solution.

        Args:
            problem: PDPTW problem instance
            initial_solution: Initial solution to start local search from
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self.problem = problem
        self.current_solution = initial_solution.clone()
        self.current_fitness = fitness(problem, self.current_solution)
        self.current_violations = calculate_constraint_violations(problem, self.current_solution)
        self.step_count = 0
        self.initial_fitness = self.current_fitness
        self.best_solution = self.current_solution.clone()
        self.best_fitness = self.current_fitness

        # Reset temperature for simulated annealing
        self.temperature = self.initial_temp

        observation = extract_solution_features(problem, self.current_solution)
        
        info = {
            'fitness': self.current_fitness,
            'violations': self.current_violations,
            'step': self.step_count
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Index of the operator to apply

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.problem is None or self.current_solution is None:
            raise RuntimeError("Environment must be reset before calling step()")

        # Apply selected operator
        operator = self.operators[action]
        new_solution = operator.apply(self.problem, self.current_solution)

        # Calculate new fitness and violations
        new_fitness = fitness(self.problem, new_solution)
        new_violations = calculate_constraint_violations(self.problem, new_solution)

        # Calculate reward
        reward = self._calculate_reward(
            self.current_fitness,
            new_fitness
        )

        # Update best solution
        if new_fitness < self.best_fitness:
            self.best_solution = new_solution.clone()
            self.best_fitness = new_fitness

        # Accept solution based on strategy
        accepted = self._accept_solution(self.current_fitness, new_fitness)
        if accepted:
            self.current_solution = new_solution
            self.current_fitness = new_fitness
            self.current_violations = new_violations

        # Increment step counter
        self.step_count += 1

        # Update acceptance strategy state
        if self.acceptance_strategy == "simulated_annealing":
            self.temperature *= self.temp_decay


        # Check termination
        terminated = False  # local search doesn't have a terminal state
        truncated = self.step_count >= self.max_steps

        # Observation and info
        observation = extract_solution_features(self.problem, self.current_solution)

        info = {
            'fitness': self.current_fitness,
            'new_fitness': new_fitness,
            'violations': self.current_violations,
            'accepted': accepted,
            'step': self.step_count,
            'best_fitness': self.best_fitness,
            'operator': operator.name if hasattr(operator, 'name') else f"Operator{action}"
        }

        return observation, reward, terminated, truncated, info

    def _calculate_reward(
        self,
        old_fitness: float,
        new_fitness: float,
    ) -> float:
        """Calculate reward based on fitness improvement.

        Args:
            old_fitness: Fitness of current solution (includes penalties)
            new_fitness: Fitness of new solution (includes penalties)

        Returns:
            Reward value (normalized)
        """
        if self.reward_strategy == "initial_improvement":
            if self.initial_fitness <= 0:
                reward = 0.0
            else:
                improvement_pct = (self.initial_fitness - new_fitness) / self.initial_fitness if self.initial_fitness > 0 else 0.0
                reward = self.alpha * improvement_pct
                
        elif self.reward_strategy == "old_improvement":
            if old_fitness <= 0:
                reward = 0.0
            else:
                improvement_pct = (old_fitness - new_fitness) / old_fitness if old_fitness > 0 else 0.0
                reward = self.alpha * improvement_pct
                
        elif self.reward_strategy == "hybrid_improvement":
            if self.initial_fitness <= 0 or old_fitness <= 0:
                reward = 0.0
            else:
                initial_improvement = (self.initial_fitness - new_fitness) / self.initial_fitness
                old_improvement = (old_fitness - new_fitness) / old_fitness
                reward = self.alpha * (0.5 * initial_improvement + 0.5 * old_improvement)
                
        elif self.reward_strategy == "distance_baseline":
            baseline = self.problem.distance_baseline
            fitness_improvement = (old_fitness - new_fitness) / baseline
            reward = self.alpha * fitness_improvement
            
        elif self.reward_strategy == "log_improvement":
            if new_fitness < old_fitness:
                reward = self.alpha * np.log(old_fitness / new_fitness)
            else:
                reward = -self.alpha * np.log(new_fitness / old_fitness)
                
        elif self.reward_strategy == "binary":
            if new_fitness < old_fitness:
                reward = 1.0
            else:
                reward = -1.0

        elif self.reward_strategy == "distance_baseline_clipped":
            baseline = self.problem.distance_baseline
            fitness_improvement = (old_fitness - new_fitness) / baseline
            reward = np.clip(self.alpha * fitness_improvement, -5.0, 5.0)

        else:
            raise ValueError(f"Unknown reward strategy: {self.reward_strategy}")

        return reward

    def _accept_solution(self, old_fitness: float, new_fitness: float) -> bool:
        """Determine whether to accept the new solution.

        Args:
            old_fitness: Fitness of current solution
            new_fitness: Fitness of new solution

        Returns:
            True if solution should be accepted, False otherwise
        """
        if self.acceptance_strategy == "greedy":
            return new_fitness < old_fitness

        elif self.acceptance_strategy == "always":
            return True

        elif self.acceptance_strategy == "epsilon_greedy":
            if new_fitness < old_fitness:
                return True
            else:
                return np.random.rand() < self.epsilon_value

        elif self.acceptance_strategy == "simulated_annealing":
            if new_fitness < old_fitness:
                return True
            else:
                delta = new_fitness - old_fitness
                acceptance_prob = np.exp(-delta / self.temperature)
                return np.random.rand() < acceptance_prob
        else:
            raise ValueError(f"Unknown acceptance strategy: {self.acceptance_strategy}")

    def get_best_solution(self) -> Tuple[PDPTWSolution, float]:
        """Get the best solution found during the episode.

        Returns:
            Tuple of (best_solution, best_fitness)
        """
        return self.best_solution, self.best_fitness
