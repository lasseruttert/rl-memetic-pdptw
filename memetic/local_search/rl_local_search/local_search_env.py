"""Gymnasium environment for RL-based local search."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from utils.pdptw_problem import PDPTWProblem, Node
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness

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
            acceptance_strategy: "greedy" (only accept improvements) or "always"
            reward_strategy: Strategy for calculating rewards
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

        # Initialize operator metrics (needed for dimension inference)
        self.operator_metrics = [{} for _ in self.operators]

        # Episode state
        self.problem: Optional[PDPTWProblem] = None
        self.current_solution: Optional[PDPTWSolution] = None
        self.current_fitness: Optional[float] = None
        self.step_count: int = 0
        self.initial_fitness: Optional[float] = None
        self.best_solution: Optional[PDPTWSolution] = None
        self.best_fitness: float = float('inf')
        
        # Infer observation space dimensions dynamically
        dummy_problem = self._create_minimal_problem()
        dummy_solution = self._create_minimal_solution(dummy_problem)
        solution_feature_dim = len(self._extract_solution_features(dummy_problem, dummy_solution))
        operator_feature_dim = len(self._get_operator_features().flatten())
        obs_dim = solution_feature_dim + operator_feature_dim

        # Observation space: feature vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )


        self.epsilon_value = 0.1

        # Simulated annealing parameters
        self.initial_temp = 1000.0
        self.temp_decay = 0.995
        self.temperature = self.initial_temp

        # Running reward normalization statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_samples = 0
        self.max_reward_samples = 100000  # Stop updating after this many samples

        # Late acceptance parameters
        self.late_acceptance_length = 20  # History buffer size L
        self.fitness_history = []
        self.operator_history = []

        # Rising epsilon parameters
        self.rising_epsilon_start = 0.05
        self.rising_epsilon_end = 0.5

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
        self.step_count = 0
        self.initial_fitness = self.current_fitness
        self.best_solution = self.current_solution.clone()
        self.best_fitness = self.current_fitness

        # Reset temperature for simulated annealing
        self.temperature = self.initial_temp

        # Clear fitness history for late acceptance
        self.fitness_history = []
        
        self.operator_history = []
        
        self.operator_metrics = [{} for _ in self.operators]

        observation = self._get_state()

        info = {
            'fitness': self.current_fitness,
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

        # Calculate new fitness
        new_fitness = fitness(self.problem, new_solution)

        # Calculate reward 
        reward = self._calculate_reward(self.current_solution, new_solution, self.current_fitness, new_fitness)

        # Update best solution
        if new_fitness < self.best_fitness:
            self.best_solution = new_solution.clone()
            self.best_fitness = new_fitness

        # Accept solution based on strategy
        accepted = self._accept_solution(self.current_fitness, new_fitness)

        # Track fitness improvement for this step
        fitness_improvement = self.current_fitness - new_fitness

        if accepted:
            self.current_solution = new_solution
            self.current_fitness = new_fitness
        
        self.operator_history.append(action)
        # Update operator metrics
        self.operator_metrics[action]['applications'] = self.operator_metrics[action].get('applications', 0) + 1
        self.operator_metrics[action]['improvements'] = self.operator_metrics[action].get('improvements', 0) + (1 if fitness_improvement > 0 else 0)
        if accepted:
            self.operator_metrics[action]['acceptances'] = self.operator_metrics[action].get('acceptances', 0) + 1
            self.operator_metrics[action]['total_improvement'] = self.operator_metrics[action].get('total_improvement', 0.0) + fitness_improvement

        # Increment step counter
        self.step_count += 1

        # Update acceptance strategy state
        if self.acceptance_strategy == "simulated_annealing":
            self.temperature *= self.temp_decay

        # Maintain fitness history for late acceptance
        self.fitness_history.append(self.current_fitness)
        if len(self.fitness_history) > self.late_acceptance_length:
            self.fitness_history.pop(0)

        # Check termination
        terminated = False  # local search doesn't have a terminal state
        truncated = self.step_count >= self.max_steps

        # Observation and info
        observation = self._get_state()

        info = {
            'fitness': self.current_fitness,
            'new_fitness': new_fitness,
            'accepted': accepted,
            'step': self.step_count,
            'best_fitness': self.best_fitness,
            'operator': operator.name if hasattr(operator, 'name') else f"Operator{action}",
            'fitness_improvement': fitness_improvement,
        }

        return observation, reward, terminated, truncated, info

    def _update_reward_stats(self, raw_reward: float) -> None:
        """Update running reward statistics using online algorithm.

        Args:
            raw_reward: Raw reward value to incorporate into statistics
        """
        if self.reward_samples >= self.max_reward_samples:
            return  # Stop updating after max samples

        self.reward_samples += 1

        # Online mean and std update (Welford's algorithm)
        delta = raw_reward - self.reward_mean
        self.reward_mean += delta / self.reward_samples
        delta2 = raw_reward - self.reward_mean

        if self.reward_samples > 1:
            # Update variance
            variance = ((self.reward_samples - 2) * (self.reward_std ** 2) + delta * delta2) / (self.reward_samples - 1)
            self.reward_std = np.sqrt(max(variance, 1e-8))

    def _calculate_reward(
        self,
        old_solution: PDPTWSolution,
        new_solution: PDPTWSolution,
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
                
        elif self.reward_strategy == "distance_baseline_clipped":
            baseline = self.problem.distance_baseline
            fitness_improvement = (old_fitness - new_fitness) / baseline
            reward = np.clip(self.alpha * fitness_improvement, -20.0, 20.0)
            
        elif self.reward_strategy == "binary":
            if new_fitness < old_fitness:
                reward = 1.0
            else:
                reward = -1.0
                
        elif self.reward_strategy == "tanh":
            fitness_improvement = (old_fitness - new_fitness) / 1
            reward = np.tanh(self.alpha * fitness_improvement)

        elif self.reward_strategy == "distance_baseline_normalized":
            # Two-stage: baseline normalization + z-score (for mixed instance sizes)
            baseline = self.problem.distance_baseline
            raw_improvement = (old_fitness - new_fitness) / baseline

            self._update_reward_stats(raw_improvement)

            if self.reward_samples > 1:
                reward = np.clip((raw_improvement - self.reward_mean) / (self.reward_std + 1e-8), -1.0, 1.0)
            else:
                reward = raw_improvement

        elif self.reward_strategy == "pure_normalized":
            # Pure z-score normalization (for single instance size training)
            raw_improvement = old_fitness - new_fitness

            self._update_reward_stats(raw_improvement)

            if self.reward_samples > 1:
                reward = np.clip((raw_improvement - self.reward_mean) / (self.reward_std + 1e-8), -1.0, 1.0)
            else:
                reward = raw_improvement 

        elif self.reward_strategy == "distance_baseline_tanh":
            baseline = self.problem.distance_baseline
            fitness_improvement = (old_fitness - new_fitness) / baseline
            reward = np.tanh(self.alpha * fitness_improvement)

        elif self.reward_strategy == "distance_baseline_asymmetric_tanh":
            # Distance baseline + asymmetric tanh (penalize degradations more)
            baseline = self.problem.distance_baseline
            fitness_improvement = (old_fitness - new_fitness) / baseline

            # Apply asymmetric scaling
            if fitness_improvement > 0:  # Improvement
                reward = np.tanh(self.alpha * fitness_improvement)
            else:  # Degradation - penalize 2x more
                reward = np.tanh(self.alpha * fitness_improvement * 2.0)
                
        elif self.reward_strategy == "another_one":
            if old_fitness <= 0 or self.initial_fitness <= 0:
                reward = 0.0
            else:
                old_improvement = (old_fitness - new_fitness) / old_fitness
                old_improvement = np.clip(old_improvement, -1.0, 1.0)
                initial_improvement = (self.initial_fitness - new_fitness) / self.initial_fitness
                initial_improvement = np.clip(initial_improvement, -1.0, 1.0)
                reward = self.alpha * max(old_improvement, initial_improvement)
                
        elif self.reward_strategy == "component":
            old_total_distance = old_solution.total_distance
            old_num_vehicles = old_solution.num_vehicles_used
            old_feasible = old_solution.is_feasible

            new_total_distance = new_solution.total_distance
            new_num_vehicles = new_solution.num_vehicles_used
            new_feasible = new_solution.is_feasible

            # Weights for different components
            w_distance = 1.0
            w_vehicles = 0.0
            w_feasibility = 10.0
            # Calculate components
            distance_improvement = 1 if new_total_distance < old_total_distance else -1
            vehicle_improvement = 1 if new_num_vehicles < old_num_vehicles else -1
            feasibility_improvement = int(new_feasible) - int(old_feasible)

            # Component contributions to reward
            distance_component = w_distance * distance_improvement
            vehicle_component = w_vehicles * vehicle_improvement
            feasibility_component = w_feasibility * feasibility_improvement

            # Total reward
            reward = distance_component + vehicle_component + feasibility_component

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

        elif self.acceptance_strategy == "late_acceptance":
            # Accept if better than fitness from L steps ago
            if len(self.fitness_history) < self.late_acceptance_length:
                # Greedy until buffer fills
                return new_fitness <= old_fitness
            # Compare to oldest fitness in history (L steps ago)
            return new_fitness <= self.fitness_history[0]

        elif self.acceptance_strategy == "rising_epsilon_greedy":
            # Always accept improvements
            if new_fitness < old_fitness:
                return True
            # Epsilon rises from start to end over max_steps
            progress = min(self.step_count / self.max_steps, 1.0)
            epsilon = self.rising_epsilon_start + (self.rising_epsilon_end - self.rising_epsilon_start) * progress
            return np.random.rand() < epsilon

        else:
            raise ValueError(f"Unknown acceptance strategy: {self.acceptance_strategy}")
        
    def _get_operator_features(self) -> np.ndarray:
        """Extract features for each operator based on historical performance.

        Returns:
            np.ndarray: Feature matrix of shape (num_operators, num_features)
        """
        features = []
        for metrics in self.operator_metrics:
            applications = metrics.get('applications', 0)
            improvements = metrics.get('improvements', 0)
            acceptances = metrics.get('acceptances', 0)
            total_improvement = metrics.get('total_improvement', 0.0)

            success_rate = improvements / applications if applications > 0 else 0.0
            acceptance_rate = acceptances / applications if applications > 0 else 0.0
            avg_improvement = total_improvement / acceptances if acceptances > 0 else 0.0

            features.append([
                applications / self.step_count if self.step_count > 0 else 0.0,
                improvements / self.step_count if self.step_count > 0 else 0.0,
                acceptances / self.step_count if self.step_count > 0 else 0.0,
                success_rate,
                acceptance_rate,
            ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_solution_features(self, problem: PDPTWProblem, solution: PDPTWSolution) -> np.ndarray:
        """Extract feature vector from a solution for RL state representation.

        Features include:
        - Problem features: num_requests, vehicle_capacity, avg_distance, time_window_tightness
        - Solution features: num_routes, num_customers_served, total_distance, route statistics

        Args:
            problem: PDPTW problem instance
            solution: PDPTW solution

        Returns:
            np.ndarray: Feature vector
        """
        # Problem features
        num_requests = problem.num_requests
        vehicle_capacity = problem.vehicle_capacity
        num_vehicles = problem.num_vehicles
        avg_distance = np.mean(problem.distance_matrix[problem.distance_matrix > 0])

        # Time window tightness: average ratio of time window span to total time horizon
        time_window_spans = problem.time_windows[:, 1] - problem.time_windows[:, 0]
        max_time_horizon = np.max(problem.time_windows[:, 1])
        avg_tw_tightness = np.mean(time_window_spans) / max_time_horizon if max_time_horizon > 0 else 0

        # Solution features
        num_routes = solution.num_vehicles_used
        num_customers_served = solution.num_customers_served
        total_distance = solution.total_distance
        feasible = solution.is_feasible

        # Route statistics
        route_lengths = [len(r) - 2 for r in solution.routes if len(r) > 2]  # excluding depot
        avg_route_length = np.mean(route_lengths) if route_lengths else 0
        max_route_length = np.max(route_lengths) if route_lengths else 0
        min_route_length = np.min(route_lengths) if route_lengths else 0
        std_route_length = np.std(route_lengths) if route_lengths else 0

        # Route distance statistics
        route_distances = [solution.route_lengths[i] for i in range(len(solution.routes)) if len(solution.routes[i]) > 2]
        avg_route_distance = np.mean(route_distances) if route_distances else 0
        max_route_distance = np.max(route_distances) if route_distances else 0
        std_route_distance = np.std(route_distances) if route_distances else 0


        # Normalized features
        features = np.array([
            # Problem features
            # num_requests,
            # vehicle_capacity,
            # num_vehicles,
            # avg_distance,
            # avg_tw_tightness,

            # Solution features (normalized)
            num_routes / num_vehicles if num_vehicles > 0 else 0,
            num_customers_served / (num_requests * 2) if num_requests > 0 else 0,  # *2 for pickup+delivery
            total_distance / problem.distance_baseline if problem.distance_baseline > 0 else 0,
            int(feasible),

            # Route statistics (normalized)
            avg_route_length / num_requests if num_requests > 0 else 0,
            max_route_length / num_requests if num_requests > 0 else 0,
            min_route_length / num_requests if num_requests > 0 else 0,
            std_route_length / num_requests if num_requests > 0 else 0,
            avg_route_distance / problem.distance_baseline if problem.distance_baseline > 0 else 0,
            max_route_distance / problem.distance_baseline if problem.distance_baseline > 0 else 0,
            std_route_distance / problem.distance_baseline if problem.distance_baseline > 0 else 0,

        ], dtype=np.float32)

        return features
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation.

        Returns:
            np.ndarray: Combined feature vector of solution and operators
        """
        if self.problem is None or self.current_solution is None:
            raise RuntimeError("Environment must be reset before getting state")

        solution_features = self._extract_solution_features(self.problem, self.current_solution)
        operator_features = self._get_operator_features().flatten()

        # Combine features
        state = np.concatenate([solution_features, operator_features])

        return state

    def _create_minimal_problem(self) -> PDPTWProblem:
        """Create minimal dummy problem for dimension inference.

        Returns:
            Minimal PDPTWProblem instance
        """
        node_depot = Node(index=0, x=0.0, y=0.0, demand=0, time_window=(0, 1000), service_time=0, pickup_index=0, delivery_index=0)
        node_pickup = Node(index=1, x=1.0, y=1.0, demand=1, time_window=(0, 1000), service_time=0, pickup_index=0, delivery_index=2)
        node_delivery = Node(index=2, x=2.0, y=2.0, demand=-1, time_window=(0, 1000), service_time=0, pickup_index=1, delivery_index=0)
        return PDPTWProblem(
            nodes=[node_depot, node_pickup, node_delivery],
            vehicle_capacity=10,
            num_vehicles=10
        )

    def _create_minimal_solution(self, problem: PDPTWProblem) -> PDPTWSolution:
        """Create minimal dummy solution for dimension inference.

        Args:
            problem: Problem instance to create solution for

        Returns:
            Minimal PDPTWSolution instance
        """
        return PDPTWSolution(
            routes=[[0, 1, 2, 0]],
            problem=problem
        )

    def get_best_solution(self) -> Tuple[PDPTWSolution, float]:
        """Get the best solution found during the episode.

        Returns:
            Tuple of (best_solution, best_fitness)
        """
        return self.best_solution, self.best_fitness

    def get_reward_stats(self) -> dict:
        """Get current reward normalization statistics.

        Returns:
            Dictionary with mean, std, and sample count
        """
        return {
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_samples': self.reward_samples
        }

    def set_reward_stats(self, stats: dict) -> None:
        """Set reward normalization statistics (for loading saved state).

        Args:
            stats: Dictionary with 'reward_mean', 'reward_std', 'reward_samples'
        """
        self.reward_mean = stats.get('reward_mean', 0.0)
        self.reward_std = stats.get('reward_std', 1.0)
        self.reward_samples = stats.get('reward_samples', 0)
