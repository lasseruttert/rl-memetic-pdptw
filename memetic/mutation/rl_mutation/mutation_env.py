import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from utils.pdptw_problem import PDPTWProblem, Node
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness
from memetic.utils.distance_measure import DistanceMeasure

class MutationEnv(gym.Env):
    """Gymnasium environment for learning mutation operator selection.

    The agent selects which mutation operator to apply to a solution within a population.
    The reward considers both solution improvement and population-level effects.

    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        operators: list[BaseOperator],
        alpha: float = 1.0,
        acceptance_strategy: str = "greedy",
        reward_strategy: str = "binary",
        max_steps: int = 100,
        max_no_improvement: Optional[int] = None
    ):
        """Initialize the mutation environment.

        Args:
            operators: List of mutation operators to choose from
            alpha: Weight for fitness improvement in reward function
            acceptance_strategy: "greedy" (only accept improvements), "always", etc.
            reward_strategy: Strategy for calculating rewards (for future use)
            max_steps: Maximum number of steps per episode
            max_no_improvement: Early stopping after N steps without improvement (None to disable)
        """
        super().__init__()

        self.operators = operators
        self.alpha = alpha
        self.acceptance_strategy = acceptance_strategy
        self.reward_strategy = reward_strategy
        self.max_steps = max_steps
        self.max_no_improvement = max_no_improvement

        # Action space: discrete selection of operators (including no-op if provided)
        self.action_space = spaces.Discrete(len(operators))

        # Distance measure for diversity calculations
        self.distance_measure = DistanceMeasure()

        # Initialize operator metrics
        self.operator_metrics = [{} for _ in self.operators]

        # Episode state
        self.problem: Optional[PDPTWProblem] = None
        self.population: Optional[list[PDPTWSolution]] = None
        self.solution_index: Optional[int] = None  # Index of solution in population
        self.current_solution: Optional[PDPTWSolution] = None
        self.current_measures: Optional[dict] = None
        self.step_count: int = 0
        self.initial_measures: Optional[dict] = None
        self.best_solution: Optional[PDPTWSolution] = None
        self.best_measures: Optional[dict] = None  # Track comprehensive measures of best solution

        # Cached population statistics
        self.population_fitnesses: Optional[list[float]] = None
        self.population_num_vehicles: Optional[list[int]] = None
        self.cached_population_diversity: Optional[float] = None

        # Early stopping tracking
        self.no_improvement_count: int = 0
        self.last_best_measures: Optional[dict] = None

        # Infer observation space dimensions dynamically
        dummy_problem = self._create_minimal_problem()
        dummy_solution = self._create_minimal_solution(dummy_problem)
        dummy_population = [dummy_solution.clone() for _ in range(3)]

        solution_features = self._get_solution_features(dummy_problem, dummy_solution)
        population_features = self._get_population_features(
            dummy_population,
            [fitness(dummy_problem, s) for s in dummy_population],
            [s.num_vehicles_used for s in dummy_population]
        )
        operator_features = self._get_operator_features()

        # Store feature dimensions
        self.solution_feature_dim = len(solution_features)
        self.population_feature_dim = len(population_features)
        self.operator_feature_dim_per_op = operator_features.shape[1]

        obs_dim = self.solution_feature_dim + self.population_feature_dim + len(operator_features.flatten())

        # Observation space: feature vector
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(
        self,
        problem: PDPTWProblem,
        population: list[PDPTWSolution],
        fitnesses: list[float],
        num_vehicles: list[int],
        solution: PDPTWSolution,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment with a new problem, population, and solution to mutate.

        Args:
            problem: PDPTW problem instance
            population: Current population of solutions
            solution: Solution from population to mutate
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Validate inputs
        if fitnesses is None or num_vehicles is None:
            raise ValueError("fitnesses and num_vehicles must be provided (not None)")
        if len(fitnesses) != len(population) or len(num_vehicles) != len(population):
            raise ValueError(f"Length mismatch: population={len(population)}, fitnesses={len(fitnesses)}, num_vehicles={len(num_vehicles)}")

        self.problem = problem
        # Store population as mutable list (will be updated on acceptance)
        self.population = [s.clone() for s in population]

        # Find the index of the solution in the population (once)
        self.solution_index = None
        for i, sol in enumerate(population):
            if sol.hashed_encoding == solution.hashed_encoding:
                self.solution_index = i
                break

        if self.solution_index is None:
            raise ValueError("Solution not found in population")

        self.current_solution = solution.clone()
        self.current_measures = self._get_solution_measures(self.problem, self.current_solution, self.population)
        self.step_count = 0
        self.initial_measures = self.current_measures.copy()
        self.best_solution = self.current_solution.clone()
        self.best_measures = self._get_solution_measures(self.problem, self.best_solution, self.population)

        # Calculate and cache population fitnesses and num_vehicles
        self.population_fitnesses = fitnesses
        self.population_num_vehicles = num_vehicles

        # Reset early stopping tracking
        self.no_improvement_count = 0
        self.last_best_measures = self.best_measures.copy()

        # Reset operator metrics
        self.operator_metrics = [{} for _ in self.operators]

        # Clear distance measure cache
        self.distance_measure.clear_cache()

        # Initialize diversity cache
        self.cached_population_diversity = self._calculate_population_diversity(self.population)

        observation = self._get_state()

        info = {
            'fitness': self.current_measures['fitness'],
            'step': self.step_count,
            'population_mean_fitness': np.mean(self.population_fitnesses),
            'population_best_fitness': np.min(self.population_fitnesses),
            'best_measures': self.best_measures
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

        # Apply selected operator (includes no-op if provided in operators list)
        operator = self.operators[action]
        operator_name = operator.name if hasattr(operator, 'name') else f"Operator{action}"

        # Handle NoOp operator - skip mutation but still calculate reward
        if operator_name == "NoOp":
            new_solution = self.current_solution.clone()
        else:
            new_solution = operator.apply(self.problem, self.current_solution)

        old_measures = self.current_measures
        new_measures = self._get_solution_measures(self.problem, new_solution, self.population)

        # Create hypothetical new population (replace solution at stored index)
        old_population = self.population
        new_population = list(self.population)  
        new_population[self.solution_index] = new_solution  

        # Update fitnesses and vehicle counts - only recalculate for the changed solution
        new_population_fitnesses = self.population_fitnesses.copy()
        new_population_fitnesses[self.solution_index] = new_measures["fitness"]

        new_population_num_vehicles = self.population_num_vehicles.copy()
        new_population_num_vehicles[self.solution_index] = new_solution.num_vehicles_used


        # Calculate reward metrics
        reward, new_diversity = self._calculate_reward(
            old_population=old_population,
            new_population=new_population,
            old_fitnesses=self.population_fitnesses,
            new_fitnesses=new_population_fitnesses,
            old_num_vehicles=self.population_num_vehicles,
            new_num_vehicles=new_population_num_vehicles,
            old_measures=old_measures,
            new_measures=new_measures
        )

        # Update best solution
        if self._compare_solutions_by_measures(new_measures, self.best_measures):
            self.best_solution = new_solution.clone()
            self.best_measures = new_measures.copy()

        # Track improvement for early stopping
        if self._compare_solutions_by_measures(self.best_measures, self.last_best_measures):
            self.no_improvement_count = 0
            self.last_best_measures = self.best_measures.copy()
        else:
            self.no_improvement_count += 1

        # Accept solution based on strategy
        accepted = self._accept_solution(old_measures, new_measures, reward)

        improvement = self._compare_solutions_by_measures(new_measures, old_measures)

        # Update state if accepted
        if accepted:
            self.current_solution = new_solution
            self.current_measures = new_measures.copy()
            # Update the stored population with the new solution (at the stored index)
            # Clone new_population list to avoid aliasing issues
            self.population = list(new_population)
            self.population_fitnesses = new_population_fitnesses
            self.population_num_vehicles = new_population_num_vehicles
            # Update diversity cache
            self.cached_population_diversity = new_diversity

        # Calculate fitness improvement for metrics

        # Update operator metrics
        self.operator_metrics[action]['applications'] = self.operator_metrics[action].get('applications', 0) + 1
        self.operator_metrics[action]['improvements'] = self.operator_metrics[action].get('improvements', 0) + (1 if improvement else 0)
        if accepted:
            self.operator_metrics[action]['acceptances'] = self.operator_metrics[action].get('acceptances', 0) + 1

        # Increment step counter
        self.step_count += 1

        # Check termination
        terminated = False

        # Early termination if no improvement for too long
        if self.max_no_improvement is not None and self.no_improvement_count >= self.max_no_improvement:
            terminated = True

        truncated = self.step_count >= self.max_steps

        # Observation and info
        observation = self._get_state()

        info = {
            'fitness': self.current_measures['fitness'],
            'new_fitness': new_measures["fitness"],
            'accepted': accepted,
            'step': self.step_count,
            'best_measures': self.best_measures,
            'operator': operator_name,
            'improvement': improvement,
            'population_mean_fitness': np.mean(self.population_fitnesses),
            'population_best_fitness': np.min(self.population_fitnesses)
        }

        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        old_population: list[PDPTWSolution],
        new_population: list[PDPTWSolution],
        old_fitnesses: list[float],
        new_fitnesses: list[float],
        old_num_vehicles: list[int],
        new_num_vehicles: list[int],
        old_measures: dict,
        new_measures: dict
    ) -> tuple[float, float]:
        """Calculate reward based on comprehensive metrics.

        This method calculates all available metrics and passes them to _compose_reward()
        for flexible reward composition.

        Args:
            old_population: Population before mutation
            new_population: Population with new solution
            old_fitnesses: Fitnesses of old population
            new_fitnesses: Fitnesses of new population
            old_num_vehicles: Number of vehicles used in old population
            new_num_vehicles: Number of vehicles used in new population
            old_measures: Measures dict for solution before mutation
            new_measures: Measures dict for solution after mutation

        Returns:
            Tuple of (reward, new_population_diversity) for cache update
        """
        metrics = self._calculate_reward_metrics(
            old_population=old_population,
            new_population=new_population,
            old_fitnesses=old_fitnesses,
            new_fitnesses=new_fitnesses,
            old_num_vehicles=old_num_vehicles,
            new_num_vehicles=new_num_vehicles,
            old_measures=old_measures,
            new_measures=new_measures
        )

        reward = self._compose_reward(metrics)
        new_diversity = metrics['population_diversity_new']

        return reward, new_diversity

    def _calculate_reward_metrics(
        self,
        old_population: list[PDPTWSolution],
        new_population: list[PDPTWSolution],
        old_fitnesses: list[float],
        new_fitnesses: list[float],
        old_num_vehicles: list[int],
        new_num_vehicles: list[int],
        old_measures: dict,
        new_measures: dict
    ) -> dict:
        """Calculate all reward metrics for flexible composition.

        Returns a dictionary containing all possible metrics that can be used
        to compose the final reward. This allows easy customization by overriding
        _compose_reward().

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Store full measures for easy access
        metrics['old_measures'] = old_measures
        metrics['new_measures'] = new_measures

        # Basic fitness metrics
        old_fitness = old_measures['fitness']
        new_fitness = new_measures['fitness']
        metrics['fitness_improvement'] = old_fitness - new_fitness
        metrics['fitness_improvement_pct'] = (old_fitness - new_fitness) / old_fitness if old_fitness > 0 else 0.0

        # Population fitness statistics
        old_mean_fitness = np.mean(old_fitnesses)
        new_mean_fitness = np.mean(new_fitnesses)
        old_best_fitness = np.min(old_fitnesses)
        new_best_fitness = np.min(new_fitnesses)
        old_worst_fitness = np.max(old_fitnesses)
        new_worst_fitness = np.max(new_fitnesses)

        metrics['population_mean_fitness_old'] = old_mean_fitness
        metrics['population_mean_fitness_new'] = new_mean_fitness
        metrics['population_best_fitness_old'] = old_best_fitness
        metrics['population_best_fitness_new'] = new_best_fitness
        metrics['population_worst_fitness_old'] = old_worst_fitness
        metrics['population_worst_fitness_new'] = new_worst_fitness

        # Comparison to population statistics
        metrics['better_than_mean'] = new_fitness < new_mean_fitness
        metrics['fitness_vs_mean'] = new_fitness - new_mean_fitness
        metrics['fitness_vs_best'] = new_fitness - new_best_fitness
        metrics['fitness_vs_worst'] = new_fitness - new_worst_fitness

        # Population fitness changes
        metrics['population_fitness_change_mean'] = old_mean_fitness - new_mean_fitness
        metrics['population_fitness_change_best'] = old_best_fitness - new_best_fitness
        metrics['population_fitness_change_worst'] = old_worst_fitness - new_worst_fitness

        # Diversity metrics - distance to all population members (excluding self)
        # Get solutions from populations at stored index
        old_solution = old_population[self.solution_index]
        new_solution = new_population[self.solution_index]

        diversity_old = 0.0
        diversity_new = 0.0
        min_distance_old = float('inf')
        min_distance_new = float('inf')
        is_duplicate = False

        for i, sol in enumerate(old_population):
            # Skip if comparing with self (using stored index)
            if i == self.solution_index:
                continue
            dist_old = self.distance_measure.edge_distance(old_solution, sol)
            diversity_old += dist_old
            min_distance_old = min(min_distance_old, dist_old)

        for i, sol in enumerate(new_population):
            # Skip if comparing with self (using stored index)
            if i == self.solution_index:
                continue
            dist_new = self.distance_measure.edge_distance(new_solution, sol)
            diversity_new += dist_new
            min_distance_new = min(min_distance_new, dist_new)
            # Check for duplicate
            if dist_new == 0.0:
                is_duplicate = True

        # Average diversity
        pop_size_minus_one = max(len(old_population) - 1, 1)
        metrics['diversity_to_population_old'] = diversity_old / pop_size_minus_one
        metrics['diversity_to_population_new'] = diversity_new / pop_size_minus_one
        metrics['diversity_change'] = (diversity_new - diversity_old) / pop_size_minus_one

        # Nearest neighbor distance
        metrics['min_distance_old'] = min_distance_old if min_distance_old != float('inf') else 0.0
        metrics['min_distance_new'] = min_distance_new if min_distance_new != float('inf') else 0.0
        metrics['min_distance_change'] = metrics['min_distance_new'] - metrics['min_distance_old']

        # Duplicate detection
        metrics['is_duplicate'] = is_duplicate

        # Feasibility metrics (from measures)
        old_feasible = old_measures['is_feasible']
        new_feasible = new_measures['is_feasible']
        metrics['feasibility_change'] = int(new_feasible) - int(old_feasible)

        # Vehicle count metrics (from measures)
        old_vehicles = old_measures['num_vehicles']
        new_vehicles = new_measures['num_vehicles']
        metrics['num_vehicles_change'] = old_vehicles - new_vehicles

        # Population diversity (average pairwise distance)
        if self.cached_population_diversity is not None:
            old_pop_diversity, new_pop_diversity = self._calculate_diversity_incremental(
                old_population, new_population, self.solution_index, self.cached_population_diversity
            )
        else:
            # Fallback to full calculation if cache is not available
            old_pop_diversity = self._calculate_population_diversity(old_population)
            new_pop_diversity = self._calculate_population_diversity(new_population)

        metrics['population_diversity_old'] = old_pop_diversity
        metrics['population_diversity_new'] = new_pop_diversity
        metrics['population_diversity_change'] = new_pop_diversity - old_pop_diversity

        return metrics

    def _calculate_population_diversity(self, population: list[PDPTWSolution]) -> float:
        """Calculate average pairwise diversity in the population.

        Args:
            population: List of solutions

        Returns:
            Average pairwise edge distance
        """
        if len(population) <= 1:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_distance += self.distance_measure.edge_distance(population[i], population[j])
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _calculate_diversity_incremental(
        self,
        old_population: list[PDPTWSolution],
        new_population: list[PDPTWSolution],
        changed_index: int,
        old_diversity: float
    ) -> tuple[float, float]:
        """Calculate population diversity incrementally when one solution changes.

        Args:
            old_population: Population before change
            new_population: Population after change
            changed_index: Index of the solution that changed
            old_diversity: Cached diversity value before change

        Returns:
            Tuple of (old_population_diversity, new_population_diversity)
        """
        n = len(old_population)
        if n <= 1:
            return 0.0, 0.0

        total_pairs = n * (n - 1) / 2

        # Calculate contribution of old solution (sum of distances to all others)
        old_solution = old_population[changed_index]
        old_contribution = 0.0
        for i in range(n):
            if i != changed_index:
                old_contribution += self.distance_measure.edge_distance(old_solution, old_population[i])

        # Calculate contribution of new solution (sum of distances to all others)
        new_solution = new_population[changed_index]
        new_contribution = 0.0
        for i in range(n):
            if i != changed_index:
                new_contribution += self.distance_measure.edge_distance(new_solution, new_population[i])

        # Incremental update: remove old contribution, add new contribution
        old_total_distance = old_diversity * total_pairs
        new_total_distance = old_total_distance - old_contribution + new_contribution
        new_diversity = new_total_distance / total_pairs

        return old_diversity, new_diversity

    def _compose_reward(self, metrics: dict) -> float:
        """Compose final reward from calculated metrics.

        Dispatches to different reward strategies based on self.reward_strategy.

        Args:
            metrics: Dictionary of all calculated metrics

        Returns:
            Scalar reward value
        """
        if self.reward_strategy == "binary":
            return self._reward_binary(metrics)
        else:
            # Default to binary
            return self._reward_binary(metrics)

    def _reward_binary(self, metrics: dict) -> float:
        """Binary reward strategy: +1/-1 for improvements/degradations.

        Simple and stable reward that treats all improvements equally.

        Args:
            metrics: Dictionary of all calculated metrics

        Returns:
            Reward value in range [-3, +3]
        """
        reward = 0.0

        # Fitness component
        if metrics['fitness_improvement'] > 0:
            reward += 1.0
        elif metrics['fitness_improvement'] < 0:
            reward -= 1.0

        # Diversity component
        if metrics['diversity_change'] > 0:
            reward += 1.0
        elif metrics['diversity_change'] < 0:
            reward -= 1.0

        # Feasibility component
        if metrics['new_measures']['is_feasible'] and not metrics['old_measures']['is_feasible']:
            reward += 1.0
        elif not metrics['new_measures']['is_feasible'] and metrics['old_measures']['is_feasible']:
            reward -= 1.0

        # Duplicate penalty
        if metrics['is_duplicate']:
            reward -= 1.0

        return reward

    def _get_solution_measures(self, problem: PDPTWProblem, solution: PDPTWSolution, population: Optional[list[PDPTWSolution]] = None) -> dict:
        """Extract comprehensive measures from a solution.

        Args:
            problem: PDPTW problem instance
            solution: Solution to extract measures from
            population: Optional population for diversity calculation

        Returns:
            Dictionary of measures
        """
        measures = {
            'fitness': fitness(problem, solution),
            'num_vehicles': solution.num_vehicles_used,
            'total_distance': solution.total_distance,
            'is_feasible': solution.is_feasible,
            'num_customers_served': solution.num_customers_served,
            'avg_distance_to_pop': 0.0,
            'min_distance_to_pop': 0.0,
            'diversity_score': 0.0
        }

        # Calculate diversity metrics if population is provided
        if population is not None and len(population) > 0:
            total_distance = 0.0
            min_dist = float('inf')
            count = 0

            for i, pop_sol in enumerate(population):
                # Skip if comparing with self (using stored index)
                if self.solution_index is not None and i == self.solution_index:
                    continue

                dist = self.distance_measure.edge_distance(solution, pop_sol)
                total_distance += dist
                min_dist = min(min_dist, dist)
                count += 1

            if count > 0:
                avg_distance = total_distance / count
                min_distance = min_dist if min_dist != float('inf') else 0.0

                measures['avg_distance_to_pop'] = avg_distance
                measures['min_distance_to_pop'] = min_distance
                # Combined diversity score: emphasize avg distance (exploration) while avoiding clustering
                measures['diversity_score'] = 0.7 * avg_distance + 0.3 * min_distance

        return measures

    def _compare_solutions_by_measures(self, measures1: dict, measures2: dict) -> bool:
        """Compare two solutions using their measure dictionaries.

        Args:
            measures1: Measures of first solution (typically the new/candidate solution)
            measures2: Measures of second solution (typically the current/reference solution)

        Returns:
            True if solution1 is better than solution2, False otherwise
        """
        # Primary criterion: fitness improvement (lower is better)
        if measures1['fitness'] < measures2['fitness']:
            return True

        # Secondary criterion: feasible solutions with better diversity
        # This helps maintain population diversity even when fitness doesn't improve
        if measures1['is_feasible'] and measures1.get('diversity_score', 0.0) > measures2.get('diversity_score', 0.0):
            return True

        return False

    def _accept_solution(self, old_measures: dict, new_measures: dict, reward: float) -> bool:
        """Determine whether to accept the new solution.

        Args:
            old_measures: Measures dict of current solution
            new_measures: Measures dict of new solution
            reward: Reward value for the mutation

        Returns:
            True if solution should be accepted, False otherwise
        """
        if self.acceptance_strategy == "reward_based":
            # Accept if reward is non-negative
            return reward >= 0

        elif self.acceptance_strategy == "greedy":
            # Accept only if new solution is better
            return self._compare_solutions_by_measures(new_measures, old_measures)

        elif self.acceptance_strategy == "always":
            # Always accept
            return True

        else:
            # Default to greedy
            return self._compare_solutions_by_measures(new_measures, old_measures)
    
    def _get_solution_features(self, problem: PDPTWProblem, solution: PDPTWSolution) -> np.ndarray:
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

        # Time window tightness
        time_window_spans = problem.time_windows[:, 1] - problem.time_windows[:, 0]
        max_time_horizon = np.max(problem.time_windows[:, 1])
        avg_tw_tightness = np.mean(time_window_spans) / max_time_horizon if max_time_horizon > 0 else 0

        # Solution features
        num_routes = solution.num_vehicles_used
        num_customers_served = solution.num_customers_served
        total_distance = solution.total_distance
        feasible = solution.is_feasible

        # Route statistics
        route_lengths = [len(r) - 2 for r in solution.routes if len(r) > 2]
        avg_route_length = np.mean(route_lengths) if route_lengths else 0
        max_route_length = np.max(route_lengths) if route_lengths else 0
        min_route_length = np.min(route_lengths) if route_lengths else 0
        std_route_length = np.std(route_lengths) if route_lengths else 0

        # Route distance statistics
        route_distances = [solution.route_lengths[i] for i in range(len(solution.routes)) if len(solution.routes[i]) > 2]
        avg_route_distance = np.mean(route_distances) if route_distances else 0
        max_route_distance = np.max(route_distances) if route_distances else 0
        std_route_distance = np.std(route_distances) if route_distances else 0

        # Diversity metrics (relative to population if available)
        avg_distance_to_pop = 0.0
        min_distance_to_pop = 0.0
        distance_to_best = 0.0
        is_duplicate = 0.0

        if self.population is not None and len(self.population) > 0:
            total_distance = 0.0
            min_dist = float('inf')
            count = 0

            # Find best solution in population
            best_idx = np.argmin(self.population_fitnesses) if self.population_fitnesses else 0
            best_solution = self.population[best_idx]

            for i, pop_sol in enumerate(self.population):
                # Skip if comparing with self (using stored index)
                if self.solution_index is not None and i == self.solution_index:
                    continue

                dist = self.distance_measure.edge_distance(solution, pop_sol)
                total_distance += dist
                min_dist = min(min_dist, dist)
                count += 1

                # Check for duplicate
                if dist == 0.0:
                    is_duplicate = 1.0

            if count > 0:
                avg_distance_to_pop = total_distance / count
                min_distance_to_pop = min_dist if min_dist != float('inf') else 0.0

            # Distance to best solution
            distance_to_best = self.distance_measure.edge_distance(solution, best_solution)

        # Normalized features
        features = np.array([
            # Problem features
            num_requests / 1000,
            vehicle_capacity / 1000,
            num_vehicles / 250,

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

            # Diversity features (normalized by typical edge distance range)
            avg_distance_to_pop,  
            min_distance_to_pop,  
            distance_to_best,     
            is_duplicate,         

        ], dtype=np.float32)

        return features

    def _get_population_features(
        self,
        population: list[PDPTWSolution],
        current_fitnesses: list[float],
        current_num_vehicles: list[int]) -> np.ndarray:
        """Extract features from the population for RL state representation.

        Features include:
        - Best/mean/worst/std fitness
        - Best/mean/worst/std num_vehicles
        - Population diversity (avg pairwise distance)
        - Proportion of feasible solutions
        - Current solution's position relative to population

        Args:
            population: List of PDPTW solutions
            current_fitnesses: Pre-calculated fitnesses for efficiency
            current_num_vehicles: Pre-calculated vehicle counts for efficiency

        Returns:
            np.ndarray: Feature vector
        """
        if len(population) == 0:
            return np.zeros(12, dtype=np.float32)

        # Fitness statistics
        best_fitness = np.min(current_fitnesses)
        mean_fitness = np.mean(current_fitnesses)
        worst_fitness = np.max(current_fitnesses)
        std_fitness = np.std(current_fitnesses)

        # Vehicle statistics
        best_vehicles = np.min(current_num_vehicles)
        mean_vehicles = np.mean(current_num_vehicles)
        worst_vehicles = np.max(current_num_vehicles)
        std_vehicles = np.std(current_num_vehicles)

        # Population diversity
        pop_diversity = self._calculate_population_diversity(population)

        # Proportion feasible
        feasible_count = sum(1 for sol in population if sol.is_feasible)
        proportion_feasible = feasible_count / len(population)

        # Current solution comparison (if available)
        if self.current_solution is not None and self.current_measures is not None:
            # Rank of current solution (0 = best, 1 = worst)
            rank = sum(1 for f in current_fitnesses if f < self.current_measures['fitness'])
            normalized_rank = rank / len(population) if len(population) > 1 else 0.5
        else:
            normalized_rank = 0.5

        # Normalize features
        baseline = self.problem.distance_baseline if self.problem else 1.0
        max_vehicles = self.problem.num_vehicles if self.problem else 1.0

        features = np.array([
            best_fitness / worst_fitness,
            mean_fitness / worst_fitness,
            worst_fitness / worst_fitness,
            std_fitness / worst_fitness,
            best_vehicles / max_vehicles,
            mean_vehicles / max_vehicles,
            worst_vehicles / max_vehicles,
            std_vehicles / max_vehicles,
            pop_diversity,  # Already normalized (0-1)
            proportion_feasible,
            normalized_rank,
            len(population) / 100.0  # Population size
        ], dtype=np.float32)

        return features
    
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

            success_rate = improvements / applications if applications > 0 else 0.0
            acceptance_rate = acceptances / applications if applications > 0 else 0.0

            features.append([
                applications / self.step_count if self.step_count > 0 else 0.0,
                improvements / self.step_count if self.step_count > 0 else 0.0,
                acceptances / self.step_count if self.step_count > 0 else 0.0,
                # success_rate,
                # acceptance_rate,
            ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation.

        Returns:
            np.ndarray: Combined feature vector of solution, population, and operators
        """
        if self.problem is None or self.current_solution is None:
            raise RuntimeError("Environment must be reset before getting state")

        solution_features = self._get_solution_features(self.problem, self.current_solution)
        population_features = self._get_population_features(
            self.population,
            self.population_fitnesses,
            self.population_num_vehicles
        )
        operator_features = self._get_operator_features().flatten()

        # Combine features
        state = np.concatenate([solution_features, population_features, operator_features])

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
        
    def get_best_solution(self) -> Tuple[PDPTWSolution, dict]:
        """Get the best solution found during the episode.

        Returns:
            Tuple of (best_solution, best_measures)
        """
        return self.best_solution, self.best_measures