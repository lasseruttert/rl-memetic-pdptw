from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.local_search.base_local_search import BaseLocalSearch

from memetic.fitness.fitness import fitness
import random

class AdaptiveLocalSearch(BaseLocalSearch):
    """Adaptive Local Search using operator performance statistics.
    
    Uses a simple bandit-like approach: tracks success rate of each operator
    and adaptively adjusts their selection probability based on recent performance.
    
    Args:
        operators: List of operators to choose from
        max_no_improvement: Terminate after this many iterations without improvement
        max_iterations: Maximum total iterations
        adaptation_method: 'probability_matching', 'epsilon_greedy', or 'ucb'
        window_size: Number of recent iterations to consider for adaptation
        epsilon: Exploration parameter for epsilon-greedy (default 0.1)
    """
    
    def __init__(self, 
                 operators: list = [], 
                 max_no_improvement: int = 3, 
                 max_iterations: int = 50,
                 adaptation_method: str = 'probability_matching',
                 window_size: int = 10,
                 epsilon: float = 0.1):
        super().__init__()
        self.operators = operators
        self.max_no_improvement = max_no_improvement
        self.max_iterations = max_iterations
        self.adaptation_method = adaptation_method
        self.window_size = window_size
        self.epsilon = epsilon
        
        # Initialize operator statistics
        self.operator_weights = [1.0] * len(operators)  # Equal initial weights
        self.operator_successes = [0] * len(operators)
        self.operator_attempts = [0] * len(operators)
        self.recent_improvements = [[] for _ in range(len(operators))]  # Sliding window
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> tuple[PDPTWSolution, float]:
        """Start the adaptive local search process.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            tuple[PDPTWSolution, float]: the best solution found and its fitness
        """
        no_improvement_count = 0
        iteration = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)
        
        while no_improvement_count < self.max_no_improvement and iteration < self.max_iterations:
            # Select operator based on adaptive strategy
            operator_idx = self._select_operator()
            operator = self.operators[operator_idx]
            
            # Apply operator
            new_solution = operator.apply(problem, best_solution)
            operator.applications += 1
            new_fitness = fitness(problem, new_solution)
            
            # Track result
            self.operator_attempts[operator_idx] += 1
            improved = new_fitness < best_fitness
            
            if improved:
                operator.improvements += 1
                self.operator_successes[operator_idx] += 1
                best_solution = new_solution
                best_fitness = new_fitness
                no_improvement_count = 0
                
                # Record improvement magnitude for sliding window
                improvement = best_fitness - new_fitness
                self.recent_improvements[operator_idx].append(improvement)
                if len(self.recent_improvements[operator_idx]) > self.window_size:
                    self.recent_improvements[operator_idx].pop(0)
            else:
                no_improvement_count += 1
                # Record zero improvement
                self.recent_improvements[operator_idx].append(0.0)
                if len(self.recent_improvements[operator_idx]) > self.window_size:
                    self.recent_improvements[operator_idx].pop(0)
            
            # Update operator weights
            self._update_weights()
            
            iteration += 1
        
        return best_solution, best_fitness
    
    def _select_operator(self) -> int:
        """Select operator index based on adaptation method"""
        if self.adaptation_method == 'probability_matching':
            return self._probability_matching()
        elif self.adaptation_method == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.adaptation_method == 'ucb':
            return self._ucb()
        else:
            return random.randint(0, len(self.operators) - 1)
    
    def _probability_matching(self) -> int:
        """Select operator proportional to its weight"""
        total_weight = sum(self.operator_weights)
        if total_weight == 0:
            return random.randint(0, len(self.operators) - 1)
        
        probabilities = [w / total_weight for w in self.operator_weights]
        return random.choices(range(len(self.operators)), weights=probabilities)[0]
    
    def _epsilon_greedy(self) -> int:
        """Select best operator with probability 1-epsilon, random otherwise"""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.operators) - 1)
        else:
            return self.operator_weights.index(max(self.operator_weights))
    
    def _ucb(self) -> int:
        """Upper Confidence Bound selection"""
        total_attempts = sum(self.operator_attempts)
        if total_attempts == 0:
            return random.randint(0, len(self.operators) - 1)
        
        ucb_values = []
        for i in range(len(self.operators)):
            if self.operator_attempts[i] == 0:
                ucb_values.append(float('inf'))  # Explore unvisited operators
            else:
                exploitation = self.operator_weights[i]
                exploration = (2 * (total_attempts ** 0.5)) / (self.operator_attempts[i] ** 0.5)
                ucb_values.append(exploitation + exploration)
        
        return ucb_values.index(max(ucb_values))
    
    def _update_weights(self):
        """Update operator weights based on recent performance"""
        for i in range(len(self.operators)):
            if self.operator_attempts[i] == 0:
                continue
            
            # Calculate success rate
            success_rate = self.operator_successes[i] / self.operator_attempts[i]
            
            # Calculate average recent improvement
            if self.recent_improvements[i]:
                avg_recent = sum(self.recent_improvements[i]) / len(self.recent_improvements[i])
            else:
                avg_recent = 0.0
            
            # Update weight: combine success rate with recent performance
            # Weight = α * success_rate + (1-α) * normalized_recent_improvement
            alpha = 0.5
            self.operator_weights[i] = alpha * success_rate + (1 - alpha) * avg_recent
            
            # Add small baseline to avoid zero weights
            self.operator_weights[i] += 0.01