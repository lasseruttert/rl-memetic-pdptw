from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.selection.base_selection import BaseSelection

import random

class RankSelection(BaseSelection):
    """Rank selection operator for selecting parents in a genetic algorithm.
    Selects individuals based on their rank in the population.
    """
    def __init__(self):
        super().__init__()
    
    def select(self, population: list[PDPTWSolution], fitnesses: list[float]) -> PDPTWSolution:
        """Select an individual from the population using rank selection.

        Args:
            population (list[PDPTWSolution]): List of individuals in the population.
            fitnesses (list[float]): Corresponding fitness values for the individuals.
        Returns:
            PDPTWSolution: The selected individual.
        """
        ranked_indices = sorted(range(len(population)), key=lambda idx: fitnesses[idx])
        rank_weights = [len(population) - rank for rank in range(len(population))]
        total_rank_weight = sum(rank_weights)
        selection_probs = [weight / total_rank_weight for weight in rank_weights]
        cumulative_probs = []
        cumulative_sum = 0.0
        for prob in selection_probs:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)
        
        rand_value = random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if rand_value <= cum_prob:
                return population[ranked_indices[i]]
        
        return population[ranked_indices[-1]]  # Fallback in case of rounding errors