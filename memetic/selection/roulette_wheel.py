from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.selection.base_selection import BaseSelection

import random

class RouletteWheelSelection(BaseSelection):
    """Roulette Wheel selection operator for selecting parents in a genetic algorithm.
    Selects individuals with a probability proportional to their fitness.
    """
    def __init__(self):
        super().__init__()
    
    def select(self, population: list[PDPTWSolution], fitnesses: list[float]) -> PDPTWSolution:
        """Select an individual from the population using roulette wheel selection.

        Args:
            population (list[PDPTWSolution]): List of individuals in the population.
            fitnesses (list[float]): Corresponding fitness values for the individuals. 
        Returns:
            PDPTWSolution: The selected individual.
        """
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)
        
        selection_probs = [f / total_fitness for f in fitnesses]
        cumulative_probs = []
        cumulative_sum = 0.0
        for prob in selection_probs:
            cumulative_sum += prob
            cumulative_probs.append(cumulative_sum)
        
        rand_value = random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if rand_value <= cum_prob:
                return population[i]
        
        return population[-1] 