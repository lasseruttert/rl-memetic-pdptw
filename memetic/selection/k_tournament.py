from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
import random

class KTournamentSelection:
    """K-Tournament selection operator for selecting parents in a genetic algorithm.
    Selects k random individuals from the population and returns the best one.
    """
    def __init__(self, k: int = 3):
        """
        Args:
            k (int, optional): Number of individuals to select for the tournament. Defaults to 3.
        """
        self.k = k
    
    def select(self, population: list[PDPTWSolution], fitnesses: list[float]) -> PDPTWSolution:
        """Select an individual from the population using k-tournament selection.

        Args:
            population (list[PDPTWSolution]): List of individuals in the population.
            fitnesses (list[float]): Corresponding fitness values for the individuals.

        Returns:
            PDPTWSolution: The selected individual.
        """
        tournament_indices = random.sample(range(len(population)), self.k)
        best_idx = min(tournament_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx]