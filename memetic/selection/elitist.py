from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

class ElitistSelection:
    """Elitist selection operator for selecting parents in a genetic algorithm.
    Always selects the best individual from the population.
    """
    def select(self, population: list[PDPTWSolution], fitnesses: list[float]) -> PDPTWSolution:
        """Select the best individual from the population.

        Args:
            population (list[PDPTWSolution]): List of individuals in the population.
            fitnesses (list[float]): Corresponding fitness values for the individuals.
        Returns:
            PDPTWSolution: The selected individual.
        """
        best_idx = min(range(len(population)), key=lambda idx: fitnesses[idx])
        return population[best_idx]