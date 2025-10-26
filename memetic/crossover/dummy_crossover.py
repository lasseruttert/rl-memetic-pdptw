from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.crossover.base_crossover import BaseCrossover

class DummyCrossover(BaseCrossover):
    """A dummy crossover that returns clones of the parents."""
    def __init__(self):
        super().__init__()

    def crossover(self, problem: PDPTWProblem, parent1: PDPTWSolution, parent2: PDPTWSolution) -> list[PDPTWSolution]:
        """Return clones of the parent solutions as offspring.

        Args:
            problem (PDPTWProblem): The PDPTW problem instance.
            parent1 (PDPTWSolution): The first parent solution.
            parent2 (PDPTWSolution): The second parent solution.

        Returns:
            list[PDPTWSolution]: A list containing clones of the parent solutions.
        """
        return [parent1.clone(), parent2.clone()]