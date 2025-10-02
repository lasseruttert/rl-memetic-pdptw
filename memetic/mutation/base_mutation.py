from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseMutation(ABC):
    """
    Abstract base class for mutation operators.
    """
    
    @abstractmethod
    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution, population = None) -> PDPTWSolution:
        """Mutate the given solution.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance
            population: the current population (optional)

        Returns:
            PDPTWSolution: the mutated solution
        """
        pass