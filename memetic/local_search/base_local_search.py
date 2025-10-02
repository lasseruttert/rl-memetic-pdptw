from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseLocalSearch(ABC):
    """Abstract base class for local search algorithms.
    """
    @abstractmethod
    def __init__(self):
        """Initialize the local search algorithm.
        """
        pass
    @abstractmethod
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> tuple[PDPTWSolution, float]:
        """Start the local search process.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            tuple[PDPTWSolution, float]: the best solution found and its fitness
        """
        pass