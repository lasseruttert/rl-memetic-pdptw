from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseCrossover(ABC):
    """Abstract base class for crossover operators in PDPTW."""
    def __init__(self):
        pass
    
    @abstractmethod
    def crossover(self, parent1: PDPTWSolution, parent2: PDPTWSolution) -> PDPTWSolution:
        """Perform crossover between two parent solutions to produce an offspring solution."""
        pass