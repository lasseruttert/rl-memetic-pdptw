from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseCrossover(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def crossover(self, parent1: PDPTWSolution, parent2: PDPTWSolution) -> PDPTWSolution:
        pass