from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, problem: PDPTWProblem, num_solutions: int) -> PDPTWSolution:
        pass
    
