from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseOperator(ABC):
    @abstractmethod
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        pass
    
    def __call__(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return self.apply(problem, solution)
    
    