from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseOperator(ABC):
    def __init__(self):
        self.applications = 0
        self.improvements = 0
        
    @abstractmethod
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        pass
    
    def __call__(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return self.apply(problem, solution)
    