from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseRepair(ABC):
    @abstractmethod
    def repair(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        pass