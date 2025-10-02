from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseInsertion(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def insert(self, problem: PDPTWProblem, solution: PDPTWSolution, unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
        pass