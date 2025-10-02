from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from abc import ABC, abstractmethod

class BaseSelection(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select(self, population: list[PDPTWSolution], fitnesses: list[float]) -> PDPTWSolution:
        pass