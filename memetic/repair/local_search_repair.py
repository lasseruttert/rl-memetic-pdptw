from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.repair.base_repair import BaseRepair

import random

class LocalSearchRepair(BaseRepair):
    def __init__(self, local_search_operator, probability: float = 0.5):
        super().__init__()
        self.local_search_operator = local_search_operator
        self.probability = probability
        
    def repair(self, solution: PDPTWSolution, problem: PDPTWProblem) -> PDPTWSolution:
        if random.random() < self.probability:
            improved_solution = self.local_search_operator(solution, problem)
            return improved_solution
        return solution