from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

import random

class TwoOptOperator(BaseOperator):
    def __init__(self, max_attempts: int = 10, single_route: bool = False):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return # TODO Implement 2-opt operator