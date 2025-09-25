from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class FlipOperator(BaseOperator):
    def __init__(self, problem: PDPTWProblem):
        super().__init__(problem)

    def apply(self, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        for route in new_solution.routes:
            if len(route) > 2:  # Ensure there are at least two nodes to swap
                i, j = random.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]
        new_solution._clear_cache()
        return new_solution