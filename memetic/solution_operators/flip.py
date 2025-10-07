from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class FlipOperator(BaseOperator):
    def __init__(self, max_attempts: None = None, single_route: bool = False):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        
        if self.single_route:
            routes = [random.choice(new_solution.routes)]
        else:
            routes = new_solution.routes
        
        attempts = 0
        for route in routes:
            if self.max_attempts is not None and attempts >= self.max_attempts:
                break
            if len(route) > 2: 
                i, j = random.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]
        new_solution._clear_cache()
        return new_solution