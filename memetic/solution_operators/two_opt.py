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
        new_solution = solution.clone()
        
        if self.single_route:
            routes = [random.choice(new_solution.routes)]
        else: 
            routes = new_solution.routes
        
        attempts = 0
        for route in routes:
            if self.max_attempts is not None and attempts >= self.max_attempts:
                break
            if len(route) >= 4:  # Ensure there are at least two customers to swap
                nodes_in_route = route[1:-1]  # Exclude depot
                idx1, idx2 = random.sample(range(len(nodes_in_route)), 2)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                
                # Perform 2-opt swap
                new_route = route[:idx1 + 1] + route[idx1 + 1:idx2 + 1][::-1] + route[idx2 + 1:]
                route[:] = new_route  # Update the original route in place
                
                attempts += 1
                
        new_solution._clear_cache()
        return new_solution