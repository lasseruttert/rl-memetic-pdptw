from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class RemoveRequestOperator(BaseOperator):
    def __init__(self, max_attempts: int = 1):
        super().__init__()
        self.max_attempts = max_attempts
        self.name = f"RemoveRequest-Max{self.max_attempts}"
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        served_requests = new_solution.get_served_requests(problem)
        if not served_requests:
            return new_solution  # No requests to remove

        attempts = min(self.max_attempts, len(served_requests))
        requests_to_remove = random.sample(served_requests, attempts)
        
        for request in requests_to_remove:
            route_idx = new_solution.node_to_route.get(request[0], None)
            if route_idx is not None:
                # Remove both pickup and delivery from the route
                new_solution.routes[route_idx] = [node for node in new_solution.routes[route_idx] if node not in request]
        
        new_solution._clear_cache()
        return new_solution