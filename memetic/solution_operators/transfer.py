from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class TransferOperator(BaseOperator):
    def __init__(self, max_attempts: int = 1, single_route: bool = False):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        self.name = f"Transfer-{'Single' if self.single_route else 'All'}-Max{self.max_attempts}"

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
            if len(route) >= 4:  # Ensure there is at least one request to transfer
                nodes_in_route = route[1:-1]  # Exclude depot
                requests = set(problem.get_pair(node) for node in nodes_in_route)
                request_to_transfer = random.choice(list(requests))
                pickup, delivery = request_to_transfer
                
                # Remove the request from the current route
                route.remove(pickup)
                route.remove(delivery)
                
                # Select a different route to transfer the request to
                other_routes = [r for r in new_solution.routes if r != route and len(r) > 2]
                if other_routes:
                    target_route = random.choice(other_routes)
                    
                    insert_position_pickup = random.randint(1, len(target_route) - 1)
                    insert_position_delivery = random.randint(insert_position_pickup + 1, len(target_route))
                    
                    target_route.insert(insert_position_pickup, pickup)
                    target_route.insert(insert_position_delivery, delivery)
                    
                    attempts += 1
                
        new_solution._clear_cache()
        return new_solution

