from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class SwapWithinOperator(BaseOperator):
    def __init__(self, problem: PDPTWProblem, max_attempts: None = None, single_route: bool = False):
        super().__init__(problem)
        self.max_attempts = max_attempts
        self.single_route = single_route

    def apply(self, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        
        if self.single_route:
            routes = [random.choice(new_solution.routes)]
        else: 
            routes = new_solution.routes
        
        attempts = 0
        for route in routes:
            if self.max_attempts is not None and attempts >= self.max_attempts:
                break
            if len(route) >= 6:  # Ensure there are at least two nodes to swap
                # select two requests in the route and swap each pickup and delivery
                nodes_in_route = route[1:-1]  # Exclude depot
                requests = set(self.problem.get_pair(node) for node in nodes_in_route)
                request1, request2 = random.sample(requests, 2)
                pickup1, delivery1 = request1
                pickup2, delivery2 = request2
                idx1_pickup = route.index(pickup1)
                idx1_delivery = route.index(delivery1)
                idx2_pickup = route.index(pickup2)
                idx2_delivery = route.index(delivery2)
                
                # Swap the positions
                route[idx1_pickup], route[idx2_pickup] = route[idx2_pickup], route[idx1_pickup]
                route[idx1_delivery], route[idx2_delivery] = route[idx2_delivery], route[idx1_delivery]
                
        new_solution._clear_cache()
        return new_solution