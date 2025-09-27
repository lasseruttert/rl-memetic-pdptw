from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class SwapBetweenOperator(BaseOperator):
    def __init__(self):
        super().__init__()

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        possible_routes = [route for route in new_solution.routes if len(route) >= 4]
        if len(possible_routes) < 2:
            return new_solution  # Not enough routes to swap between
        
        route1, route2 = random.sample(possible_routes, 2)
        requests_in_route1 = set(problem.get_pair(node) for node in route1[1:-1])
        requests_in_route2 = set(problem.get_pair(node) for node in route2[1:-1])
        if not requests_in_route1 or not requests_in_route2:
            return new_solution  # No requests to swap
        request1 = random.choice(list(requests_in_route1))
        request2 = random.choice(list(requests_in_route2))
        
        pickup1, delivery1 = request1
        pickup2, delivery2 = request2
        
        idx1_pickup = route1.index(pickup1)
        idx1_delivery = route1.index(delivery1)
        idx2_pickup = route2.index(pickup2)
        idx2_delivery = route2.index(delivery2)
        
        # Swap the positions
        route1[idx1_pickup], route2[idx2_pickup] = route2[idx2_pickup], route1[idx1_pickup]
        route1[idx1_delivery], route2[idx2_delivery] = route2[idx2_delivery], route1[idx1_delivery]
        
        new_solution._clear_cache()
        return new_solution