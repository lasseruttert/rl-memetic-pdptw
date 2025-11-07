from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

import random

class RequestShiftWithinOperator(BaseOperator):
    def __init__(self):
        super().__init__()
        self.name = "RequestShiftWithin"
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        route_index = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_index]
        
        if len(route) <= 2:
            return new_solution  
        
        requests_in_route = set(problem.get_pair(node) for node in route[1:-1])
        if not requests_in_route:
            return new_solution  
        
        request = random.choice(list(requests_in_route))
        pickup, delivery = request
        
        # Remove pickup and delivery from current positions
        route.remove(pickup)
        route.remove(delivery)
        
        # Determine new positions
        insert_positions = list(range(1, len(route)))  # Possible positions excluding depot
        new_pickup_pos = random.choice(insert_positions)
        route.insert(new_pickup_pos, pickup)
        
        insert_positions = [pos for pos in range(1, len(route)+1) if pos != new_pickup_pos]
        new_delivery_pos = random.choice(insert_positions)
        route.insert(new_delivery_pos, delivery)
        
        route = [node for node in route if node != 0]  # Remove depots temporarily
        route.insert(0, 0)
        route.append(0)

        new_solution.routes[route_index] = route
        return new_solution