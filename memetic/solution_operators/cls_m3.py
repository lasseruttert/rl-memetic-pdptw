from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.greedy_insertion import GreedyInsertion
import random

# this is the operator M3 from the paper "Large Neighborhood Search with adaptive guided ejection search for the pickup and delivery problem with time windows"

class CLSM3Operator(BaseOperator):
    """ Un-assign a PD pair (pd1) from a route (r1), un-assign a PD pair (pd2) from a route (r2) and then try and insert pd1 into route r2 and pd2 into route r1"""
    def __init__(self, insertion_heuristic: str = 'greedy'):
        super().__init__()
        if insertion_heuristic == 'greedy':
            self.insertion_heuristic = GreedyInsertion()
            
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        served_requests = solution.get_served_requests(problem)
        if len(served_requests) < 2:
            return solution
        request_to_relocate_1, request_to_relocate_2 = random.sample(served_requests, 2)
        route_with_request_1 = solution.node_to_route[request_to_relocate_1[0]]
        route_with_request_2 = solution.node_to_route[request_to_relocate_2[0]]
        new_solution = solution.clone()
        # Remove requests from current routes
        new_solution.routes[route_with_request_1] = [
            node for node in new_solution.routes[route_with_request_1]
            if node not in request_to_relocate_1
        ]
        new_solution.routes[route_with_request_2] = [
            node for node in new_solution.routes[route_with_request_2]
            if node not in request_to_relocate_2
        ]
        new_solution._clear_cache()
        # try and insert them into each other's routes
        self.insertion_heuristic.force_vehicle_idx = route_with_request_2
        new_solution = self.insertion_heuristic.insert(problem, new_solution, [request_to_relocate_1])
        self.insertion_heuristic.force_vehicle_idx = route_with_request_1
        new_solution = self.insertion_heuristic.insert(problem, new_solution, [request_to_relocate_2])
        self.insertion_heuristic.force_vehicle_idx = None
        return new_solution