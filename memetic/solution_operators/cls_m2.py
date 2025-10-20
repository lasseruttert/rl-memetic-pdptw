from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.greedy_insertion import GreedyInsertion
from memetic.insertion.regret_insertion import Regret2Insertion

import random

# this is the operator M2 from the paper "Large Neighborhood Search with adaptive guided ejection search for the pickup and delivery problem with time windows"
class CLSM2Operator(BaseOperator):
    """Un-assign an assigned PD pair and try and insert it into a different route or create a new route for the PD pair"""
    def __init__(self, insertion_heuristic: str = 'greedy'):
        super().__init__()
        if insertion_heuristic == 'greedy':
            self.insertion_heuristic = GreedyInsertion()
        if insertion_heuristic == 'regret2':
            self.insertion_heuristic = Regret2Insertion()
        self.name = f'CLS-M2'
            
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        served_requests = solution.get_served_requests(problem)
        if not served_requests:
            return solution
        request_to_relocate = random.choice(served_requests)
        route_with_request = solution.node_to_route[request_to_relocate[0]]
        new_solution = solution.clone()
        # Remove request from current route
        new_solution.routes[route_with_request] = [
            node for node in new_solution.routes[route_with_request]
            if node not in request_to_relocate
        ]
        new_solution._clear_cache()
        # try and insert it somewhere else
        self.insertion_heuristic.not_allowed_vehicle_idxs = [route_with_request]
        new_solution = self.insertion_heuristic.insert(problem, new_solution, [request_to_relocate])
        self.insertion_heuristic.not_allowed_vehicle_idxs = []
        return new_solution