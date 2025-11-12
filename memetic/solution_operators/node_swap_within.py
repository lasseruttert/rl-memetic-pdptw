from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

import random

class NodeSwapWithinOperator(BaseOperator):
    def __init__(self, check_precedence: bool = True):
        super().__init__()
        self.check_precedence = check_precedence
        self.name = f"NodeSwapWithin{'-WithPrecedence' if self.check_precedence else ''}"

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        route_index = random.randint(0, len(new_solution.routes) - 1)
        route = new_solution.routes[route_index]
        
        if len(route) <= 2:
            return new_solution  # No swap possible
        
        idx1, idx2 = random.sample(range(1, len(route) -1), 2)
        
        if self.check_precedence:
            node1, node2 = route[idx1], route[idx2]
            
            # Check if swapping violates pickup-delivery precedence
            if problem.is_pickup(node1):
                paired_node1 = problem.get_other(node1)
                if paired_node1 in route:
                    paired_index1 = route.index(paired_node1)
                    if (idx1 < paired_index1 and idx2 > paired_index1) or (idx1 > paired_index1 and idx2 < paired_index1):
                        return new_solution  # Violation
            if problem.is_pickup(node2):
                paired_node2 = problem.get_other(node2)
                if paired_node2 in route:
                    paired_index2 = route.index(paired_node2)
                    if (idx2 < paired_index2 and idx1 > paired_index2) or (idx2 > paired_index2 and idx1 < paired_index2):
                        return new_solution  # Violation
        
        route[idx1], route[idx2] = route[idx2], route[idx1]
        return new_solution