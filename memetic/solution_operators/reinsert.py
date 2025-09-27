from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.insertion_heuristic import greedy_insertion
import random

class ReinsertOperator(BaseOperator):
    def __init__(self):
        super().__init__()

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()

        requests = problem.pickups_deliveries
        random_request = requests[random.randint(0, len(requests) - 1)]
        route = new_solution.node_to_route.get(random_request[0], None)
        if route is not None:
            # Remove both pickup and delivery from the route
            new_solution.routes[route] = [node for node in new_solution.routes[route] if node not in random_request]
            new_solution._clear_cache()
            # Reinsert using greedy insertion
            new_solution = greedy_insertion(problem, new_solution, [random_request])
            
        return new_solution