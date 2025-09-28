from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.insertion_heuristic import greedy_insertion
import random

class ReinsertOperator(BaseOperator):
    def __init__(self, 
                 insertion_heuristik: str = "greedy", 
                 max_attempts: int = 1, 
                 max_attempt_interval: tuple = None, 
                 clustered: bool = False, 
                 allow_new_vehicles: bool = True, 
                 allow_same_vehicle: bool = True,
                 force_same_vehicle: bool = False
                 ):
        super().__init__()
        self.max_attempts = max_attempts
        self.insertion_heuristik = insertion_heuristik
        self.clustered = clustered
        if max_attempt_interval is not None:
            self.max_attempts = random.randint(max_attempt_interval[0], max_attempt_interval[1])
        self.allow_new_vehicles = allow_new_vehicles
        self.allow_same_vehicle = allow_same_vehicle
        self.force_same_vehicle = force_same_vehicle

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        removed_requests = []
        
        for _ in range(self.max_attempts):
            # Select a random request (pickup, delivery)
            requests = problem.pickups_deliveries
            random_request = requests[random.randint(0, len(requests) - 1)]
            route = new_solution.node_to_route.get(random_request[0], None)
            if route is not None:
                # Remove both pickup and delivery from the route
                new_solution.routes[route] = [node for node in new_solution.routes[route] if node not in random_request]
                removed_requests.append(random_request)
                new_solution._clear_cache()
                # Reinsert using greedy insertion
                if self.clustered:
                    new_solution = greedy_insertion(problem, new_solution, [random_request])
        
        if not self.clustered:
            new_solution = greedy_insertion(problem, new_solution, removed_requests)
            
        return new_solution