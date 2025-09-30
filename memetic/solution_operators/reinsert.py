from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.greedy_insertion import GreedyInsertion
from memetic.insertion.regret_insertion import Regret2Insertion
from memetic.insertion.random_insertion import RandomInsertion

import random

class ReinsertOperator(BaseOperator):
    def __init__(self, 
                 insertion_heuristic: str = "greedy", 
                 max_attempts: int = 1, 
                 max_attempt_interval: tuple = None, 
                 clustered: bool = False, 
                 allow_new_vehicles: bool = True, 
                 allow_same_vehicle: bool = True,
                 force_same_vehicle: bool = False
                 ):
        super().__init__()
        self.max_attempts = max_attempts
        self.clustered = clustered
        if max_attempt_interval is not None:
            self.max_attempts = random.randint(max_attempt_interval[0], max_attempt_interval[1])
        
        if not allow_same_vehicle and force_same_vehicle:
            raise ValueError("Cannot both allow and force same vehicle.")
        self.allow_new_vehicles = allow_new_vehicles
        self.allow_same_vehicle = allow_same_vehicle
        self.force_same_vehicle = force_same_vehicle
        
        if insertion_heuristic == "greedy":
            self.insertion_heuristic = GreedyInsertion(
                allow_new_vehicles=allow_new_vehicles,
                not_allowed_vehicle_idxs=None if allow_same_vehicle else [],
                force_vehicle_idx=None if not force_same_vehicle else 0
            )
        if insertion_heuristic == 'regret2':
            self.insertion_heuristic = Regret2Insertion()
        if insertion_heuristic == 'random':
            self.insertion_heuristic = RandomInsertion()

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        removed_requests = []
        
        for _ in range(self.max_attempts):
            # Select a random request (pickup, delivery)
            requests = problem.pickups_deliveries
            random_request = random.choice(requests)
            route_idx = new_solution.node_to_route.get(random_request[0], None)
            if route_idx is not None:
                # Remove both pickup and delivery from the route
                new_solution.routes[route_idx] = [node for node in new_solution.routes[route_idx] if node not in random_request]
                removed_requests.append(random_request)
                new_solution._clear_cache()
                # Reinsert using greedy insertion
                if self.clustered:
                    if self.allow_same_vehicle == False: self.insertion_heuristic.not_allowed_vehicle_idxs = [route_idx]
                    if self.force_same_vehicle: self.insertion_heuristic.force_vehicle_idx = route_idx
                    new_solution = self.insertion_heuristic.insert(problem, new_solution, [random_request])
                    if self.allow_same_vehicle == False: self.insertion_heuristic.not_allowed_vehicle_idxs = None
                    if self.force_same_vehicle: self.insertion_heuristic.force_vehicle_idx = None
                            
        if not self.clustered:
            new_solution = self.insertion_heuristic.insert(problem, new_solution, removed_requests)
            
        return new_solution