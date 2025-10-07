from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.insertion.simple_insertion_heuristic import _is_feasible_insertion, _cost_increase
import random

class RandomInsertion:
    """Random insertion heuristic - inserts at random feasible positions.
    
    Pure perturbation strategy without optimization. Evaluates all feasible
    positions but selects randomly among them (optionally weighted by feasibility).
    """
    
    def __init__(self, 
                 allow_new_vehicles: bool = True, 
                 not_allowed_vehicle_idxs: list = None, 
                 force_vehicle_idx: int = None,
                 weighted: bool = False):
        """
        Args:
            allow_new_vehicles (bool, optional): If True, new vehicles can be used for insertion. Defaults to True.
            not_allowed_vehicle_idxs (list, optional): List of vehicle indices that are not allowed for insertion. Defaults to None.
            force_vehicle_idx (int, optional): If set, only this vehicle index is allowed for insertion. Defaults to None.
            weighted (bool, optional): If True, feasible insertions are weighted by inverse cost increase when selecting randomly. Defaults to False (uniform random).
        """
        self.allow_new_vehicles = allow_new_vehicles
        self.not_allowed_vehicle_idxs = not_allowed_vehicle_idxs
        self.force_vehicle_idx = force_vehicle_idx
        self.weighted = weighted
    
    def insert(self, problem: PDPTWProblem, solution: PDPTWSolution, 
               unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
        """Insert unserved requests into the solution using a random heuristic.

        Args:
            problem (PDPTWProblem): a PDPTW problem instance
            solution (PDPTWSolution): a PDPTW solution instance in which requests should be inserted
            unserved_requests (list[tuple[int, int]], optional): List of unserved requests to insert. If None, will be determined from the solution. Defaults to None.

        Returns:
            PDPTWSolution: a solution with inserted requests, if possible
        """
        if unserved_requests is None:
            unserved_requests = solution.get_unserved_requests(problem)
        
        unserved_requests = list(unserved_requests)
        random.shuffle(unserved_requests)
        
        for request in unserved_requests:
            pickup, delivery = request
            feasible_insertions = []
            looked_at_empty = False
            
            for route_idx, route in enumerate(solution.routes):
                if self.not_allowed_vehicle_idxs and route_idx in self.not_allowed_vehicle_idxs:
                    continue
                if self.force_vehicle_idx is not None and route_idx != self.force_vehicle_idx:
                    continue
                
                if not route or (len(route) <= 2 and route[0] == 0 and route[-1] == 0 and not looked_at_empty):
                    if not self.allow_new_vehicles:
                        continue
                    new_route = [0, pickup, delivery, 0]
                    if _is_feasible_insertion(problem, new_route):
                        looked_at_empty = True
                        increase = (problem.distance_matrix[0, pickup] + 
                                    problem.distance_matrix[pickup, delivery] + 
                                    problem.distance_matrix[delivery, 0])
                        increase += problem.distance_baseline
                        feasible_insertions.append((route_idx, new_route, increase))
                
                for pickup_pos in range(1, len(route)):
                    for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                        new_route = (route[:pickup_pos] + [pickup] + 
                                   route[pickup_pos:delivery_pos] + [delivery] + 
                                   route[delivery_pos:])
                        if _is_feasible_insertion(problem, new_route):
                            increase = _cost_increase(problem, route, new_route)
                            feasible_insertions.append((route_idx, new_route, increase))
            
            if not feasible_insertions:
                continue
            
            if self.weighted and len(feasible_insertions) > 1:
                costs = [insertion[2] for insertion in feasible_insertions]
                max_cost = max(costs)
                weights = [max_cost - cost + 1 for cost in costs]
                selected = random.choices(feasible_insertions, weights=weights)[0]
            else:
                selected = random.choice(feasible_insertions)
            
            route_idx, new_route, _ = selected
            solution.routes[route_idx] = new_route
            solution._clear_cache()
        
        return solution