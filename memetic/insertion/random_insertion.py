from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.insertion.simple_insertion_heuristic import _is_feasible_insertion, _cost_increase
import random

class RandomInsertion:
    """Random insertion heuristic - inserts at random feasible positions.
    
    Pure perturbation strategy without optimization. Evaluates all feasible
    positions but selects randomly among them (optionally weighted by feasibility).
    
    Args:
        allow_new_vehicles: Allow creating new routes
        not_allowed_vehicle_idxs: Routes to exclude from insertion
        force_vehicle_idx: Force insertion into specific route
        weighted: If True, weight by inverse cost (slightly biased toward better positions)
    """
    
    def __init__(self, 
                 allow_new_vehicles: bool = True, 
                 not_allowed_vehicle_idxs: list = None, 
                 force_vehicle_idx: int = None,
                 weighted: bool = False):
        self.allow_new_vehicles = allow_new_vehicles
        self.not_allowed_vehicle_idxs = not_allowed_vehicle_idxs
        self.force_vehicle_idx = force_vehicle_idx
        self.weighted = weighted
    
    def insert(self, problem: PDPTWProblem, solution: PDPTWSolution, 
               unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
        if unserved_requests is None:
            unserved_requests = solution.get_unserved_requests(problem)
        
        # Shuffle to randomize insertion order
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
                
                # Empty route case
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
                
                # Regular insertion - collect all feasible positions
                for pickup_pos in range(1, len(route)):
                    for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                        new_route = (route[:pickup_pos] + [pickup] + 
                                   route[pickup_pos:delivery_pos] + [delivery] + 
                                   route[delivery_pos:])
                        if _is_feasible_insertion(problem, new_route):
                            increase = _cost_increase(problem, route, new_route)
                            feasible_insertions.append((route_idx, new_route, increase))
            
            if not feasible_insertions:
                # No feasible insertion found, skip this request
                continue
            
            # Select random insertion from feasible options
            if self.weighted and len(feasible_insertions) > 1:
                # Weight by inverse cost (slightly prefer better positions, but still random)
                costs = [insertion[2] for insertion in feasible_insertions]
                max_cost = max(costs)
                # Invert costs: high cost → low weight
                weights = [max_cost - cost + 1 for cost in costs]
                selected = random.choices(feasible_insertions, weights=weights)[0]
            else:
                # Pure uniform random
                selected = random.choice(feasible_insertions)
            
            route_idx, new_route, _ = selected
            solution.routes[route_idx] = new_route
            solution._clear_cache()
        
        return solution