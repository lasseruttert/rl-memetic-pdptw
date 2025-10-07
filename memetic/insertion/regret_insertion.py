from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.insertion.simple_insertion_heuristic import _cost_increase, _is_feasible_insertion
class Regret2Insertion:
    """Regret-2 insertion heuristic for PDPTW. Inserts requests based on the regret of not inserting them now.
    """
    def __init__(self, allow_new_vehicles = True, not_allowed_vehicle_idxs = None, force_vehicle_idx = None):
        """
        Args:
            allow_new_vehicles (bool): Whether to allow inserting into new vehicles (empty routes).
            not_allowed_vehicle_idxs (list[int] | None): List of vehicle indices that are not allowed for insertion.
            force_vehicle_idx (int | None): If set, only this vehicle index is allowed for insertion.
        """
        self.allow_new_vehicles = allow_new_vehicles
        self.not_allowed_vehicle_idxs = not_allowed_vehicle_idxs
        self.force_vehicle_idx = force_vehicle_idx
    
    def insert(self, problem: PDPTWProblem, solution: PDPTWSolution, unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
        """Insert unserved requests into the solution using the regret-2 heuristic.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance
            unserved_requests (list[tuple[int, int]], optional): List of unserved requests (pickup, delivery).

        Returns:
            PDPTWSolution: the improved solution
        """
        if unserved_requests is None:
            unserved_requests = solution.get_unserved_requests(problem)

        while unserved_requests:
            request_regrets = []
            
            for request in unserved_requests:
                pickup, delivery = request
                insertions = []  
                looked_at_empty = False
                
                for route_idx, route in enumerate(solution.routes):
                    if self.not_allowed_vehicle_idxs and route_idx in self.not_allowed_vehicle_idxs:
                        continue
                    if self.force_vehicle_idx is not None and route_idx != self.force_vehicle_idx:
                        continue
                    
                    if not route or len(route) <= 2 and route[0] == 0 and route[-1] == 0 and not looked_at_empty:
                        if not self.allow_new_vehicles:
                            continue
                        new_route = [0, pickup, delivery, 0]
                        if _is_feasible_insertion(problem, new_route):
                            looked_at_empty = True
                            increase = (problem.distance_matrix[0, pickup] + 
                                        problem.distance_matrix[pickup, delivery] + 
                                        problem.distance_matrix[delivery, 0])
                            increase += problem.distance_baseline
                            insertions.append((increase, route_idx, new_route))
                    
                    for pickup_pos in range(1, len(route)):
                        for delivery_pos in range(pickup_pos + 1, len(route) + 1):
                            new_route = route[:pickup_pos] + [pickup] + route[pickup_pos:delivery_pos] + [delivery] + route[delivery_pos:]
                            if _is_feasible_insertion(problem, new_route):
                                increase = _cost_increase(problem, route, new_route)
                                insertions.append((increase, route_idx, new_route))
                
                if len(insertions) == 0:
                    regret = float('inf')
                    best_insertion = None
                elif len(insertions) == 1:
                    regret = float('inf')
                    best_insertion = insertions[0]
                else:
                    insertions.sort(key=lambda x: x[0])
                    best_insertion = insertions[0]
                    second_best = insertions[1]
                    regret = second_best[0] - best_insertion[0]
                
                request_regrets.append((regret, request, best_insertion))
            
            if not request_regrets:
                break
            
            request_regrets.sort(key=lambda x: x[0], reverse=True)
            
            best_regret, best_request, best_insertion = request_regrets[0]
            
            if best_insertion is None:
                break
            
            increase, route_idx, new_route = best_insertion
            pickup, delivery = best_request
            solution.routes[route_idx] = new_route
            unserved_requests.remove(best_request)
            solution._clear_cache()
        
        return solution