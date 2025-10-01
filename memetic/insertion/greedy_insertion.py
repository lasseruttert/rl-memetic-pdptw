from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.insertion.simple_insertion_heuristic import _cost_increase
# from memetic.insertion.simple_insertion_heuristic import _is_feasible_insertion
from memetic.insertion.feasibility_check import is_feasible_insertion_fast as _is_feasible_insertion

class GreedyInsertion: 
    """Greedy insertion heuristic for PDPTW. Inserts unserved requests into the solution
    at the position that results in the least increase in total cost.
    """
    def __init__(self, allow_new_vehicles = True, not_allowed_vehicle_idxs = None, force_vehicle_idx = None):
        """
        Args:
            allow_new_vehicles (bool, optional): If True, new vehicles can be used for insertion. Defaults to True.
            not_allowed_vehicle_idxs (_type_, optional): List of vehicle indices that are not allowed for insertion. Defaults to None.
            force_vehicle_idx (_type_, optional): If set, only this vehicle index is allowed for insertion. Defaults to None.
        """
        self.allow_new_vehicles = allow_new_vehicles
        self.not_allowed_vehicle_idxs = not_allowed_vehicle_idxs
        self.force_vehicle_idx = force_vehicle_idx
    
    def insert(self, problem: PDPTWProblem, solution: PDPTWSolution, unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
        """Insert unserved requests into the solution using a greedy heuristic.

        Args:
            problem (PDPTWProblem): a PDPTW problem instance
            solution (PDPTWSolution): a PDPTW solution instance in which requests should be inserted
            unserved_requests (list[tuple[int, int]], optional): List of unserved requests to insert. If None, will be determined from the solution. Defaults to None.

        Returns:
            PDPTWSolution: the solution with inserted requests, if possible
        """
        if unserved_requests is None:
            unserved_requests = solution.get_unserved_requests(problem)

        while unserved_requests:
            best_insertion = None
            best_increase = float('inf')
            for request in unserved_requests:
                pickup, delivery = request
                looked_at_empty = False
                for route_idx, route in enumerate(solution.routes):
                    if self.not_allowed_vehicle_idxs and route_idx in self.not_allowed_vehicle_idxs:
                        continue
                    if self.force_vehicle_idx is not None and route_idx != self.force_vehicle_idx:
                        continue
                    if not route or len(route) <= 2 and route[0] == 0 and route[-1] == 0 and not looked_at_empty:  # Empty route, just add pickup and delivery
                        if not self.allow_new_vehicles:
                            continue
                        new_route = [0, pickup, delivery, 0]
                        looked_at_empty = True
                        increase = (problem.distance_matrix[0, pickup] + 
                                    problem.distance_matrix[pickup, delivery] + 
                                    problem.distance_matrix[delivery, 0])
                        increase += problem.distance_baseline  # Small penalty for using a new vehicle
                        if increase < best_increase:
                            best_insertion = route_idx, new_route, pickup, delivery
                            best_increase = increase
                            continue  # No need to look at other positions in this empty route
                        
                    for pickup_pos in range(1, len(route)):  # Position to insert pickup #TODO rewrite this in cpp and pybind
                        for delivery_pos in range(pickup_pos + 1, len(route) + 1):  # Position to insert delivery
                            new_route = route[:pickup_pos] + [pickup] + route[pickup_pos:delivery_pos] + [delivery] + route[delivery_pos:]
                            if _is_feasible_insertion(problem, new_route):
                                increase = _cost_increase(problem, route, new_route)
                                if increase < best_increase:
                                    best_insertion = route_idx, new_route, pickup, delivery
                                    best_increase = increase

            if best_insertion:
                route_idx, new_route, pickup, delivery = best_insertion
                solution.routes[route_idx] = new_route
                unserved_requests.remove((pickup, delivery))
                solution._clear_cache()
            else:
                # print("No feasible insertion found for remaining requests")
                break  # No feasible insertion found, exit loop
        return solution