from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.insertion.base_insertion import BaseInsertion

from memetic.insertion.simple_insertion_heuristic import _cost_increase
from memetic.insertion.insertion_core import find_best_position_for_request
class GreedyInsertion(BaseInsertion): 
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
        super().__init__()
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

                # C++ handles the heavy double-loop
                route_idx, pickup_pos, delivery_pos, increase, new_route = find_best_position_for_request(
                    problem.distance_matrix,
                    problem.time_windows,
                    problem.service_times,
                    problem.demands,
                    problem.vehicle_capacity,
                    solution.routes,
                    pickup,
                    delivery,
                    self.not_allowed_vehicle_idxs or [],
                    self.force_vehicle_idx if self.force_vehicle_idx is not None else -1,
                    problem.delivery_to_pickup,
                    problem.pickup_to_delivery  # NEU
                )
                
                
                if route_idx != -1 and increase < best_increase:
                    best_insertion = (route_idx, new_route, pickup, delivery)
                    best_increase = increase
            
            # Rest bleibt gleich
            if best_insertion:
                route_idx, new_route, pickup, delivery = best_insertion
                solution.routes[route_idx] = new_route
                unserved_requests.remove((pickup, delivery))
                solution._clear_cache()
            else:
                break
        return solution
