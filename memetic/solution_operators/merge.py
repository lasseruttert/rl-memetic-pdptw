from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

import random

class MergeOperator(BaseOperator):
    def __init__(self, type: str = "random", num_routes: int = 2, force: bool = True, reorder: bool = True):
        super().__init__()
        self.type = type
        self.num_routes = num_routes
        self.force = force
        self.reorder = reorder  # Whether to reorder routes to minimize connection distance
        self.name = f'Merge-{type}-N{num_routes}-{"F" if force else "nF"}-{"R" if reorder else "nR"}'
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        # merge number of routes into one
        new_solution = solution.clone()
        non_empty_routes = [route for route in new_solution.routes if len(route) > 2]
        
        if self.type == "random":
            # Randomly select routes to merge
            routes_to_merge = random.sample(non_empty_routes, min(self.num_routes, len(non_empty_routes)))
            self._delete_routes(new_solution, routes_to_merge)
            merged_route = self._merge_routes(routes_to_merge, problem)

        elif self.type == "min":
            # Select the shortest routes to merge
            sorted_routes = sorted(non_empty_routes, key=len)
            routes_to_merge = sorted_routes[:min(self.num_routes, len(sorted_routes))]
            self._delete_routes(new_solution, routes_to_merge)
            merged_route = self._merge_routes(routes_to_merge, problem)
                
        new_solution.routes.remove([0,0])  # Remove one empty route to maintain route count
        new_solution.routes.append(merged_route)
        new_solution._clear_cache()

        return new_solution
    
    def _merge_routes(self, routes_to_merge: list, problem: PDPTWProblem) -> list:
        """
        Merge multiple routes into a single route by concatenating and removing intermediate depots.

        Args:
            routes_to_merge: List of routes to merge
            problem: Problem instance for distance-based reordering

        Returns:
            A single merged route [0, customer1, customer2, ..., customerN, 0]
        """
        if not routes_to_merge:
            return [0, 0]

        if len(routes_to_merge) == 1:
            return routes_to_merge[0].copy()

        # Optionally reorder routes to minimize distance between consecutive endpoints
        if self.reorder and len(routes_to_merge) > 1:
            routes_to_merge = self._reorder_routes_for_merging(routes_to_merge, problem)

        # Start with depot
        merged_route = [0]

        # Add all customer visits from all routes (skip depots)
        for route in routes_to_merge:
            # Extract only non-depot nodes
            customer_visits = [node for node in route if node != 0]
            merged_route.extend(customer_visits)

        # End with depot
        merged_route.append(0)

        return merged_route

    def _reorder_routes_for_merging(self, routes: list, problem: PDPTWProblem) -> list:
        """
        Reorder routes to minimize distance between consecutive route endpoints.
        Uses a greedy nearest-neighbor approach.

        Args:
            routes: List of routes to reorder
            problem: Problem instance for distance calculations

        Returns:
            Reordered list of routes
        """
        if len(routes) <= 1:
            return routes

        # Start with the first route
        ordered = [routes[0]]
        remaining = routes[1:]

        while remaining:
            # Get the last customer of the current last route
            last_route = ordered[-1]
            last_customer = last_route[-2] if len(last_route) > 2 else 0  # -2 to skip depot at end

            # Find the route whose first customer is closest to last_customer
            best_idx = 0
            best_dist = float('inf')

            for idx, route in enumerate(remaining):
                first_customer = route[1] if len(route) > 2 else 0  # Skip depot at start
                if first_customer != 0:
                    dist = problem.distance_matrix[last_customer][first_customer]
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            # Add the best route to ordered and remove from remaining
            ordered.append(remaining.pop(best_idx))

        return ordered
    
    def _delete_routes(self, solution: PDPTWSolution, routes_to_delete):
        for route in routes_to_delete:
            solution.routes.remove(route)