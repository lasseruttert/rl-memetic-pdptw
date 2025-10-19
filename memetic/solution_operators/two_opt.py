from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

class TwoOptOperator(BaseOperator):
    """2-opt for PDPTW with precedence constraint checking"""
    
    def __init__(self):
        super().__init__()
        self.name = "TwoOpt"
    
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        
        for route_idx, route in enumerate(new_solution.routes):
            if len(route) <= 3:  # Need at least 4 nodes for 2-opt
                continue
            
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Try reversing segment [i:j+1]
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        
                        # Check if precedence constraints are satisfied
                        if self._is_feasible_precedence(problem, new_route):
                            # Check if improvement (distance-based)
                            old_cost = self._route_distance(problem, route)
                            new_cost = self._route_distance(problem, new_route)
                            
                            if new_cost < old_cost:
                                route = new_route
                                new_solution.routes[route_idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
        
        new_solution._clear_cache()
        return new_solution
    
    def _is_feasible_precedence(self, problem: PDPTWProblem, route: list) -> bool:
        """Check if all pickups come before their deliveries"""
        position = {}
        for idx, node in enumerate(route):
            position[node] = idx
        
        for pickup, delivery in problem.pickups_deliveries:
            if pickup in position and delivery in position:
                if position[pickup] >= position[delivery]:
                    return False
        return True
    
    def _route_distance(self, problem: PDPTWProblem, route: list) -> float:
        """Calculate total distance of route"""
        if len(route) <= 1:
            return 0.0
        
        total = 0.0
        for i in range(len(route) - 1):
            total += problem.distance_matrix[route[i]][route[i+1]]
        return total