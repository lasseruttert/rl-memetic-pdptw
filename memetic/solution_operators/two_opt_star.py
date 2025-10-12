from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class TwoOptStarOperator(BaseOperator): # TODO: fix, creates invalid solutions
    """2-opt* exchanges route tails between two different routes.
    
    For routes r1 and r2, cuts at positions i and j, then swaps:
    r1 = r1[:i] + r2[j:]
    r2 = r2[:j] + r1[i:]
    
    Must check precedence constraints and capacity constraints.
    """
    
    def __init__(self, first_improvement: bool = True, max_iterations: int = 100):
        super().__init__()
        self.first_improvement = first_improvement
        self.max_iterations = max_iterations
    
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        
        if len(new_solution.routes) < 2:
            return new_solution
        
        iteration = 0
        improved = True
        
        while improved and iteration < self.max_iterations:
            improved = False
            
            # Try all pairs of routes
            for r1_idx in range(len(new_solution.routes)):
                for r2_idx in range(r1_idx + 1, len(new_solution.routes)):
                    route1 = new_solution.routes[r1_idx]
                    route2 = new_solution.routes[r2_idx]
                    
                    if len(route1) <= 1 or len(route2) <= 1:
                        continue
                    
                    # Try all cut positions (skip depot at start)
                    for i in range(1, len(route1)):
                        for j in range(1, len(route2)):
                            # Create new routes by swapping tails
                            new_r1 = route1[:i] + route2[j:]
                            new_r2 = route2[:j] + route1[i:]
                            
                            # Check feasibility
                            if not (self._is_feasible_precedence(problem, new_r1) and
                                   self._is_feasible_precedence(problem, new_r2)):
                                continue
                            
                            # Calculate improvement
                            old_cost = (self._route_distance(problem, route1) + 
                                       self._route_distance(problem, route2))
                            new_cost = (self._route_distance(problem, new_r1) + 
                                       self._route_distance(problem, new_r2))
                            
                            if new_cost < old_cost:
                                new_solution.routes[r1_idx] = new_r1
                                new_solution.routes[r2_idx] = new_r2
                                new_solution._clear_cache()
                                improved = True
                                
                                if self.first_improvement:
                                    return new_solution
                                else:
                                    # Update for next iteration
                                    route1 = new_r1
                                    route2 = new_r2
            
            iteration += 1
        
        new_solution._clear_cache()
        return new_solution
    
    def _is_feasible_precedence(self, problem: PDPTWProblem, route: list) -> bool:
        """Check if all pickups come before their deliveries"""
        position = {node: idx for idx, node in enumerate(route)}
        
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