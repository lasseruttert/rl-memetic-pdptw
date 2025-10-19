from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

class RemoveRouteOperator(BaseOperator):
    def __init__(self):
        super().__init__()
        self.name = "RemoveRoute"
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        if not new_solution.routes:
            return new_solution  # No routes to remove from
        
        # Find the route with the least number of customers but more than just the depot
        route_to_remove = min((route for route in new_solution.routes if len(route) > 2), key=len)
        new_solution.routes.remove(route_to_remove)
        new_solution.routes.append([])  # Maintain the same number of routes by adding an empty one
        new_solution._clear_cache()
        
        return new_solution