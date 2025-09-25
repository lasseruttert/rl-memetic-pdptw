from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.insertion.insertion_heuristic import greedy_insertion

class RouteEliminationOperator(BaseOperator):
    def __init__(self, problem: PDPTWProblem):
        super().__init__(problem)
        
    def apply(self, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        if len(new_solution.routes) <= 1:
            print("Cannot eliminate route: only one route present.")
            return new_solution  # Cannot eliminate a route if there's only one
    
        # Find the route with the least number of customers but more than just the depot
        route_to_eliminate = min((route for route in new_solution.routes if len(route) > 2), key=len)
        print("Elim")
        print(route_to_eliminate)
        new_solution.routes.remove(route_to_eliminate)
        new_solution.routes.append([])  # Maintain the same number of routes by adding an empty one
        new_solution._clear_cache()
        
        new_solution = greedy_insertion(problem=self.problem, solution=new_solution, allow_new_vehicles=False)
        
        return new_solution