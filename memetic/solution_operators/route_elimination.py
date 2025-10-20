from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.insertion.greedy_insertion import GreedyInsertion
from memetic.insertion.regret_insertion import Regret2Insertion



class RouteEliminationOperator(BaseOperator):
    def __init__(self, insertion_heuristic: str = 'greedy'):
        super().__init__()
        if insertion_heuristic == 'greedy':
            self.insertion_heuristic = GreedyInsertion(allow_new_vehicles=False)
        if insertion_heuristic == 'regret2':
            self.insertion_heuristic = Regret2Insertion()
        self.name = f'RouteElimination'
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        if len(new_solution.routes) <= 1:
            return new_solution  # Cannot eliminate a route if there's only one
    
        # Find the route with the least number of customers but more than just the depot
        non_empty_routes = [route for route in new_solution.routes if len(route) > 2]
    
        if not non_empty_routes:
            # No routes to eliminate, return unchanged
            return new_solution
    
        route_to_eliminate = min(non_empty_routes, key=len)
        route_to_eliminate_idx = new_solution.routes.index(route_to_eliminate)
        new_solution.routes.remove(route_to_eliminate)
        new_solution.routes.append([0,0])  # Maintain the same number of routes by adding an empty one
        new_solution._clear_cache()
        
        new_solution = self.insertion_heuristic.insert(problem, new_solution)
        
        return new_solution