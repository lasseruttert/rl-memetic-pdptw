from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.insertion.greedy_insertion import GreedyInsertion
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator

import random

class GuidedEjectionSearch:
    def __init__(self, max_ejection_iterations: int = 5):
        self.max_ejection_iterations = max_ejection_iterations
        self.inserter = GreedyInsertion(allow_new_vehicles=False)
        self.swapper = SwapBetweenOperator()
        self.transferrer = TransferOperator()
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        
        if len(new_solution.routes) <= 1:
            return new_solution
            
        non_empty_routes = [route for route in new_solution.routes if len(route) > 2]
        if not non_empty_routes:
            return new_solution
        
        route_to_eliminate = random.choice(non_empty_routes)
        new_solution.routes.remove(route_to_eliminate)
        new_solution.routes.append([0, 0])
        new_solution._clear_cache()
        
        # Try to reinsert all requests
        for ejection_iter in range(self.max_ejection_iterations):
            new_solution = self.inserter.insert(problem, new_solution)
            
            unserved = new_solution.get_unserved_requests(problem=problem)
            
            if not unserved:
                break
            
            if ejection_iter < self.max_ejection_iterations - 1:
                new_solution = self._guided_ejection(problem, new_solution, unserved)
            else:
                new_solution = self._perturb_solution(problem, new_solution)
        
        return new_solution
    
    def _guided_ejection(self, problem: PDPTWProblem, solution: PDPTWSolution, 
                        unserved_requests: list) -> PDPTWSolution:
        """Eject 1-2 'easy' requests to make room for 'difficult' ones"""
        new_solution = solution.clone()
        served_requests = new_solution.get_served_requests(problem=problem)
        
        if not served_requests:
            return solution
        
        num_to_eject = min(2, len(served_requests))
        ejection_candidates = self._select_ejection_candidates(
            problem, new_solution, served_requests, num_to_eject
        )
        
        for req in ejection_candidates:
            new_solution.remove_request(problem, req)
        
        return new_solution
    
    def _select_ejection_candidates(self, problem: PDPTWProblem, solution: PDPTWSolution,
                                   candidates: list, num_to_eject: int) -> list:
        """Select requests with largest time windows (easiest to reinsert)"""
        scores = []
        for req in candidates:
            pickup_idx, delivery_idx = problem.get_request_nodes(req)
            tw_slack = (problem.time_windows[pickup_idx][1] - 
                       problem.time_windows[pickup_idx][0])
            scores.append((req, tw_slack))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [req for req, _ in scores[:num_to_eject]]
    
    def _perturb_solution(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        """Perturb solution"""
        new_solution = solution.clone()
        if random.random() < 0.5:
            new_solution = self.swapper.apply(problem, new_solution)
        else:
            new_solution = self.transferrer.apply(problem, new_solution)
        return new_solution