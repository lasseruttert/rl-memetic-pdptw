from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness
import random
class OrOptOperator(BaseOperator):
    def __init__(self,max_attempts: int = 10, single_route: bool = False, segment_length: int = 3):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        self.segment_length = segment_length
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        best_solution = solution.clone()
        best_fitness = fitness(problem, best_solution)
        attempts = 0
        
        while attempts < self.max_attempts:
            new_solution = best_solution.clone()
            # Select a random route
            if len(new_solution.routes) == 0:
                return new_solution
            route_idx = random.randint(0, len(new_solution.routes) - 1)
            route = new_solution.routes[route_idx]
            
            if len(route) < self.segment_length + 1:
                attempts += 1
                continue
            
            # Select a random segment
            start_idx = random.randint(0, len(route) - self.segment_length - 1)
            segment = route[start_idx:start_idx + self.segment_length]
            
            # Remove the segment from the route
            del route[start_idx:start_idx + self.segment_length]
            
            # Select a new position to insert the segment
            if self.single_route:
                insert_route_idx = route_idx
            else:
                insert_route_idx = random.randint(0, len(new_solution.routes) - 1)
            insert_route = new_solution.routes[insert_route_idx]
            insert_position = random.randint(0, len(insert_route))
            
            # Insert the segment at the new position
            for i, customer in enumerate(segment):
                insert_route.insert(insert_position + i, customer)
            
            new_fitness = fitness(problem, new_solution)
            if new_fitness < best_fitness and new_solution.check_feasibility():
                best_solution = new_solution
                best_fitness = new_fitness
                attempts = 0  # Reset attempts on improvement
            else:
                attempts += 1
        
        return best_solution