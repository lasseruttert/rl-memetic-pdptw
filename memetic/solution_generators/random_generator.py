from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.solution_generators.base_generator import BaseGenerator

import random

class RandomGenerator(BaseGenerator):
    def __init__(self):
        pass
    
    def generate(self, problem: PDPTWProblem, num_solutions: int) -> PDPTWSolution:
        solutions = []
        for _ in range(num_solutions):
            solution = self._generate_single_solution(problem)
            solutions.append(solution)
        return solutions

    def _generate_single_solution(self, problem: PDPTWProblem) -> PDPTWSolution:
        routes = [[] for _ in range(problem.num_vehicles)]
        vehicle_index = None
        requests = problem.pickups_deliveries.copy()
        random.shuffle(requests)
        for pickup, delivery in requests:
            switch_prob = random.uniform(0.2, 0.6)
            if vehicle_index is None or random.random() < switch_prob:
                vehicle_index = random.randint(0, problem.num_vehicles - 1)
            pickup_index = random.randint(0, len(routes[vehicle_index]))
            routes[vehicle_index].insert(pickup_index, pickup)
            delivery_index = random.randint(pickup_index + 1, len(routes[vehicle_index]))
            routes[vehicle_index].insert(delivery_index, delivery)
        
        for route in routes:
            route.insert(0, 0)
            route.append(0)
        
        solution = PDPTWSolution(problem, routes)
        
        return solution