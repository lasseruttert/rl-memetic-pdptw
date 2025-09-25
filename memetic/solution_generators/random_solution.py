from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
import random

def generate_random_solution(problem: PDPTWProblem) -> PDPTWSolution:
    routes = [[] for _ in range(problem.num_vehicles)]
    vehicle_index = None
    for pickup, delivery in problem.pickups_deliveries:
        if vehicle_index is None or random.random() < 0.3:
            vehicle_index = random.randint(0, problem.num_vehicles - 1)
        pickup_index = random.randint(0, len(routes[vehicle_index]))
        routes[vehicle_index].insert(pickup_index, pickup)
        delivery_index = random.randint(pickup_index + 1, len(routes[vehicle_index]))
        routes[vehicle_index].insert(delivery_index, delivery)
    
    for route in routes:
        if route:
            route.insert(0, 0)
            route.append(0)
    
    solution = PDPTWSolution(problem, routes)
    
    return solution