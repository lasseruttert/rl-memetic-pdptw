from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

def greedy_insertion(problem: PDPTWProblem, solution: PDPTWSolution, unserved_requests: list[tuple[int, int]] = None) -> PDPTWSolution:
    if unserved_requests is None:
        unserved_requests = solution.get_unserved_requests(problem)
    
    while unserved_requests:
        best_insertion = None
        best_increase = float('inf')
        for request in unserved_requests:
            pickup, delivery = request
            looked_at_empty = False
            for route_idx, route in enumerate(solution.routes):
                if not route or len(route) <= 2 and route[0] == 0 and route[-1] == 0 and not looked_at_empty:  # Empty route, just add pickup and delivery
                    new_route = [0, pickup, delivery, 0]
                    if _is_feasible_insertion(problem, new_route):
                        looked_at_empty = True
                        increase = (problem.distance_matrix[0, pickup] + 
                                    problem.distance_matrix[pickup, delivery] + 
                                    problem.distance_matrix[delivery, 0])
                        if increase < best_increase:
                            best_insertion = route_idx, new_route, pickup, delivery
                            best_increase = increase
                for pickup_pos in range(1, len(route)):  # Position to insert pickup
                    for delivery_pos in range(pickup_pos + 1, len(route) + 1):  # Position to insert delivery
                        new_route = route[:pickup_pos] + [pickup] + route[pickup_pos:delivery_pos] + [delivery] + route[delivery_pos:]
                        if _is_feasible_insertion(problem, new_route):
                            increase = _cost_increase(problem, route, new_route)
                            if increase < best_increase:
                                best_insertion = route_idx, new_route, pickup, delivery
                                best_increase = increase
        
        if best_insertion:
            route_idx, new_route, pickup, delivery = best_insertion
            solution.routes[route_idx] = new_route
            solution.total_distance += best_increase
            unserved_requests.remove((pickup, delivery))
        else:
            print("No feasible insertion found for remaining requests")
            break  # No feasible insertion found, exit loop
    return solution
        
def _cost_increase(problem: PDPTWProblem, old_route, new_route) -> float:
    old_distance = sum(problem.distance_matrix[old_route[i], old_route[i + 1]] for i in range(len(old_route) - 1))
    new_distance = sum(problem.distance_matrix[new_route[i], new_route[i + 1]] for i in range(len(new_route) - 1))
    return new_distance - old_distance

def _is_feasible_insertion(problem: PDPTWProblem, route) -> bool:
    if len(route) < 2 or route[0] != 0 or route[-1] != 0:
        return False
    
    load = 0
    current_time = 0
    seen = set()
    
    demands = problem.demands
    distance_matrix = problem.distance_matrix
    time_windows = problem.time_windows
    service_times = problem.service_times
    vehicle_capacity = problem.vehicle_capacity
    delivery_to_pickup = problem.delivery_to_pickup
    pickup_to_delivery = problem.pickup_to_delivery
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        if to_node in delivery_to_pickup:
            pickup = delivery_to_pickup[to_node]
            if pickup not in seen:
                return False
        elif to_node not in pickup_to_delivery and to_node != 0:
            return False
        
        current_time += distance_matrix[from_node, to_node]
        tw_start, tw_end = time_windows[to_node]
        if current_time > tw_end:
            return False
        if current_time < tw_start:
            current_time = tw_start
        current_time += service_times[to_node]
        
        if to_node in seen:
            return False
        
        if to_node == 0 and i+1 < len(route) - 1:
            return False
        
        load += demands[to_node]
        if load < 0 or load > vehicle_capacity:
            return False
        
        seen.add(to_node)
    
    return True