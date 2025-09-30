from memetic.insertion.simple_insertion_heuristic import greedy_insertion
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

def reinsertion_repair(problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
    """Remove problematic pairs iteratively."""
    
    for route_idx in range(len(solution.routes)):
        changed = True
        while changed:
            changed = False
            route = solution.routes[route_idx]
            
            if len(route) <= 2:
                break
                
            load = 0
            current_time = 0
            seen = set()
            
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                
                # Check all constraints
                violation = False
                pair_to_remove = None
                
                if to_node in seen:
                    violation = True
                    pair_to_remove = problem.get_pair(to_node) if to_node != 0 else (to_node,)
                
                if to_node in problem.delivery_to_pickup:
                    pickup = problem.delivery_to_pickup[to_node]
                    if pickup not in seen:
                        violation = True
                        pair_to_remove = (pickup, to_node)
            
                # Time and capacity checks
                current_time += problem.distance_matrix[from_node, to_node]
                current_time = max(current_time, problem.time_windows[to_node][0])
                
                if current_time > problem.time_windows[to_node][1]:
                    violation = True
                    pair_to_remove = problem.get_pair(to_node)
                
                elif not violation:
                    current_time += problem.service_times[to_node]
                    load += problem.demands[to_node]
                    
                    if load < 0 or load > problem.vehicle_capacity:
                        violation = True
                        pair_to_remove = problem.get_pair(to_node)
                
                if violation and pair_to_remove:
                    # Remove pair immediately and restart
                    nodes_to_remove = set(pair_to_remove)
                    solution.routes[route_idx] = [n for n in route if n not in nodes_to_remove]
                    changed = True
                    break  # Restart this route
                
                seen.add(to_node)
    
    # Reinsert
    unserved = solution.get_unserved_requests(problem)
    if unserved:
        solution = greedy_insertion(problem, solution, unserved)
    
    solution._clear_cache()
    return solution