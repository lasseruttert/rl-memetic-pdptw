from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

def fitness(problem: PDPTWProblem, solution: PDPTWSolution) -> float: 
    """Calculates the fitness of a solution, based on number of vehicles used and total distance of the routes. Feasible solutions are preferred over infeasible ones, based on a penalty system.

    Args:
        problem (PDPTWProblem): a PDPTW problem instance
        solution (PDPTWSolution): a PDPTW solution instance, which is evaluated on the given problem

    Returns:
        float: the fitness value, lower is better
    """
    fitness = solution.total_distance 
    fitness += _penalty(problem, solution)
    
    percent_vehicles_used = solution.num_vehicles_used / problem.num_vehicles 
    fitness *= (1 + percent_vehicles_used)
        
    return fitness

def _penalty(problem: PDPTWProblem, solution: PDPTWSolution) -> float:
    """ Calculates the penalty of a solution based on the number of violations of constraints. If no violations are found, the solution is marked as feasible.

    Args:
        problem (PDPTWProblem): a PDPTW problem instance
        solution (PDPTWSolution): a PDPTW solution instance, which is evaluated on the given problem

    Returns:
        float: the penalty value, lower is better
    """
    pickup_to_delivery = problem.pickup_to_delivery
    delivery_to_pickup = problem.delivery_to_pickup
    
    demands = problem.demands
    vehicle_capacity = problem.vehicle_capacity
    distance_matrix = problem.distance_matrix
    time_windows = problem.time_windows
    service_times = problem.service_times
    nodes = problem.nodes
    
    num_violations = 0
    
    seen_total = set()
    for route in solution.routes:
        load = 0
        current_time = 0
        seen = set()
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # * Check if the route start at the depot
            if i == 0 and from_node != 0:
                num_violations += 1
            
            # * Check if the route ends at the depot
            if i+1 == len(route) - 1 and to_node != 0:
                num_violations += 1
            
            # * Check if depot is visited in the middle of the route
            if i+1 < len(route) - 1 and to_node == 0:
                num_violations += 1
            
            # * Check if node has already been served
            if to_node in seen:
                num_violations += 1
             
            # * Check if pickup happens before delivery
            if to_node in delivery_to_pickup:
                pickup = delivery_to_pickup[to_node]
                if pickup not in seen:
                    num_violations += 1
            # * Check if node is valid (depot, pickup, or delivery)
            else:
                if to_node not in pickup_to_delivery and to_node != 0:
                    num_violations += 1
                
            # * Check if vehicle capacities are respected
            load += demands[to_node]
            if load < 0 or load > vehicle_capacity:
                num_violations += 1
            
            # * Check if time windows are respected
            travel_time = distance_matrix[from_node, to_node]
            current_time += travel_time
            
            tw_start, tw_end = time_windows[to_node]
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                num_violations += 1
            current_time += service_times[to_node]
            
            seen.add(to_node)
        
        seen_total.update(seen)

    # * Check if all nodes are served        
    if seen_total != set(node.index for node in nodes):
        not_served = set(node.index for node in nodes) - seen_total
        num_violations += len(not_served)
    
    if num_violations == 0:
        solution._is_feasible = True
        return 0.0
    else:
        solution._is_feasible = False
        return num_violations * 0.05 * problem.distance_baseline + 1 * problem.distance_baseline # TODO: make each penalty type have different weights