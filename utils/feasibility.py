from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

def is_feasible(problem: PDPTWProblem, solution: PDPTWSolution, use_prints = False) -> bool:
    """Checks if a given PDPTW solution is feasible with respect to capacity and time windows."""
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
                if use_prints: print("Route does not start at depot:", route)
                return False
            
            # * Check if the route ends at the depot
            if i+1 == len(route) - 1 and to_node != 0:
                if use_prints: print("Route does not end at depot:", route)
                return False
            
            # * Check if depot is visited in the middle of the route
            if i+1 < len(route) - 1 and to_node == 0:
                if use_prints: print("Depot visited in the middle of route:", route)
                return False  
            
            # * Check if node has already been served
            if to_node in seen:
                if use_prints: print("Node visited multiple times:", to_node)
                return False 
             
            # * Check if pickup happens before delivery
            if problem.is_delivery(to_node):
                pickup = problem.get_pair(to_node)[0]
                if pickup not in seen:
                    if use_prints: print("Delivery before pickup:", to_node)
                    return False  
            # * Check if node is valid (depot, pickup, or delivery)
            else:
                if not problem.is_pickup(to_node) and to_node != 0:
                    if use_prints: print("Invalid node:", to_node)
                    return False  
                
            # * Check if vehicle capacities are respected
            load += problem.demands[to_node]
            if load < 0 or load > problem.vehicle_capacity:
                if use_prints: print("Capacity violation at node:", to_node)
                return False
            
            # * Check if time windows are respected
            travel_time = problem.distance_matrix[from_node, to_node]
            current_time += travel_time
            
            tw_start, tw_end = problem.time_windows[to_node]
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                if use_prints: print("Arrived too late at node:", to_node)
                return False 
            current_time += problem.service_times[to_node]
            
            seen.add(to_node)
        
        seen_total.update(seen)

    # * Check if all nodes are served        
    if seen_total != set(node.index for node in problem.nodes):
        not_served = set(node.index for node in problem.nodes) - seen_total
        if use_prints: print("Not all nodes served. Not served:", not_served)
        return False  
    
    return True

# TODO do we need to check if pickup and delivery is in the same route?