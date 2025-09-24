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
            
            if i == 0 and from_node != 0:
                if use_prints: print("Route does not start at depot:", route)
                return False  # Route must start at depot
            
            if i+1 == len(route) - 1 and to_node != 0:
                if use_prints: print("Route does not end at depot:", route)
                return False  # Route must end at depot
            
            if i+1 < len(route) - 1 and to_node == 0:
                if use_prints: print("Depot visited in the middle of route:", route)
                return False  # Depot can only be at start or end
            
            if to_node in seen:
                if use_prints: print("Node visited multiple times:", to_node)
                return False  # node already served
            # Check if pickup and delivery pairs are respected
            if problem.is_delivery(to_node):
                pickup = problem.get_pair(to_node)[0]
                if pickup not in seen:
                    if use_prints: print("Delivery before pickup:", to_node)
                    return False  # Delivery before pickup
            else:
                if not problem.is_pickup(to_node) and to_node != 0:
                    if use_prints: print("Invalid node:", to_node)
                    return False  # Invalid node (not depot, pickup, or delivery)
                
            # Update load
            load += problem.demands[to_node]
            if load < 0 or load > problem.vehicle_capacity:
                if use_prints: print("Capacity violation at node:", to_node)
                return False
            
            # Travel time
            travel_time = problem.distance_matrix[from_node, to_node]
            current_time += travel_time
            
            # Check time window
            tw_start, tw_end = problem.time_windows[to_node]
            if current_time < tw_start:
                current_time = tw_start
                # print("Arrived too early at node:", to_node)
                # return False  # Arrived before the time window opens
            if current_time > tw_end:
                if use_prints: print("Arrived too late at node:", to_node)
                return False  # Arrived after the time window closes
            
            # Service time
            current_time += problem.service_times[to_node]
            
            seen.add(to_node)
        
        seen_total.update(seen)
        
    if seen_total != set(node.index for node in problem.nodes):
        if use_prints: print("Not all nodes served. Served:", seen_total)
        return False  # Not all nodes served
    
    return True