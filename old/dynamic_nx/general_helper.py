from difflib import SequenceMatcher
import random as r

def is_feasible(env, solution, location_mapping = None):
    """_summary_

    Args:
        env (_type_): _description_
        solution (_type_): _description_
        location_mapping (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if location_mapping is None:
        location_mapping = {}
        for vehicle_index, route in enumerate(solution):
            for location_index, location in enumerate(route):
                location_mapping[location] = (vehicle_index, location_index)
    
    current_requests = env.get_current_requests_indices()
    for pickup, delivery in current_requests:
        if pickup in location_mapping and delivery in location_mapping:
            pickup_position = location_mapping[pickup]
            delivery_position = location_mapping[delivery]
            # Check if pickup and delivery are done by the same vehicle
            if pickup_position[0] != delivery_position[0]: return False
            # check if pickup is done before delivery
            if pickup_position[1] > delivery_position[1]: return False
            
    if env.has_capacities:
        depot = env.get_depot().get_index()
        location_to_requests = env.get_location_to_requests()
        max_capacity = env.get_max_vehicle_capacity()
        for vehicle_index, route in enumerate(solution):
            capacity = 0
            for location in route:
                if location == depot:
                    continue
                request = location_to_requests[location]
                if location == request.get_pickup_location().get_index():
                    capacity += request.get_demand()
                if location == request.get_delivery_location().get_index():
                    capacity += -request.get_demand()
                    
                if capacity > max_capacity: return False
    
    if env.has_time_windows:
        # TODO: Check Timewindows
        pass
    
    return True

def is_viable(env, solution, location_mapping = None, check_feasible = False):
    """_summary_

    Args:
        env (_type_): _description_
        solution (_type_): _description_
        location_mapping (_type_, optional): _description_. Defaults to None.
        check_feasible (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if location_mapping is None:
        location_mapping = {}
        for vehicle_index, route in enumerate(solution):
            for location_index, location in enumerate(route):
                location_mapping[location] = (vehicle_index, location_index)
    
    if check_feasible:
        if not is_feasible(env, solution, location_mapping=location_mapping): return False
    
    # check if every request is served
    current_requests = env.get_current_requests_indices()
    vehicles = env.get_vehicles()
    for pickup, delivery in current_requests:
        if delivery not in location_mapping: return False
        if pickup not in location_mapping: 
            found = False
            for vehicle in vehicles:
                if pickup in vehicle.get_visited_indices(): found = True
            if not found: return False
    
    return True

def cost_increase(env, route, new_route):
    """_summary_

    Args:
        env (_type_): _description_
        route (_type_): _description_
        new_route (_type_): _description_

    Returns:
        _type_: _description_
    """
    depot = env.get_depot().get_index()
    
    route_copy = route.copy()
    if route and route[0] != depot: route_copy.insert(0, depot)
    route_copy.append(depot)
    
    new_route_copy = new_route.copy()
    if new_route and new_route[0] != depot: new_route_copy.insert(0, depot)
    new_route_copy.append(depot)
    
    distance_matrix = env.get_distance_matrix()
    
    route_length = 0
    for index in range(len(route_copy)-1):
        route_length += distance_matrix[index][index+1]
        
    new_route_length = 0
    for index in range(len(new_route_copy)-1):
        new_route_length += distance_matrix[index][index+1]
        
    return new_route_length - route_length


def simple_insertion(env, solution, unserved_requests):
    """_summary_

    Args:
        env (_type_): _description_
        solution (_type_): _description_
        unserved_requests (_type_): _description_

    Returns:
        _type_: _description_
    """
    depot = env.get_depot().get_index()
    while unserved_requests:
        best_insertion = None
        best_cost_increase = float("inf")
        
        for (pickup, delivery) in unserved_requests:
            for vehicle_index, route in enumerate(solution):
                if not route:
                    solution_copy = solution.copy()
                    new_route = [pickup, delivery]
                    solution_copy[vehicle_index] = new_route
                    if is_feasible(env, solution_copy):
                        diff = cost_increase(env, route, new_route)
                        if diff < best_cost_increase:
                            best_insertion = [vehicle_index, new_route, pickup, delivery]
                            best_cost_increase = diff
                for i in range(0, len(route)):
                    for j in range(i+1, len(route)):
                        solution_copy = solution.copy()
                        new_route = route.copy()
                        new_route.insert(j, delivery)
                        new_route.insert(i, pickup)
                        solution_copy[vehicle_index] = new_route
                        if is_feasible(env, solution_copy):
                            diff = cost_increase(env, route, new_route)
                            if diff < best_cost_increase:
                                best_insertion = [vehicle_index, new_route, pickup, delivery]
                                best_cost_increase = diff
        if best_insertion is None:
            print("No Solution found")
            return solution
        solution[best_insertion[0]] = best_insertion[1]
        unserved_requests.remove([best_insertion[2], best_insertion[3]])
    
    
    for route in solution:
        if route and route[0] != depot:
            route.insert(0, depot)
    return solution

def fitness(env, solution, metric="total_distance"):
    """_summary_

    Args:
        env (_type_): _description_
        solution (_type_): _description_
        metric (str, optional): _description_. Defaults to "total_distance".

    Returns:
        _type_: _description_
    """
    if not is_viable(env=env, solution=solution, check_feasible=True): return -1
    
    depot = env.get_depot().get_index()

    distance_matrix = env.get_distance_matrix()
    distances = []
    for route in solution:
        distance = 0
        if route and route[0] != depot: route = [depot] + route
        if route: route = route + [depot] #if distance to depot needs to be respected
        for i in range(len(route)-1):
            from_id = route[i]
            to_id = route[i + 1]
            distance += distance_matrix[from_id][to_id]
        distances.append(distance)

    if metric == "total_distance":
        return sum(distances)
    if metric == "maximum_distance":
        return max(distances)
    
def repair(env, solution):
    """_summary_

    Args:
        env (_type_): _description_
        solution (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * create a mapping of each location in each route of the solution to its vehicle and index
    location_mapping = {}
    for vehicle_index, route in enumerate(solution):
        for location_index, location in enumerate(route):
            location_mapping[location] = (vehicle_index, location_index)
    current_requests = env.get_current_requests_indices()
    unserved = []
    vehicles = env.get_vehicles()
    
    for request in current_requests:
        # * check if neither pickup nor delivery is part of a route
        if request[0] not in location_mapping and request[1] not in location_mapping:                
            unserved.append(request)
            continue
        # * check if only pickup is not part of the solution
        elif request[0] not in location_mapping and request[1] in location_mapping:
            pickup_done = False
            vehicle_index = None
            # search each vehicles set of visited locations for the pickup
            for index, vehicle in enumerate(vehicles):
                if request[0] in vehicle.get_visited_indices():
                    pickup_done = True
                    vehicle_index = index
            # remove the delivery if pickup hasnt been found or delivery is not in the correct vehicle
            if not pickup_done or location_mapping[request[1]][0] != vehicle_index:
                solution[location_mapping[request[1]][0]].remove(request[1])
                unserved.append(request)
                continue
        # * check if only delivery is not part of the solution
        elif request[0]  in location_mapping and request[1] not in location_mapping:
                solution[location_mapping[request[0]][0]].remove(request[0])
                unserved.append(request)
                continue
        # * check if pickup and delivery are not done by the same vehicle
        if location_mapping[request[0]][0] != location_mapping[request[1]][0]:
            solution[location_mapping[request[0]][0]].remove(request[0])
            solution[location_mapping[request[1]][0]].remove(request[1])
            unserved.append(request)
            continue
        # * check if delivery is done before pickup
        if location_mapping[request[0]][1] > location_mapping[request[1]][1]:
            solution[location_mapping[request[0]][0]].remove(request[0])
            solution[location_mapping[request[1]][0]].remove(request[1])
            unserved.append(request)
            continue

    for vehicle_index, route in enumerate(solution):
        for location_index, location in enumerate(route):
            # TODO: Check capacities
            # TODO: Check timewindows
            pass
        
    # * make sure the depot is at the start of each route if the route isnt empty
    for vehicle_index, route in enumerate(solution):
        if route:
            solution[vehicle_index].remove(env.get_depot().get_index())
            solution[vehicle_index].insert(0, env.get_depot().get_index())
            
    solution = simple_insertion(env, solution=solution, unserved_requests=unserved)
    
    return solution
    
def encode_solution(solution):
    """_summary_

    Args:
        solution (_type_): _description_

    Returns:
        _type_: _description_
    """
    routes = ["-".join(str(loc) for loc in route) for route in solution]
    routes.sort(key=len)
    return hash("".join(f"({route})" for route in routes))

def similarity(encoding1, encoding2):
    """_summary_

    Args:
        encoding1 (_type_): _description_
        encoding2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return SequenceMatcher(None, encoding1, encoding2).ratio()
