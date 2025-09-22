import numpy as np
from collections import defaultdict, Counter

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def print_solution(data, manager, routing, solution):
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    max_time = 0
    has_demands = "demands" in data
    has_time_windows = "time_windows" in data
    time_dimension = None
    if has_time_windows:
        time_dimension = routing.GetDimensionOrDie("Time")
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_time = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            extras = []
            if has_time_windows:
                time_var = time_dimension.CumulVar(index)
                start = solution.Min(time_var)
                end = solution.Max(time_var)
                extras.append(f"TimeWindow: {start}..{end}")
                route_time = max(route_time, end)
            if has_demands:
                demand = data["demands"][node]
                extras.append(f"Demand: {demand}")
            extras_str = f" ({', '.join(extras)})" if extras else ""
            plan_output += f" {node}{extras_str} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        if has_time_windows:
            plan_output += f"Time of the route: {route_time}\n"
        print(plan_output)
        total_distance += route_distance
        max_time = max(route_time, max_time)
    print(f"Total Distance of all routes: {total_distance}m")
    if has_time_windows:
        print(f"Total Time of all routes: {max_time}")


def expand_distance_matrix(distance_matrix, new_node_id, source_node_id):
    size = len(distance_matrix)
    if new_node_id >= size:
        for row in distance_matrix:
            row.append(1e9)
        new_row = [distance_matrix[source_node_id][j] for j in range(size)]
        new_row.append(0)
        distance_matrix.append(new_row)
        for i in range(size):
            distance_matrix[i][-1] = distance_matrix[i][source_node_id]
        distance_matrix[-1][-1] = 0

def convert_solution_to_routes(data, manager, routing, solution, mapping):
    routes = [[] for vehicle in range(data["num_vehicles"])]
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node in mapping:
                node = mapping[node]
            routes[vehicle_id].append(node)
            index = solution.Value(routing.NextVar(index))
    return routes

class OR_PDP_Solver:
    def __init__(self, max_seconds = 10, use_span = False, use_total_distance = False):
        self.max_seconds = max_seconds
        self.use_span = use_span
        self.use_total_distance = use_total_distance

    def solve_pdp(self, data, optimize ="distance", initial_routes = None, fixed_routes = None):
        # print information about the run
        instance_type = data.get("tag", "PDP")
        #print(f"Solving a {instance_type} instance...")
        distance_matrix = data.get("distance_matrix").copy()
        num_locations = data.get("num_locations", 10)
        num_vehicles = data.get("num_vehicles", 10)
        pairs = data.get("pickups_deliveries", [])
        max_vehicle_capacity = data.get("max_vehicle_capacity", 0)
        grid_size = data.get("grid_size", 100)
        depot = data.get("depot", 0)

        # fix duplicate pickup or delivery points to work for or tools
        count_as_pickup = defaultdict(int)
        count_as_delivery = defaultdict(int)
        reverse_mapping = {}
        free_id = num_locations

        seen = set()
        for index, (pickup, delivery) in enumerate(pairs):
            new_pickup = pickup
            new_delivery = delivery

            if pickup in seen:
                new_pickup = free_id
                free_id += 1
                reverse_mapping[new_pickup] = pickup
                expand_distance_matrix(distance_matrix, new_pickup, pickup)

            if delivery in seen:
                new_delivery = free_id
                free_id += 1
                reverse_mapping[new_delivery] = delivery
                expand_distance_matrix(distance_matrix, new_delivery, delivery)

            pairs[index] = [new_pickup, new_delivery]
            seen.add(pickup)
            seen.add(delivery)

        #data["pickups_deliveries"] = pairs
        data["num_locations"] = free_id
        num_locations = free_id

        if fixed_routes is not None:
            for index, location in enumerate(fixed_routes):
                for pair in pairs:
                    if pair[0] in reverse_mapping and reverse_mapping[pair[0]] == location:
                        fixed_routes[index] = pair[0]
                    if pair[1] in reverse_mapping and reverse_mapping[pair[1]] == location:
                        fixed_routes[index] = pair[1]
                    if pair[0] == location or pair[1] == location: break


        # track the dimensions which act as constraints for the solution space
        dimension_names = set()
        # create the manager
        if fixed_routes is not None:
            start_nodes = [depot] * num_vehicles
            end_nodes = [depot] * num_vehicles
            for vehicle_id, route in enumerate(fixed_routes):
                if not route:
                    continue
                start_nodes[vehicle_id] = route[0]
            manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, start_nodes, end_nodes)
        else:
            manager = pywrapcp.RoutingIndexManager(
                len(distance_matrix), num_vehicles, depot
            )

        # create the routing model
        routing = pywrapcp.RoutingModel(manager)

        # create the distance callback, based on the distance matrix
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        dimension_name = "Distance"
        max_vehicle_distance = data.get("max_vehicle_distance", grid_size * grid_size * num_locations)
        if self.use_total_distance:
            pass
        else:
            if fixed_routes is not None:
                vehicle_distances_with_buffer = []
                for route in fixed_routes:
                    route_length = 0
                    for i in range(len(route) - 1):
                        route_length += distance_matrix[route[i]][route[i+1]]
                    vehicle_distances_with_buffer.append(max_vehicle_distance + route_length)
                routing.AddDimensionWithVehicleCapacity(
                    transit_callback_index,
                    0,
                    vehicle_distances_with_buffer,  # maximum distance per vehicle, can be adjusted
                    True,
                    dimension_name,
                )
            else:
                routing.AddDimension(
                    transit_callback_index,
                    0,
                    max_vehicle_distance,  # maximum distance per vehicle, can be adjusted
                    True,
                    dimension_name,
                )

        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        dimension_names.add("Distance")

        # if demands are part of the instance, add demands as a constraint
        if "demands" in data:
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return data["demands"][from_node]
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,
                [max_vehicle_capacity] * num_vehicles, # maximum capacity per vehicle, can be adjusted
                True,
                "Capacity",
            )
            dimension_names.add("Capacity")

        # if time windows are part of the instance, add time as a constraint
        if "time_windows" in data:
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                travel_time = distance_matrix[from_node][to_node]
                service_time = 0
                if "service_time" in data:
                    service_time = data["service_times"][from_node]
                return travel_time + service_time
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            horizon = max(t[1] for t in data["time_windows"])
            routing.AddDimension(
                time_callback_index,
                30_000,
                horizon,
                False,
                "Time"
            )
            time_dimension = routing.GetDimensionOrDie("Time")
            dimension_names.add("Time")
            # add time windows to the locations
            for location_idx, (start, end) in enumerate(data["time_windows"]):
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(start, end)
            # add time window to the depot, in this case (0, total_time)
            depot_index = manager.NodeToIndex(data.get("depot", 0))
            time_dimension.CumulVar(depot_index).SetRange(
                data["time_windows"][depot_index][0], data["time_windows"][depot_index][1]
            )

        # set the dimension to be optimized
        if optimize == "distance":
            if self.use_span: distance_dimension.SetGlobalSpanCostCoefficient(100)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        elif optimize == "time":
            if "time_windows" not in data:
                raise ValueError("To optimize for time, time windows must be provided in data.")
            if self.use_span: time_dimension.SetGlobalSpanCostCoefficient(100)
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
        else:
            raise ValueError(f"Unknown optimize value: {optimize}")

        # add specific constraints for each dimension
        visited = set()
        if fixed_routes is not None:
            for vehicle_id, route in enumerate(fixed_routes):
                if not route or len(route) < 2:
                    continue
                actual_route = route[1:-1] if route[0] == data.get("depot", 0) and route[-1] == data.get("depot",0) else route
                if len(actual_route) < 2:
                    continue
                for i in range(len(actual_route) - 1):
                    from_node = actual_route[i]
                    to_node = actual_route[i + 1]
                    if from_node >= len(distance_matrix) or to_node >= len(distance_matrix):
                        print(f"Warning: Node {from_node} or {to_node} not in distance matrix")
                        continue
                    from_index = manager.NodeToIndex(from_node)
                    to_index = manager.NodeToIndex(to_node)
                    visited.add(from_node)
                    visited.add(to_node)
                    routing.solver().Add(routing.NextVar(from_index) == to_index)
                    routing.solver().Add(routing.VehicleVar(from_index) == vehicle_id)
                    routing.solver().Add(routing.VehicleVar(to_index) == vehicle_id)

        max_distance = max(max(row) for row in distance_matrix)
        penalty = max_distance * 5
        for request in pairs:
            if fixed_routes is not None and request[0] in visited and request[1] in visited:
                continue
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index) # add a pickup delivery pair
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index) # one pickup and delivery pair must be serviced by a single vehicle
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index)
                <= distance_dimension.CumulVar(delivery_index)
            ) # total distance traveled must be smaller at the pickup points than at the delivery point
            if "time_windows" in data:
                time_dimension = routing.GetDimensionOrDie("Time")
                routing.solver().Add(
                    time_dimension.CumulVar(pickup_index)
                    <= time_dimension.CumulVar(delivery_index)
                ) # pickup time must be earlier than delivery time
            routing.AddDisjunction([pickup_index], penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION # set the first solution heuristic
        )
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = self.max_seconds # time limit to find a solution
        if initial_routes is not None:
            assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)
            solution = routing.SolveFromAssignmentWithParameters(assignment, search_parameters)
        else:
            solution = routing.SolveWithParameters(search_parameters)

        #print(f"Constraints: {dimension_names}")
        if solution:
            #print_solution(data, manager, routing, solution)
            #print("Solution found")
            routes = convert_solution_to_routes(data, manager, routing, solution, reverse_mapping)
            return routes
        else:
            print("No solution")