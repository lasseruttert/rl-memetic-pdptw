from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from reader import li_liam_to_or
import os
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
def create_routing_model(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'],
                                           data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [data['vehicle_capacity']] * data['num_vehicles'],
        True,
        'Capacity'
    )

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['distance_matrix'][from_node][to_node]
        return travel_time + data['service_times'][from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        30_000,  
        30_000,
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')

    for node_idx, (start, end) in enumerate(data['time_windows']):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetRange(start, end)

    for pickup_idx, delivery_idx in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(pickup_idx)
        delivery_index = manager.NodeToIndex(delivery_idx)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        routing.solver().Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    return routing, manager

def solve(data):
    routing, manager = create_routing_model(data)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return return_solution(data, routing, manager, solution)
    else:
        print("Keine Lösung gefunden.")

def print_solution(data, routing, manager, solution):
    time_dimension = routing.GetDimensionOrDie('Time')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route für Fahrzeug {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            load = data['demands'][node_index]
            cumul_time = solution.Value(time_dimension.CumulVar(index))
            route_load += load
            plan_output += f' {node_index} (Ladung: {route_load}, Zeit: {cumul_time}) ->'
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        node_index = manager.IndexToNode(index)
        cumul_time = solution.Value(time_dimension.CumulVar(index))
        plan_output += f' {node_index} (Zeit: {cumul_time})\n'
        plan_output += f'Distanz der Route: {route_distance}\n'
        plan_output += f'Gesamtladung: {route_load}\n'
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f'Gesamtdistanz aller Routen: {total_distance}')

def return_solution(data, routing, manager, solution):
    time_dimension = routing.GetDimensionOrDie('Time')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route für Fahrzeug {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            load = data['demands'][node_index]
            cumul_time = solution.Value(time_dimension.CumulVar(index))
            route_load += load
            plan_output += f' {node_index} (Ladung: {route_load}, Zeit: {cumul_time}) ->'
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        node_index = manager.IndexToNode(index)
        cumul_time = solution.Value(time_dimension.CumulVar(index))
        plan_output += f' {node_index} (Zeit: {cumul_time})\n'
        plan_output += f'Distanz der Route: {route_distance}\n'
        plan_output += f'Gesamtladung: {route_load}\n'
        total_distance += route_distance
        total_load += route_load
    return total_distance

if __name__ == "__main__":
    solutions = {}
    folder_path = '../data/benchmarks/pdp_100'

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data = li_liam_to_or(file_path)
            clear_terminal()
            print(f"Solving instance: {filename}...")
            print(f"Current found solutions: \n {solutions}")
            solutions[filename] = solve(data)
    print(solutions)