from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from copy import deepcopy

class ORToolsSolver:
    def __init__(self, max_seconds: int = 10, use_span: bool = False, minimize_num_vehicles: bool = False):
        self.max_seconds = max_seconds
        self.use_span = use_span
        self.minimize_num_vehicles = minimize_num_vehicles
        
    def solve(self, problem: PDPTWProblem) -> PDPTWSolution:
        data = problem.data
        if self.minimize_num_vehicles:
            for i in range(1, data['num_vehicles']):
                new_data = deepcopy(data)
                new_data['num_vehicles'] = i
                try:
                    return self.get_solution(problem, new_data)
                except ValueError:
                    continue
            raise ValueError("No solution found with the given number of vehicles")
        else:
            return self.get_solution(problem, data)
        
    def get_solution(self, problem: PDPTWProblem, data) -> PDPTWSolution:
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data['distance_matrix'][from_node][to_node] * 100)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        routing.AddDimension(
            transit_callback_index,
            0,
            data["time_windows"][data['depot']][1].item() * 100,
            True,
            "Distance"
        )
        distance_dimension = routing.GetDimensionOrDie("Distance")
        
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(data['demands'][from_node])
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [int(data['vehicle_capacity'])] * data['num_vehicles'],
            True,
            'Capacity'
        )
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = data['distance_matrix'][from_node][to_node]
            service_time = data['service_times'][from_node]
            return int((travel_time + service_time) * 100)
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            int(data["time_windows"][data['depot']][1].item() * 100),
            int(data["time_windows"][data['depot']][1].item() * 100),
            True,
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        for node_index in range(1, len(data['time_windows'])):
            index = manager.NodeToIndex(node_index)
            time_window = data['time_windows'][node_index]
            time_dimension.CumulVar(index).SetRange(time_window[0].item() * 100, time_window[1].item() * 100)
            
        for request in data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(delivery_index)
            )
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index) + int(data["service_times"][request[0]]) <= time_dimension.CumulVar(delivery_index)
            )
            
        if self.use_span:
            distance_dimension.SetGlobalSpanCostCoefficient(100)
        
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.time_limit.seconds = self.max_seconds
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._get_solution(data, manager, routing, solution)
        else:
            raise ValueError("No solution found")
        
    def _get_solution(self, data, manager, routing, solution) -> PDPTWSolution:
        routes = []
        total_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            if not routing.IsVehicleUsed(solution, vehicle_id):
                continue
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            routes.append(route)
            total_distance += route_distance
        
        total_distance /= 100  # convert back to original scale
        return PDPTWSolution(routes=routes, total_distance=total_distance)