import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import random as r
import numpy as np
from dataclasses import dataclass
import os

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_pickup_deliveries(num_locations, num_pairs):
    candidates = list(range(1, num_locations))
    r.shuffle(candidates)
    pickup_deliveries = []
    used = set()
    for _ in range(num_pairs):
        pickup = None
        while candidates:
            candidate = candidates.pop()
            if candidate not in used:
                pickup = candidate
                used.add(pickup)
                break
        if pickup is None:
            break

        delivery = None
        while candidates:
            candidate = candidates.pop()
            if candidate not in used and candidate != pickup:
                delivery = candidate
                used.add(delivery)
                break
        if delivery is None:
            break
        pickup_deliveries.append([pickup, delivery])
    return pickup_deliveries

class DynamicPDPEnv:
    def __init__(self, grid_size, num_locations, num_initial_pairs, num_vehicles, max_vehicle_distance = None, max_vehicle_total_distance = None,
                    has_capacities = False, max_demand = None, max_vehicle_capacity = None, has_time_windows = False, horizon = None, service_time = None,
                    has_priorities = False, use_li_lim = False):
        """_summary_

        Args:
            grid_size (_type_): _description_
            num_locations (_type_): _description_
            num_initial_pairs (_type_): _description_
            num_vehicles (_type_): _description_
            max_vehicle_distance (_type_, optional): _description_. Defaults to None.
            max_vehicle_total_distance (_type_, optional): _description_. Defaults to None.
            has_capacities (bool, optional): _description_. Defaults to False.
            max_demand (_type_, optional): _description_. Defaults to None.
            max_vehicle_capacity (_type_, optional): _description_. Defaults to None.
            has_time_windows (bool, optional): _description_. Defaults to False.
            horizon (_type_, optional): _description_. Defaults to None.
            service_time (_type_, optional): _description_. Defaults to None.
            has_priorities (bool, optional): _description_. Defaults to False.
            use_li_lim (bool, optional): _description_. Defaults to False.
        """
        # * set all the variables given to the constructor
        self.grid_size = grid_size
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.has_capacities = has_capacities
        self.max_demand = max_demand
        self.max_vehicle_capacity = max_vehicle_capacity
        self.has_time_windows = has_time_windows
        self.horizon = horizon
        self.service_time = service_time
        self.has_priorities = has_priorities
        self.current_timestep = 0

        # * if use_li_lim is true, use a li lim dataset as base for the environment
        if use_li_lim:
            # TODO: Instead of a random initial data object, use one of li lim datasets
            pass

        # * create the locations of the environment, including the depot
        self.locations = []
        self.locations_dict = {}
        for i in range(self.num_locations):
            if i == 0:
                coord = (int(grid_size/2),int(grid_size/2))
                depot = Location(env=self, index=0, coords=coord,location_type= "Depot")
                self.locations.append(depot)
                self.locations_dict[0] = depot
                self.depot = depot
            else:
                coord = (r.randint(0, grid_size), r.randint(0, grid_size))
                location = Location(env=self, index=i, coords=coord, location_type="Generic")
                self.locations.append(location)
                self.locations_dict[i] = location

        # * create the distance matrix
        self.distance_matrix = [
            [
                manhattan(self.locations[i].get_coords(), self.locations[j].get_coords())
                for j in range(self.num_locations)
            ]
            for i in range(self.num_locations)
        ]

        # * create the vehicles
        self.vehicles = []
        self.vehicles_dict = {}
        for i in range(self.num_vehicles):
            vehicle = Vehicle(env=self, index=i, capacity=self.max_vehicle_capacity)
            self.vehicles.append(vehicle)
            self.vehicles_dict[i] = vehicle
            self.depot.vehicles.add(vehicle)

        if max_vehicle_distance is None: self.max_vehicle_distance = self.grid_size * self.grid_size * self.num_locations
        else: self.max_vehicle_distance = max_vehicle_distance

        if max_vehicle_total_distance is None: self.max_vehicle_total_distance = self.grid_size * self.grid_size * self.num_locations
        else: self.max_vehicle_total_distance = max_vehicle_total_distance
        
        # * create the initial requests
        self.current_requests = []
        self.current_requests_dict = {}
        self.request_history = []
        self.pickup_to_delivery = {}
        self.delivery_to_pickup = {}
        self.location_to_request = {}
        initial_pairs = generate_pickup_deliveries(self.num_locations, num_initial_pairs)
        self.num_pairs = len(initial_pairs)
        for i, pair in enumerate(initial_pairs):
            pickup_location_index = pair[0]
            delivery_location_index = pair[1]
            pickup_location = self.locations_dict[pickup_location_index]
            delivery_location = self.locations_dict[delivery_location_index]
            request = Request(env=self, index=i, creation_time=0, pickup_location=pickup_location, delivery_location=delivery_location)
            self.add_request(request)


        self.graph = None
        
    def __str__(self):
        """Defines the string a DynamicPDPEnv object returns, therefore also defining its print statement

        Returns:
            str: A string containing information about the current state of the environment, like its requests and vehicles
        """
        string = ""
        string += "Requests: \n"
        for request in self.current_requests:
            string += "Request " + str(request.get_index()) + f", created at {request.get_creation_time()}" + f", Pickup {request.get_pickup_location().get_index()} within {request.get_time_window()[0]}, Delivery {request.get_delivery_location().get_index()} within {request.get_time_window()[1]}" + os.linesep
            
        string += "Vehicles: \n"
        for vehicle in self.vehicles:
            if vehicle.get_next_location() is None: continue
            string += f"Vehicle {vehicle.get_index()}, Last: {vehicle.get_last_location().get_index()}, Next: {vehicle.get_next_location().get_index()}, Distance: {vehicle.get_distance_to_next()}" + os.linesep
            string += f"Visited: {vehicle.get_visited_indices()}" + os.linesep
        return string
    
    def step(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        update = False
        for vehicle in self.vehicles:
            if vehicle.step(): update = True
        self.current_timestep += 1
        if update:
            for vehicle in self.vehicles:
                for index, request in enumerate(self.current_requests):
                    pickup = request.get_pickup_location()
                    delivery = request.get_delivery_location()
                    if delivery in vehicle.get_visited() and pickup in vehicle.get_visited():
                        self.remove_request(request, vehicle)
        if update: 
            return True
        else:
            return False
    
    def add_request(self, request):
        """_summary_

        Args:
            request (_type_): _description_
        """
        # * get the correct locations and indices
        index = request.get_index()
        pickup_index = request.get_pickup_location().get_index()
        delivery_index = request.get_delivery_location().get_index()
        # * add the mappings
        self.pickup_to_delivery[pickup_index] = delivery_index
        self.delivery_to_pickup[delivery_index] = pickup_index
        self.location_to_request[pickup_index] = request
        self.location_to_request[delivery_index] = request
        # * check for constraints
        if self.has_capacities:
            demand_value = r.randint(1, self.max_demand)
            request.set_demand(demand_value)
        if self.has_time_windows:
            time_window = self.generate_time_windows(request)
            request.set_time_windows(time_window)
        # * add the request
        self.current_requests.append(request)
        self.current_requests_dict[index] = request
        self.request_history.append(request)
        
        return
    
    def remove_request(self, request, vehicle = None):
        """_summary_

        Args:
            request (_type_): _description_
            vehicle (_type_, optional): _description_. Defaults to None.
        """
        # * get the correct locations and indices
        index = request.get_index()
        pickup = request.get_pickup_location()
        pickup_index = pickup.get_index()
        delivery = request.get_delivery_location()
        delivery_index = delivery.get_index()
        # * remove the request
        self.current_requests.remove(request)
        self.current_requests_dict.pop(index)
        # * remove mappings
        self.pickup_to_delivery.pop(pickup_index)
        self.delivery_to_pickup.pop(delivery_index)
        self.location_to_request.pop(pickup_index)
        self.location_to_request.pop(delivery_index)
        # * remove locations from vehicles visited set
        if vehicle is None: 
            for vehicle_iter in self.vehicles:
                if delivery in vehicle_iter.get_visited() and pickup in vehicle_iter.get_visited(): vehicle = vehicle_iter
        vehicle.visited.remove(pickup)
        vehicle.visited.remove(delivery)
        
        return
        
    def generate_time_windows(self, request):
        """_summary_

        Args:
            request (_type_): _description_

        Returns:
            _type_: _description_
        """
        # * get the correct locations and indices
        pickup = request.get_pickup_location()
        pickup_index = pickup.get_index()
        delivery = request.get_delivery_location()
        delivery_index = delivery.get_index()
        
        # * calculate the distances to the pickup from the depot and between pickup and delivery
        distance_start = self.distance_matrix[self.depot.get_index()][pickup_index]
        distance_between = self.distance_matrix[pickup_index][delivery_index]
        
        # * create the timewindow for pickup
        pickup_earliest = self.current_timestep + distance_start + r.randint(0, (self.horizon * 0.2))
        pickup_latest = pickup_earliest + r.randint(self.horizon * 0.1, self.horizon * 0.3)
        
        # * create the timewindow for delivery
        delivery_earliest = pickup_earliest + distance_between 
        delivery_latest = min(delivery_earliest * 2, self.current_timestep + self.horizon)
        
        return (pickup_earliest, pickup_latest), (delivery_earliest, delivery_latest)
    
    def readjust_time_windows(self, request):
        """_summary_

        Args:
            request (_type_): _description_
        """
        # * get the needed time steps
        creation_time = request.get_creation_time()
        time_window = request.get_time_window()
        # * adjust the timewindow of the pickup
        pickup_earliest = time_window[0][0] - creation_time + self.current_timestep
        pickup_latest = time_window[0][1] - creation_time + self.current_timestep
        # * adjust the timewindow of the delivery
        delivery_earliest = time_window[1][0] - creation_time + self.current_timestep
        delivery_latest = time_window[1][1] - creation_time + self.current_timestep
        # * adjust the timewindow of the request
        time_window = (pickup_earliest, pickup_latest), (delivery_earliest, delivery_latest)
        request.set_time_windows(time_window)
        return

    def generate_requests(self, limited_by_locations = True ,max_new_requests = None):
        """_summary_

        Args:
            limited_by_locations (bool, optional): _description_. Defaults to True.
            max_new_requests (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # * get all locations not currently part of a request
        possible_locations = set()
        for location in self.locations:
            possible_locations.add(location)
        possible_locations.remove(self.get_depot())
        if limited_by_locations:
            for pair in self.current_requests:
                possible_locations.remove(pair.get_pickup_location())
                possible_locations.remove(pair.get_delivery_location())
        for v in self.vehicles:
            if v.next_location is not None and v.next_location != 0 and v.next_location in possible_locations: possible_locations.remove(v.next_location)

        if max_new_requests is None: max_new_requests = len(possible_locations) // 2

        # * generate the new pairs
        new_pairs = []
        num_new_pairs = 0
        possible_locations = list(possible_locations)
        r.shuffle(possible_locations)
        while len(possible_locations) >= 2 and num_new_pairs < max_new_requests:
            location_1 = possible_locations.pop()
            location_2 = possible_locations.pop()
            if r.randint(0,1) == 1:
                new_pairs.append([location_1, location_2])
            else:
                new_pairs.append([location_2, location_1])
            num_new_pairs += 1

        # * add new pairs to current requests
        new_requests = []
        for pair in new_pairs:
            request = Request(env=self, index=self.num_pairs, creation_time=self.current_timestep, pickup_location=pair[0], delivery_location=pair[1])
            self.add_request(request)
            new_requests.append(request)
            self.num_pairs += 1
        return new_requests

    def are_new_request_createable(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        num_locations_in_requests = len(self.current_requests) * 2
        if self.num_locations - num_locations_in_requests >= 2:
            return True
        else:
            return False

    def was_visited_by_vehicle(self, location):
        """_summary_

        Args:
            location (_type_): _description_

        Returns:
            _type_: _description_
        """
        for vehicle in self.vehicles:
            if location in vehicle.get_visited():
                return True
        return False
    
    def is_any_next_location(self, location):
        """_summary_

        Args:
            location (_type_): _description_

        Returns:
            _type_: _description_
        """
        for vehicle in self.vehicles:
            if location == vehicle.get_next_location():
                return True
        return False
        

    def all_vehicles_idle(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        for vehicle in self.vehicles:
            if vehicle.route or vehicle.distance_to_next != 0:
                return False
        return True

    def set_routes(self, routes):
        """_summary_

        Args:
            routes (_type_): _description_
        """
        # TODO: validate the routes
        for index, route in enumerate(routes):
            self.vehicles[index].set_route(route)

    def get_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        data = {"distance_matrix": self.distance_matrix, 
                "num_locations": self.num_locations, 
                "num_vehicles": self.num_vehicles,
                "depot": self.depot.get_index(), 
                "grid_size": self.grid_size, 
                "max_vehicle_distance": self.max_vehicle_distance,
                "max_vehicle_total_distance": self.max_vehicle_total_distance ,
                "max_vehicle_capacity": self.max_vehicle_capacity,
                "current_timestep": self.current_timestep,
                "horizon": self.horizon}

        pickups_deliveries = []
        if self.has_capacities: demands = [0] * self.num_locations
        if self.has_time_windows: time_windows = [(self.current_timestep, self.current_timestep + self.horizon)] * self.num_locations
        for request in self.current_requests:
            pair = [request.get_pickup_location().get_index(), request.get_delivery_location().get_index()]
            if self.has_capacities:
                demand = request.get_demand()
                demands[pair[0]] = demand
                demands[pair[1]] = -demand
            if self.has_time_windows:
                time_window = request.get_time_window()
                if self.was_visited_by_vehicle(request.get_pickup_location()) or self.is_any_next_location(request.get_delivery_location()):
                    print("hit")
                    time_windows[pair[0]] = (self.current_timestep, self.current_timestep + self.horizon)
                    time_windows[pair[1]] = (self.current_timestep, self.current_timestep + self.horizon)
                else:
                    time_windows[pair[0]] = time_window[0]
                    time_windows[pair[1]] = time_window[1]
            pickups_deliveries.append(pair)
        data["pickups_deliveries"] = pickups_deliveries
        if self.has_capacities: data["demands"] = demands
        if self.has_time_windows: data["time_windows"] = time_windows

        return data

    def get_locations_in_requests(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        location_in_requests = set()
        for request in self.current_requests:
            location_in_requests.add(request.get_pickup_location())
            location_in_requests.add(request.get_delivery_location())
        return location_in_requests

    def get_all_visited(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        all_visited = set()
        for vehicle in self.vehicles:
            all_visited.update(vehicle.get_visited())
        return all_visited
    
    def get_current_requests_indices(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        current_requests_indices = []
        for request in self.current_requests:
            current_requests_indices.append([request.get_pickup_location().get_index(),request.get_delivery_location().get_index()])
        return current_requests_indices

    def get_depot(self):
        return self.depot
    def get_distance_matrix(self):
        return self.distance_matrix
    def get_num_locations(self):
        return self.num_locations
    def get_vehicles(self):
        return self.vehicles
    def get_vehicles_dict(self):
        return self.vehicles_dict
    def get_num_vehicles(self):
        return self.num_vehicles
    def get_max_vehicle_capacity(self):
        return self.max_vehicle_capacity
    def get_current_requests(self):
        return self.current_requests
    def get_pickup_to_delivery(self):
        return self.pickup_to_delivery
    def get_location_to_requests(self):
        return self.location_to_request
    def get_locations_dict(self):
        return self.locations_dict
    def get_current_timestep(self):
        return self.current_timestep


    # * NetworkX Graph Utilities and Visualization
    def get_graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        graph = nx.Graph()
        for location in self.locations:
            graph.add_node(location.get_index(), pickups=self.get_pickups_at_location(location), deliveries=self.get_deliveries_at_location(location),
                            vehicles=self.get_vehicles_at_location(location))

        for i in range(self.num_locations):
            for j in range(self.num_locations):
                if i == j: continue
                graph.add_edge(i, j, label="connection" ,distance=self.distance_matrix[i][j])

        for vehicle in self.vehicles:
            driven_route = vehicle.get_driven_route_indices()
            if len(driven_route) >= 2:
                for i in range(len(driven_route) - 1):
                    graph.add_edge(driven_route[i], driven_route[i+1], label="route", vehicle=vehicle.get_index())


        self.graph = graph
        return graph

    def get_vehicles_at_location(self, location):
        """_summary_

        Args:
            location (_type_): _description_

        Returns:
            _type_: _description_
        """
        vehicles_indices = []
        for vehicle in self.vehicles:
            if vehicle.get_last_location() == location:
                vehicles_indices.append(vehicle.get_index())
        return vehicles_indices

    def get_pickups_at_location(self, location):
        """_summary_

        Args:
            location (_type_): _description_

        Returns:
            _type_: _description_
        """
        pickups_indices = []
        for request in self.current_requests:
            if request.get_pickup_location() == location:
                pickups_indices.append(request.get_index())
        return pickups_indices

    def get_deliveries_at_location(self, location):
        """_summary_

        Args:
            location (_type_): _description_

        Returns:
            _type_: _description_
        """
        deliveries_indices = []
        for request in self.current_requests:
            if request.get_delivery_location() == location:
                deliveries_indices.append(request.get_index())
        return deliveries_indices

    def visualize(self):
        """_summary_
        """
        graph = self.get_graph()
        pos = nx.spring_layout(graph, seed=42)

        connection_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("label") == "connection"]
        route_edges_by_vehicle = {}
        for u, v, d in graph.edges(data=True):
            if d.get("label") == "route":
                vehicle_idx = d.get("vehicle")
                route_edges_by_vehicle.setdefault(vehicle_idx, []).append((u, v))

        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500)

        nx.draw_networkx_edges(graph, pos, edgelist=connection_edges, edge_color='gray', alpha=0.2)

        cmap = cm.get_cmap("tab10")
        vehicle_indices = sorted(route_edges_by_vehicle.keys())
        for i, vehicle_idx in enumerate(vehicle_indices):
            color = cmap(i % 10)
            edges = route_edges_by_vehicle[vehicle_idx]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=[color], width=2.5, label=f"Vehicle {vehicle_idx}")

        labels = {}
        for node, data in graph.nodes(data=True):
            pickups = data.get("pickups", [])
            deliveries = data.get("deliveries", [])
            vehicles = data.get("vehicles", [])
            label = f"{node}\nP:{pickups}\nD:{deliveries}\nV:{vehicles}"
            labels[node] = label
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

        plt.title("PDP")
        plt.axis("off")
        plt.tight_layout()
        plt.legend()
        plt.show()

class Location:
    def __init__(self, env, index, coords, location_type = "Generic"):
        """_summary_

        Args:
            env (_type_): _description_
            index (_type_): _description_
            coords (_type_): _description_
            location_type (str, optional): _description_. Defaults to "Generic".
        """
        self.env = env
        self.index = index
        self.coords = coords
        self.location_type = location_type
        self.vehicles = set()

    def get_index(self):
        return self.index
    def get_coords(self):
        return self.coords
    def get_vehicles(self):
        return self.vehicles

class Request:
    def __init__(self, env, index, pickup_location, delivery_location, creation_time, demand=None, time_window=None, priority=None):
        """_summary_

        Args:
            env (_type_): _description_
            index (_type_): _description_
            pickup_location (_type_): _description_
            delivery_location (_type_): _description_
            creation_time (_type_): _description_
            demand (_type_, optional): _description_. Defaults to None.
            time_window (_type_, optional): _description_. Defaults to None.
            priority (_type_, optional): _description_. Defaults to None.
        """
        self.env = env
        self.index = index
        self.creation_time = creation_time
        self.pickup_location = pickup_location
        self.delivery_location = delivery_location
        self.demand = demand
        self.time_window = time_window
        self.priority = priority

    def set_demand(self, demand):
        self.demand = demand
    def set_time_windows(self, time_window):
        self.time_window = time_window

    def get_index(self):
        return self.index
    def get_creation_time(self):
        return self.creation_time
    def get_pickup_location(self):
        return self.pickup_location
    def get_delivery_location(self):
        return self.delivery_location
    def get_pair(self):
        return self.pickup_location, self.delivery_location
    def get_demand(self):
        return self.demand
    def get_time_window(self):
        return self.time_window


class Vehicle:
    def __init__(self, env, index, velocity = 1, capacity = None, vehicle_type = "Generic"):
        """_summary_

        Args:
            env (_type_): _description_
            index (_type_): _description_
            velocity (int, optional): _description_. Defaults to 1.
            capacity (_type_, optional): _description_. Defaults to None.
            vehicle_type (str, optional): _description_. Defaults to "Generic".
        """
        self.env = env
        self.index = index
        self.velocity = velocity
        self.capacity = capacity
        self.vehicle_type = vehicle_type

        self.last_location = self.env.get_depot()
        self.next_location = None
        self.distance_to_next = 0

        self.visited = set()
        self.driven_route = []
        self.driven_route.append(self.env.get_depot())
        self.route = []

    def step(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.distance_to_next > 0: self.distance_to_next -= self.velocity
        # * check if the vehicle arrived at a location, if so update location and distance variables
        if self.distance_to_next == 0 and self.last_location is not None and self.next_location is not None and self.last_location != self.next_location:
            # Make the previously next location the new last location as it's been reached
            self.last_location.vehicles.remove(self)
            self.last_location = self.next_location
            self.last_location.vehicles.add(self)
            self.driven_route.append(self.last_location)
            # Add the last location to the visited locations
            if self.last_location in self.env.get_locations_in_requests():
                self.visited.add(self.last_location)
            #
            self.next_location = self.route.pop(0) if self.route else self.env.get_depot()
            # calculate the new distance between the last and next location
            last_location_index = self.last_location.get_index()
            next_location_index = self.next_location.get_index()
            self.distance_to_next = self.env.distance_matrix[last_location_index][next_location_index]
            return True
        return False

    def set_route(self, route):
        """_summary_

        Args:
            route (_type_): _description_
        """
        if not route: return
        if route[0] == self.env.get_depot().get_index(): route.pop(0)
        locations = self.env.get_locations_dict()
        route_location_objects = []
        for index in route:
            location = locations[index]
            route_location_objects.append(location)
        self.route = route_location_objects
        self.next_location = self.route.pop(0)
        last_location_index = self.last_location.get_index()
        next_location_index = self.next_location.get_index()
        self.distance_to_next = self.env.distance_matrix[last_location_index][next_location_index]

    def get_index(self):
        return self.index
    def get_last_location(self):
        return self.last_location
    def get_next_location(self):
        return self.next_location
    def get_visited(self):
        return self.visited
    def get_visited_indices(self):
        visited = []
        for location in self.visited:
            visited.append(location.get_index())
        return visited
    def get_driven_route(self):
        return self.driven_route
    def get_route_indices(self):
        route = []
        for location in self.route:
            route.append(location.get_index())
        return route
    def get_driven_route_indices(self):
        driven_route = []
        for location in self.driven_route:
            driven_route.append(location.get_index())
        return driven_route
    def get_distance_to_next(self):
        return self.distance_to_next