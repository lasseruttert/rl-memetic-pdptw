import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx

from cobra.experiments.pdp.deprecated.instance_creator import create_pdp_instance, manhattan_dist

class Environment:
    def __init__(self, grid_size = 200, num_locations = 20, num_pairs = 10, num_vehicles = 4, distance_function = manhattan_dist, demand = False, max_demand = 10, timewindows = False, seed = 1,initial_instance = None):
        self.grid_size = grid_size
        self.num_locations = num_locations
        self.num_pairs = num_pairs
        self.num_vehicles = num_vehicles
        self.vehicles = []
        self.location_mapping = None

        if initial_instance is None: initial_instance = create_pdp_instance(grid_size, num_locations, num_pairs, num_vehicles, distance_function, demand, max_demand, timewindows, seed)
        self.distance_matrix = initial_instance["distance_matrix"]
        self.requests = initial_instance["pickups_deliveries"].copy()
        self.all_requests = initial_instance["pickups_deliveries"].copy()
        self.locations = initial_instance["locations"]

        for i in range(self.num_vehicles):
            vehicle = Vehicle(self)
            self.vehicles.append(vehicle)

        self.requests_set = set()
        for request in self.requests:
            self.requests_set.add(request[0])
            self.requests_set.add(request[1])

    def step(self):
        update_needed = False
        for vehicle in self.vehicles:
            if vehicle.update_pos():
                update_needed = True
        if update_needed:
            self.update_requests()
            self.print_state()
        return

    def update_routes(self, routes):
        if routes is None:
            # TODO: add fallback option
            return
        # TODO add route validation
        for index, route in enumerate(routes):
            self.vehicles[index].update_route(route)

    def update_requests(self):
        for v in self.vehicles:
            for pair in self.requests:
                if pair[0] in v.visited and pair[1] in v.visited:
                    v.visited.remove(pair[0])
                    v.visited.remove(pair[1])
                    self.requests_set.remove(pair[0])
                    self.requests_set.remove(pair[1])
                    self.requests.remove(pair)

    def get_data(self):
        data = {"distance_matrix": self.distance_matrix, "pickups_deliveries": self.requests,
                "num_vehicles": self.num_vehicles, "depot": 0}
        return data

    def get_fixed_routes(self):
        fixed_routes = [[] for v in self.vehicles]
        for index, v in enumerate(self.vehicles):
            if v.next_location is not None:
                fixed_routes[index].append(0)
                for l in v.visited:
                    fixed_routes[index].append(l)
                fixed_routes[index].append(v.next_location)
        return fixed_routes

    def all_vehicles_done(self):
        return all(v.next_location == v.last_location for v in self.vehicles)

    def visited_by_vehicle(self, location):
        for v in self.vehicles:
            if location in v.visited:
                return True
        return False

    def get_open_pairs(self):
        possible_locations = set()
        for i in range(1, self.num_locations):
            possible_locations.add(i)
        for pair in self.requests:
            possible_locations.remove(pair[0])
            possible_locations.remove(pair[1])
        for v in self.vehicles:
            if v.next_location is not None and v.next_location != 0 and v.next_location in possible_locations: possible_locations.remove(v.next_location)
        new_pairs = []
        while len(possible_locations) >= 2:
            location_1 = possible_locations.pop()
            location_2 = possible_locations.pop()
            if self.visited_by_vehicle(location_1):
                new_pairs.append([location_1, location_2])
            else:
                new_pairs.append([location_2, location_1])
        return new_pairs

    def create_new_requests(self):
        open_pairs = self.get_open_pairs()
        if open_pairs:
            self.requests.extend(open_pairs)
            self.all_requests.extend(open_pairs)
            for request in open_pairs:
                self.requests_set.add(request[0])
                self.requests_set.add(request[1])
            return True
        return False

    def print_state(self):
        for i,v in enumerate(self.vehicles):
            if v.last_location is not None:
                print(f"Vehicle: {i} - Last Location: {v.last_location} - Next Location: {v.next_location} - Distance: {v.distance_to_next}")

    def print_final_state(self):
        print()
        print(f"All requests: {self.all_requests}")
        print(f"Current requests: {self.requests}")
        for i,v in enumerate(self.vehicles):
            print(f"Vehicle {i} Route: {v.driven_route}")

    def render_routes(self, ax=None):
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.clear()

        G = nx.complete_graph(self.num_locations)
        pos = {i: tuple(self.locations[i]) for i in range(self.num_locations)}

        nx.draw(G, pos, node_color='lightgray', node_size=300, alpha=0.5, with_labels=False, ax=ax)

        labels = {i: str(i) for i in range(self.num_locations)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black', ax=ax)

        colors = ['red', 'blue', 'green', 'orange']
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.driven_route:
                path = [0] + vehicle.driven_route + ([vehicle.next_location] if vehicle.next_location else [])
                nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=colors[i % len(colors)], ax=ax)
                nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])),
                                       edge_color=colors[i % len(colors)], ax=ax)

        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.001)


class Vehicle:
    def __init__(self, env):
        self.velocity = 1
        self.env = env
        self.last_location = None
        self.next_location = None
        self.distance_to_next = None
        self.route_to_go = None
        self.driven_route = []
        self.visited = set()

    def update_route(self, route):
        if self.route_to_go is None:
            self.route_to_go = route.copy()
        else:
            if route: route.remove(0)
            for index in route:
                if index in self.visited:
                    route.remove(index)
            self.route_to_go = route.copy()
        return

    def update_pos(self):
        if self.last_location is None and self.next_location is None:
            self.last_location = self.route_to_go.pop(0) if self.route_to_go else None
            self.next_location = self.route_to_go.pop(0) if self.route_to_go else None
            if self.last_location in self.env.requests_set: self.visited.add(self.last_location)
            self.distance_to_next = self.env.distance_matrix[self.last_location][self.next_location] if self.route_to_go else None
            if self.route_to_go: self.driven_route.append(self.last_location)
        else:
            if self.distance_to_next is not None and self.distance_to_next > 0:
                self.distance_to_next -= self.velocity
            if self.distance_to_next == 0:
                self.last_location = self.next_location
                self.next_location = self.route_to_go.pop(0) if self.route_to_go else 0
                if self.last_location in self.env.requests_set: self.visited.add(self.last_location)
                self.distance_to_next = self.env.distance_matrix[self.last_location][self.next_location]
                if self.route_to_go: self.driven_route.append(self.last_location)
                return True
        return False
