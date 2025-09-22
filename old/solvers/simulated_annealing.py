import random as r
import math, copy


class SA_PDP_Solver:
    def __init__(self, env, fitness_fn):
        self.env = env
        self.fitness_fn = fitness_fn

        self.current_data = None
        self.current_requests = None
        self.current_pickups = None
        self.current_deliveries = None
        self.pickup_to_delivery = None
        self.delivery_to_pickup = None
        self.vehicles = None
        self.num_vehicles = None

    def update_variables(self):
        self.current_data = self.env.get_data()
        self.num_vehicles = self.current_data.get("num_vehicles", 1)
        self.current_requests = self.current_data.get("pickups_deliveries", []).copy()
        self.current_pickups = set()
        self.current_deliveries = set()
        self.pickup_to_delivery = {}
        self.delivery_to_pickup = {}
        for pickup, delivery in self.current_requests:
            self.current_pickups.add(pickup)
            self.current_deliveries.add(delivery)
            self.pickup_to_delivery[pickup] = delivery
            self.delivery_to_pickup[delivery] = pickup


    def generate_solution(self):
        solution = [[] for i in range(self.num_vehicles)]
        requests = self.current_requests.copy()
        for vehicle_index, vehicle in enumerate(self.env.get_vehicles()):
            for pickup in vehicle.get_visited():
                for request in requests:
                    if request[0] == pickup.get_index():
                        solution[vehicle_index].append(request[1])
                        requests.remove(request)

        r.shuffle(requests)
        for pickup, delivery in requests:
            vehicle_index = r.randint(0, self.num_vehicles-1)
            if r.randint(1,2) == 1:
                for index, vehicle in enumerate(self.env.get_vehicles()):
                    if solution[index]: vehicle_index = index
                    break
            route = solution[vehicle_index]

            pickup_insert_index = r.randint(0, len(route))
            delivery_insert_index = r.randint(pickup_insert_index+1, len(route)+1)

            route.insert(delivery_insert_index, delivery)
            route.insert(pickup_insert_index, pickup)
        for route in solution:
            if route: route.insert(0, self.env.get_depot().get_index())

        return solution

    def fitness(self, individual):
        return self.fitness_fn(env=self.env, solution=individual)

    def mutate(self, individual, mutation = None):
        def find_position(node):
            for vehicle_index, route in enumerate(individual):
                for pos, loc in enumerate(route):
                    if loc == node:
                        return (vehicle_index, pos)
            return None

        individual = copy.deepcopy(individual)
        if mutation is None:
            rand = r.randint(1, 3)
            mutation = {1: "transfer", 2: "shuffle", 3: "swap"}[rand]

        if mutation == "swap":
            request1, request2 = None, None
            while request1 == request2 and len(self.current_requests) > 1:
                request1 = self.current_requests[r.randint(0, len(self.current_requests) - 1)]
                request2 = self.current_requests[r.randint(0, len(self.current_requests) - 1)]

            r1p = find_position(request1[0])
            r1d = find_position(request1[1])
            r2p = find_position(request2[0])
            r2d = find_position(request2[1])

            if None in (r1p, r1d, r2p, r2d):
                return individual

            individual[r1p[0]][r1p[1]], individual[r2p[0]][r2p[1]] = individual[r2p[0]][r2p[1]], individual[r1p[0]][r1p[1]]
            individual[r1d[0]][r1d[1]], individual[r2d[0]][r2d[1]] = individual[r2d[0]][r2d[1]], individual[r1d[0]][r1d[1]]

        if mutation == "transfer":
            request = self.current_requests[r.randint(0, len(self.current_requests) - 1)]
            rp = find_position(request[0])
            rd = find_position(request[1])

            if None in (rp, rd):
                return individual

            if rp[0] == rd[0] and rp[1] < rd[1]:
                individual[rp[0]].pop(rd[1])
                individual[rd[0]].pop(rp[1])
            else:
                individual[rp[0]].pop(rp[1])
                individual[rd[0]].pop(rd[1])

            if len(individual[rp[0]]) == 1: individual[rp[0]].pop(0)

            vehicle_index = r.randint(0, self.num_vehicles - 1)
            if r.randint(1,2) == 1:
                for index, vehicle in enumerate(self.env.get_vehicles()):
                    if individual[index]: vehicle_index = index
                    break
            depot = self.env.get_depot().get_index()
            if individual[vehicle_index]: individual[vehicle_index].remove(depot)
            if len(individual[vehicle_index]) < 2:
                individual[vehicle_index].extend(request)
            else:
                index1, index2 = sorted(r.sample(range(len(individual[vehicle_index]) + 1), 2))
                individual[vehicle_index].insert(index2, request[1])
                individual[vehicle_index].insert(index1, request[0])
            individual[vehicle_index].insert(0, depot)

        if mutation == "shuffle":
            vehicle_index = r.randint(0, self.num_vehicles - 1)
            route = individual[vehicle_index]

            if not route:
                return individual
            r.shuffle(route)

            seen = set()
            location_positions = {}
            for idx, loc in enumerate(route):
                was_visited = False
                for vehicle in self.env.get_vehicles():
                    if loc in vehicle.get_visited_indices(): was_visited = True
                if was_visited: continue
                if loc in self.pickup_to_delivery:
                    delivery = self.pickup_to_delivery[loc]
                    if delivery in seen:
                        delivery_idx = location_positions[delivery]
                        route[idx], route[delivery_idx] = route[delivery_idx], route[idx]
                        location_positions[loc], location_positions[delivery] = delivery_idx, idx

                seen.add(loc)
                location_positions[loc] = idx

            depot = self.env.get_depot().get_index()
            if depot in route:
                route.remove(depot)
            route.insert(0, depot)

            individual[vehicle_index] = route

        return individual

    def solve(self, num_iterations, initial_temp, min_temp, cooling_rate):
        self.update_variables()
        current_solution = self.generate_solution()
        best_solution = copy.deepcopy(current_solution)
        current_fitness = self.fitness(current_solution)
        best_fitness = current_fitness
        temperature = initial_temp
        iteration = 0

        while temperature > min_temp and iteration < num_iterations:
            neighbor_solution = self.mutate(current_solution)
            neighbor_fitness = self.fitness(neighbor_solution)

            delta_fitness = neighbor_fitness - current_fitness

            if delta_fitness < 0:
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness:
                    best_solution = copy.deepcopy(current_solution)
                    best_fitness = current_fitness
            else:
                acceptance_prob = math.exp(-delta_fitness / temperature)
                if r.random() < acceptance_prob:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
            temperature *= cooling_rate

            iteration += 1

        return best_solution