import random as r
import numpy as np

class Ant_PDP_Solver:
    def __init__(self, env, fitness_fn, num_ants, alpha, beta, evaporation_rate, Q):
        self.env = env
        self.fitness_fn = fitness_fn
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q

    def is_feasible(self, route):
        location_mapping = {}
        for index, location in enumerate(route):
            location_mapping[location] = index
        visited = self.env.get_all_visited()
        for request in self.env.get_current_requests():
            pickup = request.get_pickup_location().get_index()
            delivery = request.get_delivery_location().get_index()
            if pickup not in location_mapping and delivery not in location_mapping: continue
            if pickup not in visited:
                if delivery in location_mapping and pickup not in location_mapping: return False
                if pickup in location_mapping and delivery not in location_mapping: continue
                else:
                    if location_mapping[pickup] >= location_mapping[delivery]: return False
        return True

    def compute_total_distance(self, route):
        vehicles = self.env.get_vehicles()
        solution = [[] for vehicle in vehicles]
        solution[0].extend(route)
        return self.fitness_fn(env=self.env, solution=solution)

    def generate_solutions(self, pheromone):
        solutions = []

        distance_matrix = self.env.get_distance_matrix()
        depot = self.env.get_depot().get_index()
        pickups = []
        deliveries = []
        visited = self.env.get_all_visited()
        for request in self.env.get_current_requests():
            pickup = request.get_pickup_location().get_index()
            delivery = request.get_delivery_location().get_index()
            if pickup not in visited: pickups.append(pickup)
            deliveries.append(delivery)
        for _ in range(self.num_ants):
            route = [depot]
            unvisited = set(pickups + deliveries)

            while unvisited:
                current = route[-1]
                feasible_next_nodes = [
                    n for n in unvisited
                    if self.is_feasible(route + [n])
                ]

                if not feasible_next_nodes:
                    break

                probabilities = []
                for n in feasible_next_nodes:
                    if current == n: continue
                    tau = pheromone[current][n] ** self.alpha
                    eta = (1 / distance_matrix[current][n]) ** self.beta
                    probabilities.append(tau * eta)

                next_node = r.choices(feasible_next_nodes, weights=probabilities)[0]
                route.append(next_node)
                unvisited.remove(next_node)

            #route.append(depot)
            solutions.append(route)
        return solutions

    def update_pheromones(self, pheromone, solutions):
        for i in range(len(pheromone)):
            for j in range(len(pheromone)):
                pheromone[i][j] *= (1 - self.evaporation_rate)

        for route in solutions:
            length = self.compute_total_distance(route)
            delta_pheromone = self.Q / length
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                pheromone[a][b] += delta_pheromone
                pheromone[b][a] += delta_pheromone

    def solve(self, num_iterations):
        best_solution = None
        best_length = float('inf')
        num_nodes = self.env.get_num_locations()

        pheromone = np.full((num_nodes, num_nodes), 1, dtype=float)
        np.fill_diagonal(pheromone, 0.0)

        for iteration in range(num_iterations):
            solutions = self.generate_solutions(pheromone)
            valid_solutions = [s for s in solutions if self.is_feasible(s)]

            for s in valid_solutions:
                length = self.compute_total_distance(s)
                if length < best_length:
                    best_solution = s
                    best_length = length

            self.update_pheromones(pheromone, valid_solutions)

        vehicles = self.env.get_vehicles()
        solution = [[] for vehicle in vehicles]
        solution[0].extend(best_solution)
        return solution
