import random as r
import copy
from dynamic_nx.general_helper import fitness, encode_solution, similarity, repair, is_feasible, is_viable

def fisher_yates_shuffle(lst):
    for i in range(len(lst) - 1, 0, -1):
        j = r.randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

class Genetic_PDP_Solver:
    def __init__(self, env, fitness_fn):
        self.env = env
        self.fitness_fn = fitness_fn
        self.current_data = None
        self.num_vehicles = None
        self.current_requests = None
        self.current_pickups = None
        self.current_deliveries = None
        self.pickup_to_delivery = {}
        self.delivery_to_pickup = {}
        self.fitness_cache = {}

    def generate_population(self, population_size):
        population = []
        for i in range(population_size):
            population.append((self.generate_individual(), None))
        population = self.calc_fitness_for_population(population)
        population = sorted(population, key=lambda x: x[1])
        return population

    def generate_individual(self):
        individual = [[] for i in range(self.num_vehicles)]
        requests = self.current_requests.copy()
        for vehicle_index, vehicle in enumerate(self.env.get_vehicles()):
            for pickup in vehicle.get_visited_indices():
                delivery = self.pickup_to_delivery[pickup]
                if delivery not in individual[vehicle_index]:
                    individual[vehicle_index].append(delivery)
                    requests.remove([pickup, delivery])

        r.shuffle(requests)
        for pickup, delivery in requests:
            vehicle_index = r.randint(0, self.num_vehicles-1)
            if r.randint(1,2) == 1:
                for index, vehicle in enumerate(self.env.get_vehicles()):
                    if individual[index]: vehicle_index = index
                    break
            route = individual[vehicle_index]

            pickup_insert_index = r.randint(0, len(route))
            delivery_insert_index = r.randint(pickup_insert_index+1, len(route)+1)

            route.insert(delivery_insert_index, delivery)
            route.insert(pickup_insert_index, pickup)
        for route in individual:
            if route: route.insert(0, self.env.get_depot().get_index())

        return individual

    def calc_fitness_for_population(self, population):
        MISSING = object()
        
        for index, pair in enumerate(population):
            individual = pair[0]
            encoding = encode_solution(individual)
            
            fitness = self.fitness_cache.get(encoding, MISSING)
            
            if fitness is MISSING:
                fitness = self.fitness(individual)
                self.fitness_cache[encoding] = fitness
            
            population[index] = [individual, fitness]
        
        return population

    def fitness(self, individual):
        return fitness(env=self.env, solution=individual)
    
    def build_position_map(self, individual):
        position_map = {}
        for vehicle_idx, route in enumerate(individual):
            for pos, loc in enumerate(route):
                position_map[loc] = (vehicle_idx, pos)
        return position_map

    def mutate(self, individual, mutation = None):
        position_map = self.build_position_map(individual)
        vehicles = self.env.get_vehicles()

        individual = copy.deepcopy(individual)
        if mutation is None:
            rand = r.randint(1, 3)
            mutation = {1: "transfer", 2: "shuffle", 3: "swap"}[rand]

        if mutation == "swap":
            request1, request2 = None, None
            while request1 == request2 and len(self.current_requests) > 1:
                request1 = self.current_requests[r.randint(0, len(self.current_requests) - 1)]
                request2 = self.current_requests[r.randint(0, len(self.current_requests) - 1)]

            r1p = position_map.get(request1[0])
            r1d = position_map.get(request1[1])
            r2p = position_map.get(request2[0])
            r2d = position_map.get(request2[1])

            if None in (r1p, r1d, r2p, r2d):
                return individual

            individual[r1p[0]][r1p[1]], individual[r2p[0]][r2p[1]] = individual[r2p[0]][r2p[1]], individual[r1p[0]][r1p[1]]
            individual[r1d[0]][r1d[1]], individual[r2d[0]][r2d[1]] = individual[r2d[0]][r2d[1]], individual[r1d[0]][r1d[1]]

        if mutation == "transfer":
            request = self.current_requests[r.randint(0, len(self.current_requests) - 1)]
            rp = position_map.get(request[0])
            rd = position_map.get(request[1])

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
                for index, vehicle in enumerate(vehicles):
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
            route = fisher_yates_shuffle(route)

            seen = set()
            location_positions = {}
            for idx, loc in enumerate(route):
                was_visited = False
                for vehicle in vehicles:
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

    def crossover(self, parent1, parent2):
        parent1_mapping = {}
        parent2_mapping = {}
        for vehicle_index, route in enumerate(parent1):
            for location in parent1[vehicle_index]: parent1_mapping[location] = vehicle_index
            for location in parent2[vehicle_index]: parent2_mapping[location] = vehicle_index

        child = [[] for i in range(self.num_vehicles)]
        requests = self.current_requests.copy()
        r.shuffle(requests)
        for pickup, delivery in requests:
            index1, index2, index = None, None, None
            vehicle_index = None
            if r.randint(0,1) == 1 and pickup in parent1_mapping:
                if not child[parent1_mapping[pickup]]:
                    child[parent1_mapping[pickup]].extend([pickup,delivery])
                    continue
                index1, index2 = sorted(r.sample(range(len(child[parent1_mapping[pickup]]) + 1), 2))
                vehicle_index = parent1_mapping[pickup]

            elif pickup in parent2_mapping:
                if not child[parent2_mapping[pickup]]:
                    child[parent2_mapping[pickup]].extend([pickup,delivery])
                    continue
                index1, index2 = sorted(r.sample(range(len(child[parent2_mapping[pickup]]) + 1), 2))
                vehicle_index = parent2_mapping[pickup]

            elif pickup not in parent1_mapping and pickup not in parent2_mapping:
                for v_index, vehicle in enumerate(self.env.get_vehicles()):
                    if pickup in vehicle.get_visited_indices():
                        index = r.randint(0, len(child[v_index]))
                        vehicle_index = v_index


            if index1 is not None and index2 is not None and index is None:
                if index1 > index2: index1, index2 = index2, index1
                child[vehicle_index].insert(index2, delivery)
                child[vehicle_index].insert(index1, pickup)
            if index is not None and index1 is None and index2 is None:
                child[vehicle_index].insert(index, delivery)

        for route in child:
            if route: route.insert(0, self.env.get_depot().get_index())

        return child

    def local_search(self, individual):
        MISSING = object()
        no_improvement = 0
        encoding = encode_solution(individual)
        best_fitness = self.fitness_cache.get(encoding, MISSING)
        
        if best_fitness is MISSING:
            best_fitness = self.fitness(individual)
            self.fitness_cache[encoding] = best_fitness
        
        best_solution = individual

        for i in range(10):
            neighbor = self.mutate(best_solution, mutation="transfer")
            neighbor_encoding = encode_solution(neighbor)
            
            fitness = self.fitness_cache.get(neighbor_encoding, MISSING)
            
            if fitness is MISSING:
                fitness = self.fitness(neighbor)
                self.fitness_cache[neighbor_encoding] = fitness
            
            if fitness < best_fitness:
                best_solution = neighbor
                best_fitness = fitness
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement > 2:
                break

        return best_solution

    def repair(self, solution):
        return repair(self.env, solution)

    def solve(self, data, population_size, num_iterations):
        self.current_data = data
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

        population = self.generate_population(population_size)

        for i in range(num_iterations):
            elite_size = len(population) // 4
            elites = population[:elite_size]
            for index, (individual, fitness) in enumerate(elites):
                elites[index] = (individual, None)

            mutated = []
            while len(mutated) < len(population) // 2:
                # index1, index2 = r.sample(range(len(population)), 2)
                # new_individual = self.crossover(population[index1][0], population[index2][0])
                new_individual = population[r.randint(0, len(population) -1)][0]
                new_individual = self.mutate(new_individual)
                new_individual = self.local_search(new_individual)
                mutated.append((new_individual, None))

            new_individuals = []
            while len(new_individuals) < len(population) // 4:
                new_individuals.append((self.generate_individual(), None))

            population = elites + mutated + new_individuals
            population = self.calc_fitness_for_population(population)
            population = sorted(population, key=lambda x: x[1])

        return population[0][0]

# TODO:
# * calc_similarity function to keep the population diverse
# * add probabilities for calling of mutation and local search
# * add hashing for individuals and cache to store fitness values
# * add some way to find best vehicle to insert something into
