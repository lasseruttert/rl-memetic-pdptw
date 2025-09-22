import random as r

from dynamic_nx.dynamic_pdp_nx_env import DynamicPDPEnv
from dynamic_nx.general_helper import is_feasible, is_viable, simple_insertion, repair

def generate_individual(env):
    vehicles = env.get_vehicles()
    num_vehicles = len(vehicles)
    requests = env.get_current_requests_indices().copy()
    pickup_to_delivery = env.get_pickup_to_delivery()
    individual = [[] for i in range(num_vehicles)]
    for vehicle_index, vehicle in enumerate(vehicles):
        for pickup in vehicle.get_visited_indices():
            delivery = pickup_to_delivery[pickup]
            if delivery not in individual[vehicle_index]:
                individual[vehicle_index].append(delivery)
                requests.remove([pickup, delivery])
    r.shuffle(requests)
    for pickup, delivery in requests:
        vehicle_index = r.randint(0, num_vehicles-1)
        if r.randint(1,2) == 1:
            for index, vehicle in enumerate(vehicles):
                if individual[index]: vehicle_index = index
                break
        route = individual[vehicle_index]
        pickup_insert_index = r.randint(0, len(route))
        delivery_insert_index = r.randint(pickup_insert_index+1, len(route)+1)
        if r.randint(1,10) != 1:
            route.insert(delivery_insert_index, delivery)
            route.insert(pickup_insert_index, pickup)
    for route in individual:
        if route: route.insert(0, env.get_depot().get_index())
    return individual

if __name__ == "__main__":
    
    for i in range(10):
        env = DynamicPDPEnv(grid_size=100, num_locations=20, num_initial_pairs=9, num_vehicles=10)
        solution = generate_individual(env)
        solution = repair(env, solution)
        if not is_feasible(env, solution): print(i, solution)
        if not is_viable(env, solution): print(i, solution)
    print("Done")