from dataclasses import dataclass
from cobra.experiments.pdp.dynamic_nx.dynamic_pdp_nx_env import DynamicPDPEnv
from cobra.experiments.pdp.dynamic_nx.general_helper import fitness
from cobra.experiments.pdp.solvers.ant_colony import Ant_PDP_Solver

@dataclass
class Args:
    grid_size: int = 100
    num_locations: int = 9
    num_pairs: int = 4
    num_vehicles: int = 10
    demand: bool = False
    max_demand: int = 50
    timewindows: bool = False

    seed: int = 2

if __name__ == "__main__":
    env = DynamicPDPEnv(grid_size=Args.grid_size, num_locations=Args.num_locations, num_initial_pairs=Args.num_pairs, num_vehicles=Args.num_vehicles, has_capacities=False, max_demand=1, max_vehicle_capacity=1)
    solver = Ant_PDP_Solver(env=env, fitness_fn=fitness, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, Q=100)

    data = env.get_data()
    pairs = data["pickups_deliveries"]
    print(f"Pairs: {pairs}")
    routes = solver.solve(num_iterations=50)
    print(f"Routes: {routes}")
    fitness_value = fitness(env=env, solution=routes)
    print(f"Fitness pre-trimmed: {fitness_value}")
    if fitness_value == -1:
        input()
    print()
    env.set_routes(routes)
    # env.visualize()

    while True:

        if env.step():
            updated = True
            data = env.get_data()
            pairs = data["pickups_deliveries"]
            print(f"Pairs: {pairs}")
            for vehicle in env.vehicles:
                if vehicle.driven_route != [env.get_depot()]:
                    print(f"Vehicle: {vehicle.get_index()}")
                    print(f"Route: {vehicle.get_route_indices()}")
                    print(f"Visited: {vehicle.get_visited_indices()}")
                    print(f"Last Location: {vehicle.get_last_location().get_index()}")
                    print(f"Next Location: {vehicle.get_next_location().get_index()}")
                    print(f"Driven Route: {vehicle.get_driven_route_indices()}")
                    #print(f"Distance: {vehicle.distance_to_next}")
                    print()
            #env.visualize()


