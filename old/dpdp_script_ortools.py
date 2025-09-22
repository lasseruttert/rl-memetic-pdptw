from dataclasses import dataclass
from dynamic_nx.dynamic_pdp_nx_env import DynamicPDPEnv
from dynamic_nx.ortools_helper import ORToolsHelper
from dynamic_nx.general_helper import fitness
from solvers.ortools_solver import OR_PDP_Solver
from pprint import pprint
@dataclass
class Args:
    grid_size: int = 100
    num_locations: int = 9
    num_pairs: int = 4
    num_vehicles: int = 10
    demand: bool = False
    max_demand: int = 1
    max_vehicle_capacitiy: int = 1
    timewindows: bool = True
    horizon: int = 300

    seed: int = 2

if __name__ == "__main__":
    env = DynamicPDPEnv(grid_size=Args.grid_size, num_locations=Args.num_locations, num_initial_pairs=Args.num_pairs, num_vehicles=Args.num_vehicles, has_capacities=Args.demand, max_demand=Args.max_demand, max_vehicle_capacity=Args.max_vehicle_capacitiy, has_time_windows = Args.timewindows, horizon=Args.horizon)
    or_helper = ORToolsHelper(env=env)
    solver = OR_PDP_Solver(max_seconds=1)

    while True:
        if env.get_current_timestep() % 300 == 0:
            if env.are_new_request_createable():
                print("Generating new Requests...")
                for request in env.current_requests:
                    env.readjust_time_windows(request)
                env.generate_requests(limited_by_locations=True, max_new_requests=6)
            print(env)
            data = env.get_data()
            data = or_helper.normalize_time_windows(data)
            pprint(data)
            fixed_routes = or_helper.get_fixed_routes()
            print(f"Fixed routes: {fixed_routes}")
            pairs = data["pickups_deliveries"]
            print(f"Pairs: {pairs}")
            routes = solver.solve_pdp(data=data, fixed_routes=fixed_routes)
            print(f"Routes: {routes}")
            print(f"Fitness pre-trimmed: {fitness(env=env, solution=routes)}")
            trimmed_routes = or_helper.trim_routes(routes)
            print(f"Fitness trimmed: {fitness(env=env, solution=trimmed_routes)}")
            #print(f"Routes trimmed: {trimmed_routes}")
            print()
            env.set_routes(trimmed_routes)
            #env.visualize()

        if env.step():
            data = env.get_data()
            pairs = data["pickups_deliveries"]
            print(f"Pairs: {pairs}")
            for vehicle in env.vehicles:
                if vehicle.driven_route != [env.get_depot()]:
                    print(f"Vehicle: {vehicle.get_index()}")
                    print(f"Route: {vehicle.get_route_indices()}")
                    print(f"Last Location: {vehicle.get_last_location().get_index()}")
                    print(f"Next Location: {vehicle.get_next_location().get_index()}")
                    print(f"Driven Route: {vehicle.get_driven_route_indices()}")
                    print(f"Distance: {vehicle.distance_to_next}")
                    print()
            #env.visualize()


