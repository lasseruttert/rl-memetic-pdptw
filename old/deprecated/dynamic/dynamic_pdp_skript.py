from dataclasses import dataclass
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from dynamic_pdp_env import Environment
from cobra.experiments.pdp.solvers.ortools_solver import pdp_solver

@dataclass
class Args:
    grid_size: int = 100
    num_locations: int = 10
    num_pairs: int = 4
    num_vehicles: int = 4
    demand: bool = False
    max_demand: int = 50
    timewindows: bool = False

    total_timesteps = 10000

    seed: int = 5

if __name__ == "__main__":
    env = Environment(grid_size=Args.grid_size, num_locations=Args.num_locations, num_pairs=Args.num_pairs, num_vehicles=Args.num_vehicles, demand=Args.demand, max_demand=Args.max_demand,timewindows=Args.timewindows, seed=Args.seed)
    solver = pdp_solver(grid_size=Args.grid_size, num_locations=Args.num_locations, num_vehicles=Args.num_vehicles, max_demand=Args.max_demand)
    print(env.get_data())
    routes = solver.solve_pdp(data=env.get_data())
    env.update_routes(routes)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()

    timestep = 0
    while True:
        env.step()
        #env.render_routes(ax)
        if timestep % 100 == 0:
            if env.create_new_requests():
                data = env.get_data()
                for v in env.vehicles:
                    if v.driven_route: print(v.driven_route)
                fixed_routes = env.get_fixed_routes()
                print(f"Fixed: {fixed_routes}")
                routes = solver.solve_pdp(data=data, fixed_routes=fixed_routes)
                print(f"Routen: {routes}")
                env.update_routes(routes)

        timestep +=1
        if timestep >= Args.total_timesteps:
            env.print_final_state()
            print("Time ran out")
            break