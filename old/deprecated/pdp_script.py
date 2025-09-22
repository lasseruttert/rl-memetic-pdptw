from dataclasses import dataclass

from instance_creator import create_pdp_instance
from cobra.experiments.pdp.solvers.ortools_solver import pdp_solver


@dataclass
class Args:
    grid_size: int = 10000
    num_locations: int = 9
    num_pairs: int = 4
    num_vehicles: int = 10
    demand: bool = False
    max_demand: int = 50
    timewindows: bool = False

    seed: int = 2

if __name__ == "__main__":
    fixed_routes = None
    fixed_routes = [[0,8,5], [3,1], [7,2], [], [], [], [], [], [], []]
    solver = pdp_solver(grid_size=Args.grid_size, num_locations=Args.num_locations, num_vehicles=Args.num_vehicles, max_demand=Args.max_demand)
    current_instance = create_pdp_instance(grid_size=Args.grid_size, num_locations=Args.num_locations, num_pairs=Args.num_pairs, num_vehicles=Args.num_vehicles, demand=Args.demand, max_demand=Args.max_demand,timewindows=Args.timewindows, seed=Args.seed)
    #current_instance["pickups_deliveries"].append(current_instance["pickups_deliveries"][0])
    #current_instance = li_liam_to_or("data/benchmarks/pdp_100/lc101.txt")
    print(solver.solve_pdp(data=current_instance, optimize="distance", fixed_routes=fixed_routes))
