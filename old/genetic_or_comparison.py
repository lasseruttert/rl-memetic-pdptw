from dataclasses import dataclass
from dynamic_nx.dynamic_pdp_nx_env import DynamicPDPEnv
from dynamic_nx.general_helper import fitness
from solvers.basic_genetic import Genetic_PDP_Solver
from dynamic_nx.ortools_helper import ORToolsHelper
from solvers.ortools_solver import OR_PDP_Solver
from solvers.ant_colony import Ant_PDP_Solver
from solvers.simulated_annealing import SA_PDP_Solver
@dataclass
class Args:
    grid_size: int = 100
    num_locations: int = 20
    num_pairs: int = 10
    num_vehicles: int = 10
    demand: bool = False
    max_demand: int = 50
    timewindows: bool = False

    seed: int = 2

if __name__ == "__main__":
    n = 25

    or_solutions = []
    genetic_solutions = []
    ant_solutions = []
    sa_solutions = []

    for i in range(n):
        env = DynamicPDPEnv(grid_size=Args.grid_size, num_locations=Args.num_locations, num_initial_pairs=Args.num_pairs, num_vehicles=Args.num_vehicles, has_capacities=False, max_demand=1, max_vehicle_capacity=1)
        or_helper = ORToolsHelper(env=env)
        or_solver = OR_PDP_Solver(max_seconds=1)
        genetic_solver = Genetic_PDP_Solver(env=env, fitness_fn=fitness)
        ant_solver = Ant_PDP_Solver(env=env, fitness_fn=fitness, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, Q=100)
        sa_solver = SA_PDP_Solver(env=env, fitness_fn=fitness)

        data = env.get_data()
        genetic_solution = genetic_solver.solve(data=data, population_size=200, num_iterations=50)
        vfitness = fitness(env, genetic_solution)
        genetic_solutions.append(vfitness)
        #print(f"Genetic Routes: {genetic_solution}")
        print(f"Genetic Fitness: {vfitness}")

        # ant_solution = ant_solver.solve(num_iterations=100)
        # vfitness = fitness(env, ant_solution)
        # ant_solutions.append(vfitness)
        # #print(f"Ant Routes: {ant_solution}")
        # print(f"Ant Fitness: {vfitness}")

        sa_solution = sa_solver.solve(num_iterations=100000, initial_temp=1500, min_temp=0.001, cooling_rate=0.8)
        vfitness = fitness(env, sa_solution)
        sa_solutions.append(vfitness)
        print(f"SA Routes: {sa_solution}")
        print(f"SA Fitness: {vfitness}")

        or_solution = or_solver.solve_pdp(data)
        vfitness = fitness(env, or_solution)
        or_solutions.append(vfitness)
        #print(f"OR Routes: {or_solution}")
        print(f"OR Fitness: {vfitness}")

        print()
        print()

    print(f"Genetic AVG: {sum(genetic_solutions)/len(genetic_solutions)}")
    print(f"OR AVG: {sum(or_solutions)/len(or_solutions)}")