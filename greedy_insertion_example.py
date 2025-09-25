from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.feasibility import is_feasible
from utils.li_lim_reader import li_lim_reader
from memetic.insertion.insertion_heuristic import greedy_insertion
from memetic.fitness.fitness import fitness

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc102.txt')
    initial_solution = PDPTWSolution(problem, routes=[[] for _ in range(problem.num_vehicles)])
    improved_solution = greedy_insertion(problem, initial_solution)
    print(is_feasible(problem, improved_solution, use_prints=True))
    print(improved_solution)
    print(fitness(problem=problem, solution=improved_solution))