from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.memetic_algorithm import MemeticSolver

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc203.txt')
    print(problem)
    memetic_algorithm = MemeticSolver(max_no_improvement=3, verbose=True)
    best_solution = memetic_algorithm.solve(problem)
    print(best_solution)