from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader
from utils.mendeley_reader import mendeley_reader

from memetic.memetic_algorithm import MemeticSolver

if __name__ == "__main__":
    # problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc109.txt')
    problem = mendeley_reader('G:/Meine Ablage/rl-memetic-pdptw/data/n100/bar-n100-1.txt')
    print(problem)
    memetic_algorithm = MemeticSolver(max_time_seconds=30, verbose=True)
    best_solution = memetic_algorithm.solve(problem)
    print(best_solution)
    best_solution.save_solution_txt('results/solution_txts')