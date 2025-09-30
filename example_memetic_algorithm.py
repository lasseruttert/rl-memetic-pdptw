from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.memetic_algorithm import MemeticSolver

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    memetic_algorithm = MemeticSolver()
    best_solution = memetic_algorithm.solve(problem)
    print("Best Solution:", best_solution)