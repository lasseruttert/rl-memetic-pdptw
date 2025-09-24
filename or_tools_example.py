from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader
from utils.feasibility import is_feasible
from or_tools.or_tools import ORToolsSolver

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    solver = ORToolsSolver(max_seconds=10, use_span=False, minimize_num_vehicles=True)
    solution = solver.solve(problem)
    print(is_feasible(problem, solution, use_prints=True))
    print(solution)