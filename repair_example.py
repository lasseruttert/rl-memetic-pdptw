from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.solution_generators.random_solution import generate_random_solution
from memetic.repair.reinsertion_repair import reinsertion_repair

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    solution = generate_random_solution(problem)
    print(solution)
    print(solution.check_feasibility())  # Evaluate and print initial feasibility
    repaired_solution = reinsertion_repair(problem, solution)
    print(repaired_solution)
    print(repaired_solution.check_feasibility())  # Re-evaluate and print feasibility after repair