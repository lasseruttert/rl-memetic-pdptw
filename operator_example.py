from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.solution_generators.random_solution import generate_random_solution
from memetic.repair.reinsertion_repair import reinsertion_repair
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator

if __name__ == "__main__":
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    solution = generate_random_solution(problem)
    print(solution)
    print(solution.check_feasibility())  # Evaluate and print initial feasibility
    # repaired_solution = reinsertion_repair(problem, solution)
    
    # print(repaired_solution)
    # print(repaired_solution.check_feasibility())  # Re-evaluate and print feasibility after repair
    operator = SwapBetweenOperator(problem)
    modified_solution = operator.apply(solution)
    print(modified_solution)
    print(modified_solution.check_feasibility())  # Check feasibility after applying the operator
    # repaired_solution = reinsertion_repair(problem, modified_solution)
    # print(repaired_solution)
    # print(repaired_solution.check_feasibility())  # Re-evaluate and print feasibility after repair