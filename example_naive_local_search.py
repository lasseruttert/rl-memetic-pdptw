from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.solution_generators.random_solution import generate_random_solution
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator

if __name__ == "__main__":
    # Example usage
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    initial_solution = generate_random_solution(problem)
    initial_solution_s = initial_solution.clone()
    operators = [
        ReinsertOperator(problem),
        FlipOperator(problem, single_route=True),
        SwapWithinOperator(problem, single_route=True),
        SwapBetweenOperator(problem),
        TransferOperator(problem, single_route=True)
    ]
    
    
    local_search = NaiveLocalSearch(operators=operators, max_no_improvement=10)
    improved_solution = local_search.search(problem, initial_solution)
    
    print("Initial Solution:", initial_solution_s)
    print("Improved Solution:", improved_solution)
    print("Feasible?:", improved_solution.check_feasibility())