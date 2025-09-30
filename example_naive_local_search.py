from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.solution_generators.random_solution import generate_random_solution

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.local_search.adaptive_local_search import AdaptiveLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator

if __name__ == "__main__":
    # Example usage
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc201.txt')
    initial_solution = generate_random_solution(problem)
    initial_solution_s = initial_solution.clone()
    operators = [
        ReinsertOperator(max_attempts=5,clustered=True),
        ReinsertOperator(allow_same_vehicle=False),
        ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        
        RouteEliminationOperator(),
        
        SwapBetweenOperator(),
        
        TransferOperator(single_route=True),
        
        CLSM1Operator(),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator()
    ]
    
    
    # local_search = NaiveLocalSearch(operators=operators, max_no_improvement=300, max_iterations=1000, first_improvement=False)
    local_search = AdaptiveLocalSearch(operators=operators, max_no_improvement=500, max_iterations=2000)
    improved_solution, fitness = local_search.search(problem, initial_solution)
    
    print("Initial Solution:", initial_solution_s)
    print("Improved Solution:", improved_solution)
    print("Feasible?:", improved_solution.check_feasibility())
    
    for operator in operators:
        print(f"{operator.__class__.__name__}: Applications={operator.applications}, Improvements={operator.improvements}")