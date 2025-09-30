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
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator
import time

if __name__ == "__main__":
    operators = [
        ReinsertOperator(),
        ReinsertOperator(max_attempts=5,clustered=True),
        ReinsertOperator(force_same_vehicle=True),
        ReinsertOperator(allow_same_vehicle=False),
        ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
        
        RouteEliminationOperator(),
        
        SwapBetweenOperator(),
        
        TransferOperator(),
        TransferOperator(single_route=True),
        TransferOperator(max_attempts=5,single_route=True),
        
        CLSM1Operator(),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator()
    ]

    local_search = NaiveLocalSearch(operators, max_no_improvement=100, max_iterations=1000, first_improvement=False)
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc101.txt')
    start_time = time.time()    
    for i in range(100):
        print(f"--- Iteration {i}, Seconds {time.time()-start_time} ---")
        if i % 10 == 0:
            for operator in operators:
                print(f"{operator.__class__.__name__}: applied {operator.applications} times, improved {operator.improvements} times")
            print("-----------------------------------------------------------------------------------------------")
        initial_solution = generate_random_solution(problem)
        improved_solution, fitness = local_search.search(problem, initial_solution)
        