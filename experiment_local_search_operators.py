from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader
from utils.instance_manager import InstanceManager

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
        
        FlipOperator(),
        FlipOperator(max_attempts=5),
        FlipOperator(single_route=True),
        
        SwapWithinOperator(),
        SwapWithinOperator(max_attempts=5),
        SwapWithinOperator(single_route=True),
        SwapWithinOperator(single_route=True, type="best"),
        SwapWithinOperator(single_route=False, type="best"),

        SwapBetweenOperator(),
        SwapBetweenOperator(type="best"),
        
        TransferOperator(),
        TransferOperator(single_route=True),
        TransferOperator(max_attempts=5,single_route=True),
        
        TwoOptOperator(),
        
        CLSM1Operator(),
        CLSM2Operator(),
        CLSM3Operator(),
        CLSM4Operator()
    ]

    instance_manager = InstanceManager()

    i = 0

    for problem in instance_manager.get_all(100):
        print(i)
        local_search = NaiveLocalSearch(operators, max_no_improvement=100, max_iterations=1000, first_improvement=False)
        problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc101.txt')
        start_time = time.time()
        for j in range(3):
            initial_solution = generate_random_solution(problem)
            improved_solution, fitness = local_search.search(problem, initial_solution)
        i += 1
        
    for operator in operators:
        print(f"{operator.__class__.__name__}: applied {operator.applications} times, improved {operator.improvements} times")
        
# ? ReinsertOperator: applied 25666 times, improved 2082 times
# ? ReinsertOperator: applied 25666 times, improved 4879 times
# ? ReinsertOperator: applied 25666 times, improved 981 times
# ? ReinsertOperator: applied 25666 times, improved 2088 times
# ? ReinsertOperator: applied 25666 times, improved 2188 times
# ? RouteEliminationOperator: applied 25666 times, improved 3179 times
# ? FlipOperator: applied 25666 times, improved 0 times
# ? FlipOperator: applied 25666 times, improved 0 times
# ? FlipOperator: applied 25666 times, improved 66 times
# ? SwapWithinOperator: applied 25666 times, improved 88 times
# ? SwapWithinOperator: applied 25666 times, improved 84 times
# ? SwapWithinOperator: applied 25666 times, improved 202 times
# ? SwapWithinOperator: applied 25666 times, improved 297 times
# ? SwapWithinOperator: applied 25666 times, improved 550 times
# ? SwapBetweenOperator: applied 25666 times, improved 201 times
# ? SwapBetweenOperator: applied 25666 times, improved 1057 times
# ? TransferOperator: applied 25666 times, improved 254 times
# ? TransferOperator: applied 25666 times, improved 150 times
# ? TransferOperator: applied 25666 times, improved 139 times
# ? TwoOptOperator: applied 25666 times, improved 845 times
# ? CLSM1Operator: applied 25666 times, improved 318 times
# ? CLSM2Operator: applied 25666 times, improved 2174 times
# ? CLSM3Operator: applied 25666 times, improved 836 times
# ? CLSM4Operator: applied 25666 times, improved 1298 times