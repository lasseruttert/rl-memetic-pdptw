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
from memetic.solution_operators.shift import ShiftOperator
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

        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5),
        ShiftOperator(type="random", segment_length=2, max_shift_distance=4, max_attempts=5),
        ShiftOperator(type="random", segment_length=4, max_shift_distance=2, max_attempts=3),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=5, max_attempts=3),
        ShiftOperator(type="best", segment_length=2, max_shift_distance=3),
        ShiftOperator(type="best", segment_length=3, max_shift_distance=2),
        ShiftOperator(type="random", segment_length=3, max_shift_distance=3, max_attempts=5, single_route=True),

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
        
# ? ReinsertOperator: applied 26148 times, improved 2158 times
# ? ReinsertOperator: applied 26148 times, improved 5016 times
# ? ReinsertOperator: applied 26148 times, improved 1006 times
# ? ReinsertOperator: applied 26148 times, improved 2137 times
# ? ReinsertOperator: applied 26148 times, improved 2155 times
# ? RouteEliminationOperator: applied 26148 times, improved 3284 times
# ? FlipOperator: applied 26148 times, improved 0 times
# ? FlipOperator: applied 26148 times, improved 0 times
# ? FlipOperator: applied 26148 times, improved 61 times
# ? SwapWithinOperator: applied 26148 times, improved 102 times
# ? SwapWithinOperator: applied 26148 times, improved 88 times
# ? SwapWithinOperator: applied 26148 times, improved 202 times
# ? SwapWithinOperator: applied 26148 times, improved 277 times
# ? SwapWithinOperator: applied 26148 times, improved 541 times
# ? SwapBetweenOperator: applied 26148 times, improved 221 times
# ? SwapBetweenOperator: applied 26148 times, improved 1041 times
# ? TransferOperator: applied 26148 times, improved 270 times
# ? TransferOperator: applied 26148 times, improved 165 times
# ? TransferOperator: applied 26148 times, improved 145 times
# ? ShiftOperator: applied 26148 times, improved 196 times
# ? ShiftOperator: applied 26148 times, improved 138 times
# ? ShiftOperator: applied 26148 times, improved 209 times
# ? ShiftOperator: applied 26148 times, improved 219 times
# ? ShiftOperator: applied 26148 times, improved 147 times
# ? ShiftOperator: applied 26148 times, improved 234 times
# ? ShiftOperator: applied 26148 times, improved 58 times
# ? TwoOptOperator: applied 26148 times, improved 731 times
# ? CLSM1Operator: applied 26148 times, improved 357 times
# ? CLSM2Operator: applied 26148 times, improved 2143 times
# ? CLSM3Operator: applied 26148 times, improved 861 times
# ? CLSM4Operator: applied 26148 times, improved 1404 times