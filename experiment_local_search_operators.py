from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader
from utils.li_lim_instance_manager import LiLimInstanceManager

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.solution_generators.random_solution import generate_random_solution

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.merge import MergeOperator
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
from memetic.solution_operators.request_shift_within import RequestShiftWithinOperator
from memetic.solution_operators.node_swap_within import NodeSwapWithinOperator

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
        
        MergeOperator(type="random", num_routes=2),
        MergeOperator(type="random", num_routes=2, reorder=False),
        
        MergeOperator(type="min", num_routes=2),
        MergeOperator(type="min", num_routes=2, reorder=False),
        
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
        CLSM4Operator(),
        
        RequestShiftWithinOperator(),
        NodeSwapWithinOperator(check_precedence=True),
        NodeSwapWithinOperator(check_precedence=False),
    ]

    instance_manager = LiLimInstanceManager()

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
        print(f"{operator.name}: applied {operator.applications} times, improved {operator.improvements} times")
        
#? Reinsert-nC-Max1-NewV-SameV-nF_SameV: applied 25947 times, improved 2117 times
#? Reinsert-C-Max5-NewV-SameV-nF_SameV: applied 25947 times, improved 4964 times     
#? Reinsert-nC-Max1-NewV-SameV-F_SameV: applied 25947 times, improved 953 times      
#? Reinsert-nC-Max1-NewV-NoSameV-nF_SameV: applied 25947 times, improved 2123 times  
#? Reinsert-nC-Max1-NoNewV-NoSameV-nF_SameV: applied 25947 times, improved 2120 times
#? RouteElimination: applied 25947 times, improved 3045 times
#? Flip-All-Unlimited: applied 25947 times, improved 0 times
#? Flip-All-Max5: applied 25947 times, improved 0 times
#? Flip-Single-Unlimited: applied 25947 times, improved 57 times
#? Merge-random-N2-F-R: applied 25947 times, improved 472 times
#? Merge-random-N2-F-nR: applied 25947 times, improved 453 times
#? Merge-min-N2-F-R: applied 25947 times, improved 765 times
#? Merge-min-N2-F-nR: applied 25947 times, improved 765 times
#? SwapWithin-random-A-MaxNone: applied 25947 times, improved 98 times
#? SwapWithin-random-A-Max5: applied 25947 times, improved 99 times
#? SwapWithin-random-S-MaxNone: applied 25947 times, improved 213 times
#? SwapWithin-best-S-MaxNone: applied 25947 times, improved 310 times
#? SwapWithin-best-A-MaxNone: applied 25947 times, improved 573 times
#? SwapBetween-random: applied 25947 times, improved 216 times
#? SwapBetween-best: applied 25947 times, improved 1036 times
#? Transfer-A-Max1: applied 25947 times, improved 250 times
#? Transfer-S-Max1: applied 25947 times, improved 185 times
#? Transfer-S-Max5: applied 25947 times, improved 149 times
#? Shift-random-A-Max5-Seg3-Dist3: applied 25947 times, improved 2 times
#? Shift-random-A-Max5-Seg2-Dist4: applied 25947 times, improved 14 times
#? Shift-random-A-Max3-Seg4-Dist2: applied 25947 times, improved 34 times
#? Shift-random-A-Max3-Seg3-Dist5: applied 25947 times, improved 12 times
#? Shift-best-A-Max5-Seg2-Dist3: applied 25947 times, improved 154 times
#? Shift-best-A-Max5-Seg3-Dist2: applied 25947 times, improved 216 times
#? Shift-random-S-Max5-Seg3-Dist3: applied 25947 times, improved 81 times
#? TwoOpt: applied 25947 times, improved 713 times
#? CLS-M1: applied 25947 times, improved 251 times
#? CLS-M2: applied 25947 times, improved 2069 times
#? CLS-M3: applied 25947 times, improved 836 times
#? CLS-M4: applied 25947 times, improved 1351 times
#? RequestShiftWithin: applied 26499 times, improved 100 times
#? NodeSwapWithin-WithPrecedence: applied 26499 times, improved 36 times
#? NodeSwapWithin: applied 26499 times, improved 31 times