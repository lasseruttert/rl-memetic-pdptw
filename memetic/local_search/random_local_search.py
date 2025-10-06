from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.local_search.base_local_search import BaseLocalSearch

from memetic.fitness.fitness import fitness
import random

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

class RandomLocalSearch(BaseLocalSearch):
    """A simple local search framework that applies a random operator in each iteration. (random walk)
    """
    def __init__(self, operators: list = [], max_no_improvement: int = 3, max_iterations: int = 50):
        """
        Args:
            operators (list, optional): List of local search operators to apply. Defaults to [].
            max_no_improvement (int, optional): Maximum number of consecutive iterations without improvement before stopping. Defaults to 3.
            max_iterations (int, optional): Maximum total iterations before stopping. Defaults to 50.
            first_improvement (bool, optional): Whether to accept the first improving move found (True) or search all operators for the best move (False). Defaults to True.
            random_operator_order (bool, optional): Whether to randomize the order of operators in each iteration. Defaults to False.
        """
        super().__init__()
        if not operators:
            operators = [
                ReinsertOperator(max_attempts=5,clustered=True),
                ReinsertOperator(allow_same_vehicle=False),
                ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
                
                RouteEliminationOperator()
            ]
        
        self.operators = operators
        self.max_no_improvement = max_no_improvement
        self.max_iterations = max_iterations
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> tuple[PDPTWSolution, float]:
        """Start the local search process.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            tuple[PDPTWSolution, float]: the best solution found and its fitness
        """
        no_improvement_count = 0
        iteration = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)
        
        while no_improvement_count < self.max_no_improvement and iteration < self.max_iterations:
            improved = False
            operator = random.choice(self.operators)
            new_solution = operator.apply(problem, best_solution)
            operator.applications += 1
            new_fitness = fitness(problem, new_solution)
            if new_fitness < best_fitness:
                operator.improvements += 1
                best_solution = new_solution
                best_fitness = new_fitness
                improved = True
            
            no_improvement_count = 0 if improved else no_improvement_count + 1
            iteration += 1
        return best_solution, best_fitness