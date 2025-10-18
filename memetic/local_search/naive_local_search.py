from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.local_search.base_local_search import BaseLocalSearch

from memetic.fitness.fitness import fitness
from memetic.utils.compare import compare
import random
import numpy as np

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

class NaiveLocalSearch(BaseLocalSearch):
    """A simple local search framework that applies a list of operators iteratively until no improvement is found.
    """
    def __init__(self, operators: list = [], max_no_improvement: int = 3, max_iterations: int = 50, first_improvement: bool = True, random_operator_order: bool = False):
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
        self.first_improvement = first_improvement
        self.random_operator_order = random_operator_order
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution, deterministic_rng: bool = False, base_seed: int = 0) -> tuple[PDPTWSolution, float]:
        """Start the local search process.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance
            deterministic_rng (bool): If True, use deterministic seeding for reproducible operator applications
            base_seed (int): Base seed for deterministic RNG (only used if deterministic_rng=True)

        Returns:
            tuple[PDPTWSolution, float]: the best solution found and its fitness
        """
        no_improvement_count = 0
        iteration = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)

        while no_improvement_count < self.max_no_improvement and iteration < self.max_iterations:
            improved = False
            best_neighbor = None
            best_num_vehicles_neighbor = float('inf')
            best_fitness_neighbor = float('inf')
            if self.random_operator_order: random.shuffle(self.operators)
            for op_idx, operator in enumerate(self.operators):
                # Deterministic seeding
                if deterministic_rng:
                    op_seed = base_seed + iteration * 1000 + op_idx
                    random.seed(op_seed)
                    np.random.seed(op_seed)

                new_solution = operator.apply(problem, best_solution)
                operator.applications += 1
                new_fitness = fitness(problem, new_solution)
                if compare(new_fitness, new_solution.num_vehicles_used, best_fitness, best_solution.num_vehicles_used):
                    operator.improvements += 1
                    if self.first_improvement:
                        best_solution = new_solution
                        best_fitness = new_fitness
                        improved = True
                        break
                    else:
                        if compare(new_fitness, new_solution.num_vehicles_used, best_fitness_neighbor, best_num_vehicles_neighbor):
                            best_neighbor = new_solution
                            best_fitness_neighbor = new_fitness
                            best_num_vehicles_neighbor = new_solution.num_vehicles_used
            
            if not self.first_improvement and best_neighbor is not None:
                best_solution = best_neighbor
                best_fitness = best_fitness_neighbor
                improved = True
            
            no_improvement_count = 0 if improved else no_improvement_count + 1
            iteration += 1
        return best_solution, best_fitness