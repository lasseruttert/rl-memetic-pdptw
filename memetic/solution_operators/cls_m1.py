from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

from memetic.insertion.greedy_insertion import GreedyInsertion
from memetic.insertion.regret_insertion import Regret2Insertion
import random

# this is the operator M1 from the paper "Large Neighborhood Search with adaptive guided ejection search for the pickup and delivery problem with time windows"

class CLSM1Operator(BaseOperator):
    """Insert an un-assigned pickup and delivery (PD) pair into an existing route or create a new route for the PD pair"""
    def __init__(self, insertion_heuristic: str = 'greedy'):
        super().__init__()
        if insertion_heuristic == 'greedy':
            self.insertion_heuristic = GreedyInsertion()
        if insertion_heuristic == 'regret2':
            self.insertion_heuristic = Regret2Insertion()
            
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        unserved_requests = solution.get_unserved_requests(problem)
        if not unserved_requests:
            return solution
        request_to_insert = random.choice(unserved_requests)
        new_solution = solution.clone()
        new_solution = self.insertion_heuristic.insert(problem, new_solution, [request_to_insert])
        return new_solution