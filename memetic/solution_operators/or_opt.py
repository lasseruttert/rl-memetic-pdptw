from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
from memetic.fitness.fitness import fitness
import random
class OrOptOperator(BaseOperator):
    def __init__(self,max_attempts: int = 10, single_route: bool = False, segment_length: int = 3):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        self.segment_length = segment_length
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return # TODO Implement Or-opt operator