from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

class NoOpOperator(BaseOperator):
    def __init__(self):
        super().__init__()
        self.name = "NoOp"
        
    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return solution.clone()