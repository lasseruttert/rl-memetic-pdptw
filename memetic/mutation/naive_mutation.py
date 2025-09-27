from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
import random

class NaiveMutation:
    def __init__(self, operators: list = [], max_iterations: int = 5):
        self.operators = operators
        self.max_iterations = max_iterations
    
    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        for _ in range(self.max_iterations):
            operator = random.choice(self.operators)
            new_solution = operator.apply(problem, new_solution)
        return new_solution