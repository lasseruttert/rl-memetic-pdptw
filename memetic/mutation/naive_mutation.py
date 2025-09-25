from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

class NaiveMutator:
    def __init__(self):
        pass
    
    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return solution 