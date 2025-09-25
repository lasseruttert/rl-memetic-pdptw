from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

class NaiveLocalSearch:
    def __init__(self):
        pass
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        return solution