from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

class SREXCrossover:
    def __init__(self):
        pass

    def crossover(self, problem: PDPTWProblem, parent1: PDPTWSolution, parent2: PDPTWSolution) -> PDPTWSolution:
        return parent1.clone() #TODO