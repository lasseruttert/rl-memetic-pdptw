from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.local_search.base_local_search import BaseLocalSearch
from memetic.fitness.fitness import fitness

class DummyLocalSearch(BaseLocalSearch):
    """Dummy local search that does nothing."""
    def __init__(self):
        super().__init__()

    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        f_value = fitness(problem, solution)
        return solution, f_value