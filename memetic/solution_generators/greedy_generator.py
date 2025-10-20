from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.solution_generators.base_generator import BaseGenerator

from memetic.insertion.greedy_insertion import GreedyInsertion

class GreedyGenerator(BaseGenerator):
    def __init__(self):
        self.inserter = GreedyInsertion()
    
    def generate(self, problem: PDPTWProblem, num_solutions: int) -> PDPTWSolution:
        solutions = []
        for _ in range(num_solutions):
            solution = self._generate_single_solution(problem)
            solutions.append(solution)
        return solutions

    def _generate_single_solution(self, problem: PDPTWProblem) -> PDPTWSolution:
        solution = PDPTWSolution(problem, [[0,0] for _ in range(problem.num_vehicles)])
        solution = self.inserter.insert(problem, solution)
        return solution