from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.solution_generators.base_generator import BaseGenerator
from memetic.solution_generators.greedy_generator import GreedyGenerator
from memetic.solution_generators.random_generator import RandomGenerator

class HybridGenerator(BaseGenerator):
    def __init__(self, greedy_ratio: float = 0.5):
        self.greedy_ratio = greedy_ratio
        self.greedy_generator = GreedyGenerator()
        self.random_generator = RandomGenerator()
    
    def generate(self, problem: PDPTWProblem, num_solutions: int) -> PDPTWSolution:
        num_greedy = int(num_solutions * self.greedy_ratio)
        num_random = num_solutions - num_greedy
        
        solutions = []
        if num_greedy > 0:
            solutions.extend(self.greedy_generator.generate(problem, num_greedy))
        if num_random > 0:
            solutions.extend(self.random_generator.generate(problem, num_random))
        
        return solutions