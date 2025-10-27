from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.solution_generators.base_generator import BaseGenerator
from memetic.solution_generators.greedy_generator import GreedyGenerator
from memetic.solution_generators.random_generator import RandomGenerator

import random

class HybridGenerator(BaseGenerator):
    def __init__(self, greedy_ratio: float = 0.5):
        self.greedy_ratio = greedy_ratio
        self.greedy_generator = GreedyGenerator()
        self.random_generator = RandomGenerator()
    
    def generate(self, problem: PDPTWProblem, num_solutions: int) -> list[PDPTWSolution]:
        solutions = []
        
        for i in range(num_solutions):
            if random.random() < self.greedy_ratio:
                solution = self.greedy_generator.generate(problem, num_solutions=1)[0]
            else:
                solution = self.random_generator.generate(problem, num_solutions=1)[0]
            solutions.append(solution)
        
        return solutions
    