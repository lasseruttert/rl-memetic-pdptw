from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness
import random

class NaiveLocalSearch:
    def __init__(self, operators: list = [], max_no_improvement: int = 3, max_iterations: int = 50):
        self.operators = operators
        self.max_no_improvement = max_no_improvement
        self.max_iterations = max_iterations
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        no_improvement_count = 0
        iteration = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)
        
        while no_improvement_count < self.max_no_improvement and iteration < self.max_iterations:
            improved = False
            for operator in self.operators:
                new_solution = operator.apply(problem, best_solution)
                operator.applications += 1
                new_fitness = fitness(problem, new_solution)
                if new_fitness < best_fitness:
                    operator.improvements += 1
                    best_solution = new_solution
                    best_fitness = new_fitness
                    improved = True
                    break
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            iteration += 1
        return best_solution, best_fitness