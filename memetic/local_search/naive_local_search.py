from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness
class NaiveLocalSearch:
    def __init__(self, operators: list = [], max_no_improvement: int = 10):
        self.operators = operators
        self.max_no_improvement = max_no_improvement
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        no_improvement_count = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)
        
        while no_improvement_count < self.max_no_improvement:
            improved = False
            for operator in self.operators:
                new_solution = operator.apply(best_solution)
                new_fitness = fitness(problem, new_solution)
                print(new_fitness)
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    improved = True
                    break
            
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        print("Final fitness:", best_fitness)
        return best_solution