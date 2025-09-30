from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness
import random

class NaiveLocalSearch:
    """A simple local search framework that applies a list of operators iteratively until no improvement is found.
    """
    def __init__(self, operators: list = [], max_no_improvement: int = 3, max_iterations: int = 50, first_improvement: bool = True, random_operator_order: bool = False):
        """
        Args:
            operators (list, optional): List of local search operators to apply. Defaults to [].
            max_no_improvement (int, optional): Maximum number of consecutive iterations without improvement before stopping. Defaults to 3.
            max_iterations (int, optional): Maximum total iterations before stopping. Defaults to 50.
            first_improvement (bool, optional): Whether to accept the first improving move found (True) or search all operators for the best move (False). Defaults to True.
            random_operator_order (bool, optional): Whether to randomize the order of operators in each iteration. Defaults to False.
        """
        self.operators = operators
        self.max_no_improvement = max_no_improvement
        self.max_iterations = max_iterations
        self.first_improvement = first_improvement
        self.random_operator_order = random_operator_order
    
    def search(self, problem: PDPTWProblem, solution: PDPTWSolution) -> tuple[PDPTWSolution, float]:
        """Start the local search process.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            tuple[PDPTWSolution, float]: the best solution found and its fitness
        """
        no_improvement_count = 0
        iteration = 0
        best_solution = solution
        best_fitness = fitness(problem, best_solution)
        
        while no_improvement_count < self.max_no_improvement and iteration < self.max_iterations:
            improved = False
            best_neighbor = None
            best_fitness_neighbor = float('inf')
            if self.random_operator_order: random.shuffle(self.operators)
            for operator in self.operators:
                new_solution = operator.apply(problem, best_solution)
                operator.applications += 1
                new_fitness = fitness(problem, new_solution)
                if new_fitness < best_fitness:
                    operator.improvements += 1
                    if self.first_improvement:
                        best_solution = new_solution
                        best_fitness = new_fitness
                        improved = True
                        break
                    else:
                        if new_fitness < best_fitness_neighbor:
                            best_neighbor = new_solution
                            best_fitness_neighbor = new_fitness
            
            if not self.first_improvement and best_neighbor is not None:
                best_solution = best_neighbor
                best_fitness = best_fitness_neighbor
                improved = True
            
            no_improvement_count = 0 if improved else no_improvement_count + 1
            iteration += 1
        return best_solution, best_fitness