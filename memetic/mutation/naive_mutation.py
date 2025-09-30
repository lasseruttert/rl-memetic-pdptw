from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
import random

class NaiveMutation:
    """
    A simple mutation operator that randomly applies one of the given operators a specified number of times.
    """
    def __init__(self, operators: list = [], max_iterations: int = 1):
        """
        Args:
            operators (list, optional): List of mutation operators to choose from. Defaults to [].
            max_iterations (int, optional): Number of times to apply mutation. Defaults to 1.
        """
        self.operators = operators
        self.max_iterations = max_iterations
    
    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        """Mutate the given solution by randomly applying mutation operators.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            PDPTWSolution: the mutated solution
        """
        new_solution = solution.clone()
        for _ in range(self.max_iterations):
            operator = random.choice(self.operators)
            new_solution = operator.apply(problem, new_solution)
        return new_solution