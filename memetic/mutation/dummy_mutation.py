from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.mutation.base_mutation import BaseMutation

class DummyMutation(BaseMutation):
    """A dummy mutation operator that does nothing."""
    def __init__(self):
        super().__init__()

    def mutate(self, problem: PDPTWProblem, solution: PDPTWSolution, population = None) -> PDPTWSolution:
        """Returns the solution unchanged.

        Args:
            problem (PDPTWProblem): a problem instance
            solution (PDPTWSolution): a solution instance

        Returns:
            PDPTWSolution: the unchanged solution
        """
        return solution.clone()