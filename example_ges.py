from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.solution_generators.random_solution import generate_random_solution
from memetic.vehicle_minimization.guided_ejection_search import GuidedEjectionSearch
from memetic.solution_operators.route_elimination import RouteEliminationOperator

if __name__ == "__main__":
    # Example usage
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc101.txt')
    initial_solution = generate_random_solution(problem)
    initial_solution_s = initial_solution.clone()
    
    ges = GuidedEjectionSearch(5)
    improved_solution = ges.apply(problem, initial_solution)
    
    # route_elim = RouteEliminationOperator()
    # improved_solution = route_elim.apply(problem, initial_solution)
    
    print("Initial Solution:", initial_solution_s)
    print("Improved Solution:", improved_solution)
    print("Feasible?:", improved_solution.check_feasibility())