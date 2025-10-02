from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from utils.li_lim_reader import li_lim_reader

from memetic.solution_generators.random_solution import generate_random_solution

from memetic.crossover.srex import SREXCrossover
from memetic.local_search.naive_local_search import NaiveLocalSearch

if __name__ == "__main__":
    # Example usage
    problem = li_lim_reader('G:/Meine Ablage/rl-memetic-pdptw/data/pdp_100/lc109.txt')
    parent1 = generate_random_solution(problem)
    parent2 = generate_random_solution(problem)
    
    local_search = NaiveLocalSearch(max_no_improvement=10)
    parent1, _ = local_search.search(problem, parent1)
    parent2, _ = local_search.search(problem, parent2)
    
    crossover = SREXCrossover()
    offsprings = crossover.crossover(problem, parent1, parent2)
    offspring = offsprings[3] if offsprings else None
    
    print("Parent 1:", parent1)
    print("Parent 2:", parent2)
    print("Offspring:", offspring)
    print("Feasible?:", offspring.check_feasibility())