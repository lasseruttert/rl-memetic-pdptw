from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness

from memetic.solution_generators.random_solution import generate_random_solution

from memetic.crossover.srex import SREXCrossover
from memetic.mutation.naive_mutation import NaiveMutation
from memetic.local_search.naive_local_search import NaiveLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator

import time
import random

class MemeticSolver:
    def __init__(
        self, 
        population_size: int = 30, 
        max_generations: int = 100,
        max_time_seconds: int = 600,
        max_no_improvement: int = 20,
        
        inital_solution_generator = generate_random_solution, 
        
        selection_operator = None,
        crossover_operator = None,
        mutation_operator = None,
        local_search_operator = None,
        
        fitness_function = fitness,
        
        evaluation_interval: int = 10,
        verbose: bool = False,
        track_history: bool = False
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.max_no_improvement = max_no_improvement
        
        self.initial_solution_generator = inital_solution_generator
        
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator if crossover_operator is not None else SREXCrossover()
        
        self.mutatation_operator = mutation_operator if mutation_operator is not None else NaiveMutation(
            operators=[
                ReinsertOperator(),
                FlipOperator(),
                SwapWithinOperator(),
                SwapBetweenOperator(),
                TransferOperator()
            ],
            max_iterations=1
        )
        
        if local_search_operator is None:
            operators = [
                ReinsertOperator(max_attempts=5,clustered=True),
                ReinsertOperator(allow_same_vehicle=False),
                ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
                
                RouteEliminationOperator(),
                
                SwapBetweenOperator(),
                
                TransferOperator(single_route=True),
                
                CLSM1Operator(),
                CLSM2Operator(),
                CLSM3Operator(),
                CLSM4Operator()
            ]
            local_search_operator = NaiveLocalSearch(operators=operators, max_no_improvement=3, max_iterations=50)
        
        self.local_search_operator = local_search_operator
        
        self.fitness_function = fitness_function
        
        self.evaluation_interval = evaluation_interval
        self.verbose = verbose
        self.track_history = track_history
    
    def solve(self, problem: PDPTWProblem, initial_solution: PDPTWSolution = None) -> PDPTWSolution:
        best_solution = None
        best_fitness = float('inf')
        
        fitness_cache = {}
        current_fitnesses = [None for _ in range(self.population_size)]
        
        done = False
        generation = 0
        no_improvement_count = 0
        
        start_time = time.time()
        
        population = [self.initial_solution_generator(problem) for _ in range(self.population_size)]
        for i in range(len(population)):
            population[i], current_fitnesses[i] = self.local_search_operator.search(problem, population[i])
            if current_fitnesses[i] < best_fitness:
                best_solution = population[i]
                best_fitness = current_fitnesses[i]
        
        while not done:
            print(f"Generation {generation}, Best Fitness: {best_fitness}")
            no_improvement_in_generation = True
            for i in range(len(population) - 1):
                parent1 = population[i]
                parent2 = population[i + 1]
                
                child = self.crossover_operator.crossover(problem, parent1, parent2)
                child = self.mutatation_operator.mutate(problem, child)
                child, fitness = self.local_search_operator.search(problem, child)
                current_fitnesses[i] = fitness
                
                if fitness < best_fitness:
                    best_solution = child
                    best_fitness = fitness
                    
                    population[i] = child
                    
                    no_improvement_count = 0
                    no_improvement_in_generation = False
                
            
            if no_improvement_in_generation:
                no_improvement_count += 1
                
            generation += 1
            
            if generation >= self.max_generations or no_improvement_count >= self.max_no_improvement or (time.time() - start_time) >= self.max_time_seconds:
                done = True
        
        return best_solution
        