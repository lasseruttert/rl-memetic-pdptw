from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness

from memetic.solution_generators.base_generator import BaseGenerator
from memetic.crossover.base_crossover import BaseCrossover
from memetic.mutation.base_mutation import BaseMutation
from memetic.local_search.base_local_search import BaseLocalSearch
from memetic.selection.base_selection import BaseSelection

from memetic.solution_generators.random_generator import RandomGenerator
from memetic.selection.k_tournament import KTournamentSelection
from memetic.crossover.srex import SREXCrossover
from memetic.mutation.naive_mutation import NaiveMutation
from memetic.local_search.naive_local_search import NaiveLocalSearch

from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.solution_operators.route_elimination import RouteEliminationOperator
from memetic.solution_operators.flip import FlipOperator
from memetic.solution_operators.swap_within import SwapWithinOperator
from memetic.solution_operators.swap_between import SwapBetweenOperator
from memetic.solution_operators.transfer import TransferOperator
from memetic.solution_operators.cls_m1 import CLSM1Operator
from memetic.solution_operators.cls_m2 import CLSM2Operator
from memetic.solution_operators.cls_m3 import CLSM3Operator
from memetic.solution_operators.cls_m4 import CLSM4Operator

import time
import random
import copy

class MemeticSolver:
    def __init__(
        self, 
        population_size: int = 10, 
        max_generations: int = 100,
        max_time_seconds: int = 600,
        max_no_improvement: int = 10,
        
        inital_solution_generator: BaseGenerator = None,
        
        selection_operator: BaseSelection = None,
        crossover_operator: BaseCrossover = None,
        mutation_operator: BaseMutation = None,
        local_search_operator: BaseLocalSearch = None,
        
        fitness_function = fitness,
        
        ensure_diversity_interval: int = 3,
        
        evaluation_interval: int = 10,
        verbose: bool = False,
        track_history: bool = False
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.max_no_improvement = max_no_improvement
        
        self.initial_solution_generator = inital_solution_generator if inital_solution_generator is not None else RandomGenerator()
        
        self.selection_operator = selection_operator if selection_operator is not None else KTournamentSelection(k=2)
        self.crossover_operator = crossover_operator if crossover_operator is not None else SREXCrossover()
        
        self.mutation_operator = mutation_operator if mutation_operator is not None else NaiveMutation(
            operators=[
                ReinsertOperator(),
                FlipOperator(),
                SwapWithinOperator(),
                SwapBetweenOperator(),
                TransferOperator()
            ],
            max_iterations=1
        )
        
        self.local_search_operator = local_search_operator if local_search_operator is not None else NaiveLocalSearch(
            operators=[
                ReinsertOperator(),
                ReinsertOperator(max_attempts=5,clustered=True),
                ReinsertOperator(allow_same_vehicle=False),
                ReinsertOperator(allow_same_vehicle=False, allow_new_vehicles=False),
                
                RouteEliminationOperator(),
                
                CLSM1Operator(),
                CLSM2Operator(),
                CLSM3Operator(),
                CLSM4Operator(),
                
                TransferOperator(single_route=True),
                TransferOperator(max_attempts=5,single_route=True),
                
                SwapBetweenOperator(),
            ],
            max_no_improvement=10,
            max_iterations=30
        )
        
        self.fitness_function = fitness_function
        
        self.ensure_diversity_interval = ensure_diversity_interval
        
        self.evaluation_interval = evaluation_interval
        self.evaluations = {}
        self.verbose = verbose
        self.track_history = track_history
        if self.track_history:
            self.history = {}
    
    def solve(self, problem: PDPTWProblem, initial_solution: PDPTWSolution = None) -> PDPTWSolution:
        best_solution = None
        best_fitness = float('inf')
        
        fitness_cache = {}
        current_fitnesses = [None for _ in range(self.population_size)]
        
        done = False
        generation = 0
        no_improvement_count = 0
        
        start_time = time.time()
        
        population = self.initial_solution_generator.generate(problem, self.population_size)
        for i in range(len(population)):
            population[i], current_fitnesses[i] = self.local_search_operator.search(problem, population[i])
            if current_fitnesses[i] < best_fitness:
                best_solution = population[i]
                best_fitness = current_fitnesses[i]
        
        while not done:
            if self.verbose: print(f"Generation {generation}, Best Fitness: {best_fitness}")
            if self.track_history:
                self.history[generation] = {
                    'population': copy.deepcopy(population),
                    'fitnesses': current_fitnesses.copy(),
                    'best_solution': copy.deepcopy(best_solution),
                    'best_fitness': best_fitness
                }
            
            if generation % self.evaluation_interval == 0:
                if self.verbose: print(f"Evaluating population at generation {generation}")
                self._evaluate(problem, population, current_fitnesses, generation)
            if generation % self.ensure_diversity_interval == 0:
                if self.verbose: print(f"Ensuring diversity at generation {generation}")
                population, current_fitnesses = self._ensure_diversity(problem, population, current_fitnesses)
            
            no_improvement_in_generation = True
            for i in range(len(population)):
                parent1 = population[i]
                parent2 = population[i + 1] if i + 1 < len(population) else population[0]
                
                children = self.crossover_operator.crossover(problem, parent1, parent2)
                child = random.choice(children) if children else None
                child = self.mutation_operator.mutate(problem, child)
                child, fitness = self.local_search_operator.search(problem, child)
                
                if fitness < current_fitnesses[i]:
                    population[i] = child
                    current_fitnesses[i] = fitness
                
                if fitness < best_fitness:
                    if self.verbose: print(f"New best solution found with fitness {fitness} at generation {generation}")
                    best_solution = child
                    best_fitness = fitness
                    
                    no_improvement_count = 0
                    no_improvement_in_generation = False
                
            
            if no_improvement_in_generation:
                if self.verbose: print(f"No improvement in generation {generation}")
                no_improvement_count += 1
                
            generation += 1
            
            if generation >= self.max_generations or no_improvement_count >= self.max_no_improvement or (time.time() - start_time) >= self.max_time_seconds:
                if self.verbose: print("Stopping criteria met.")
                done = True
        
        return best_solution
        
    def _ensure_diversity(self, problem: PDPTWProblem, population: list[PDPTWSolution], current_fitnesses: list):
        # Remove duplicates
        seen = set()
        for i in range(len(population)):
            identifier = population[i].hashed_encoding
            if identifier in seen:
                population[i] = self.initial_solution_generator.generate(population[i].problem, 1)[0]
                current_fitnesses[i] = self.fitness_function(problem, population[i])
            else:
                seen.add(identifier)
        
        # TODO: Use some distance metric to ensure diversity
        
        return population, current_fitnesses
        
    def _evaluate(self, problem: PDPTWProblem, population: list[PDPTWSolution], current_fitnesses: list, iteration: int):
        min_fitness = float('inf')
        max_fitness = float('-inf')
        avg_fitness = 0.0
        for i in range(len(population)):
            if current_fitnesses[i] is None:
                current_fitnesses[i] = self.fitness_function(problem, population[i])
            fitness = current_fitnesses[i]
            if fitness < min_fitness:
                min_fitness = fitness
            if fitness > max_fitness:
                max_fitness = fitness
            avg_fitness += fitness
        avg_fitness /= len(population)
        
        self.evaluations[iteration] = {
            'min': min_fitness,
            'max': max_fitness,
            'avg': avg_fitness
        }
        return # TODO