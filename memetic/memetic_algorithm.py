from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.fitness.fitness import fitness
from memetic.utils.compare import compare

from memetic.solution_generators.base_generator import BaseGenerator
from memetic.crossover.base_crossover import BaseCrossover
from memetic.mutation.base_mutation import BaseMutation
from memetic.local_search.base_local_search import BaseLocalSearch
from memetic.selection.base_selection import BaseSelection

from memetic.solution_generators.random_generator import RandomGenerator
from memetic.solution_generators.greedy_generator import GreedyGenerator
from memetic.solution_generators.hybrid_generator import HybridGenerator

from memetic.selection.k_tournament import KTournamentSelection

from memetic.crossover.srex import SREXCrossover

from memetic.mutation.naive_mutation import NaiveMutation

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.local_search.adaptive_local_search import AdaptiveLocalSearch
from memetic.local_search.random_local_search import RandomLocalSearch

from memetic.solution_operators.base_operator import BaseOperator
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
from memetic.solution_operators.two_opt import TwoOptOperator
from memetic.solution_operators.two_opt_star import TwoOptStarOperator

from memetic.utils.distance_measure import DistanceMeasure
from memetic.utils.edge_frequency import SparseCentroid, compute_sparse_edges 

import time
import random
import copy
from dataclasses import dataclass
from typing import Union, Optional, Callable

class MemeticSolver: 

    def __init__(
        self, 
        population_size: int = 10, 
        max_generations: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        max_no_improvement: Optional[int] = None,
        
        initial_solution_generator: Union[BaseGenerator, str] = None,

        selection_operator: Union[BaseSelection, str] = None,
        crossover_operator: Union[BaseCrossover, str] = None,
        mutation_operators: Union[list[BaseOperator], list[str]] = None,
        mutation_operator: Union[BaseMutation, str] = None,
        local_search_operators: Union[list[BaseOperator], list[str]] = None,
        local_search_operator: Union[BaseLocalSearch, str] = None,

        fitness_function: Callable = None,
        
        ensure_diversity_interval: int = 3,
        use_centroid: bool = False,
        
        calculate_upper_bound_vehicles: bool = False,

        init_with_local_search: bool = True,

        reproduction_strategy: str = 'sequential',

        evaluation_interval: int = 10,
        verbose: bool = False,
        track_convergence: bool = False,
        track_history: bool = False
    ):
        """
        A modular memetic algorithm for solving the Pickup and Delivery Problem with Time Windows (PDPTW).
        
        Args:
            population_size (int): Number of individuals in the population.
            max_generations (Optional[int]): Maximum number of generations to run. If None, no limit on generations.
            max_time_seconds (Optional[int]): Maximum time in seconds to run the algorithm. If None, no limit on time.
            max_no_improvement (Optional[int]): Maximum number of generations without improvement before stopping. If None, no limit on no improvement.
            
            initial_solution_generator (Union[BaseGenerator, str]): Initial solution generator to use. Can be an instance of BaseGenerator or a string identifier. Choose from 'random', 'greedy', or 'hybrid'.
            selection_operator (Union[BaseSelection, str]): Selection operator to use. Can be an instance of BaseSelection or a string identifier. Choose from 'k_tournament'.
            crossover_operator (Union[BaseCrossover, str]): Crossover operator to use. Can be an instance of BaseCrossover or a string identifier. Choose from 'srex'.
            mutation_operators (Union[list[BaseOperator], list[str]]): List operators to use within mutation to apply to a solution. Choose from 'reinsert', 'route_elimination', 'flip', 'swap_within', 'swap_between', 'transfer', 'cls_m1', 'cls_m2', 'cls_m3', 'cls_m4', 'two_opt', 'two_opt_star'.
            mutation_operator (Union[BaseMutation, str]): Mutation operator to use. Can be an instance of BaseMutation or a string identifier. Choose from 'naive'.
            local_search_operators (Union[list[BaseOperator], list[str]]): List of local search operators to use within local search to apply to a solution. Choose from 'reinsert', 'route_elimination', 'flip', 'swap_within', 'swap_between', 'transfer', 'cls_m1', 'cls_m2', 'cls_m3', 'cls_m4', 'two_opt', 'two_opt_star'.
            local_search_operator (Union[BaseLocalSearch, str]): Local search operator to use. Choose from 'naive', 'first_improvement', or 'best_improvement'.
            fitness_function: Function to evaluate the fitness of a solution.
            ensure_diversity_interval (int): Interval (in generations) to ensure diversity in the population.
            use_centroid (bool): Whether to use centroid-based diversity measure.
            calculate_upper_bound_vehicles (bool): Whether to calculate an upper bound on the number of vehicles needed.
            init_with_local_search (bool): Whether to improve the initial population with local search.
            reproduction_strategy (str): Strategy for reproduction. Default is 'sequential'.
            evaluation_interval (int): Interval (in generations) to evaluate the population.
            verbose (bool): Whether to print verbose output.
            track_convergence (bool): Whether to track convergence metrics over time.
            track_history (bool): Whether to track the history of the algorithm.
        """
        
        self.population_size = population_size
        
        if all(param is None for param in [max_generations, max_time_seconds, max_no_improvement]):
            raise ValueError("At least one stopping criteria must be provided (max_generations, max_time_seconds, max_no_improvement).")
        
        self.max_generations = max_generations
        self.max_time_seconds = max_time_seconds
        self.max_no_improvement = max_no_improvement
        
        # * Initial Solution Generator
        if initial_solution_generator is None:
            initial_solution_generator = 'random'
        if isinstance(initial_solution_generator, str):
            if initial_solution_generator.lower() == 'random':
                initial_solution_generator = RandomGenerator()
            elif initial_solution_generator.lower() == 'greedy':
                initial_solution_generator = GreedyGenerator()
            elif initial_solution_generator.lower() == 'hybrid':
                initial_solution_generator = HybridGenerator()
            else:
                raise ValueError(f"Unknown initial solution generator: {initial_solution_generator}")
        if isinstance(initial_solution_generator, BaseGenerator):
            self.initial_solution_generator = initial_solution_generator
        
        # * Selection Operator
        if selection_operator is None:
            selection_operator = 'k_tournament'
        if isinstance(selection_operator, str):
            if selection_operator.lower() == 'k_tournament':
                selection_operator = KTournamentSelection(k=2)
            else:
                raise ValueError(f"Unknown selection operator: {selection_operator}")
        if isinstance(selection_operator, BaseSelection):
            self.selection_operator = selection_operator
            
        # * Crossover Operator
        if crossover_operator is None:
            crossover_operator = 'srex'
        if isinstance(crossover_operator, str):
            if crossover_operator.lower() == 'srex':
                crossover_operator = SREXCrossover()
            else:
                raise ValueError(f"Unknown crossover operator: {crossover_operator}")
        if isinstance(crossover_operator, BaseCrossover):
            self.crossover_operator = crossover_operator

        # * Mutation Operators
        if isinstance(mutation_operators, list) and all(isinstance(op, str) for op in mutation_operators):
            ops = []
            for op in mutation_operators:
                if op.lower() == 'reinsert':
                    ops.append(ReinsertOperator())
                elif op.lower() == 'reinsert_clustered':
                    ops.append(ReinsertOperator(max_attempts=5, clustered=True))
                elif op.lower() == 'reinsert_no_same_vehicle':
                    ops.append(ReinsertOperator(allow_same_vehicle=False))
                elif op.lower() == 'reinsert_no_same_vehicle_no_new_route':
                    ops.append(ReinsertOperator(allow_same_vehicle=False, allow_new_route=False))
                elif op.lower() == 'route_elimination':
                    ops.append(RouteEliminationOperator())
                elif op.lower() == 'flip':
                    ops.append(FlipOperator())
                elif op.lower() == 'flip_single':
                    ops.append(FlipOperator(single=True))
                elif op.lower() == 'swap_within':
                    ops.append(SwapWithinOperator())
                elif op.lower() == 'swap_within_single':
                    ops.append(SwapWithinOperator(single=True))
                elif op.lower() == 'swap_between':
                    ops.append(SwapBetweenOperator())
                elif op.lower() == 'transfer':
                    ops.append(TransferOperator())
                elif op.lower() == 'transfer_single':
                    ops.append(TransferOperator(single=True))
                elif op.lower() == 'cls_m1':
                    ops.append(CLSM1Operator())
                elif op.lower() == 'cls_m2':
                    ops.append(CLSM2Operator())
                elif op.lower() == 'cls_m3':
                    ops.append(CLSM3Operator())
                elif op.lower() == 'cls_m4':
                    ops.append(CLSM4Operator())
                elif op.lower() == 'two_opt':
                    ops.append(TwoOptOperator())
                else:
                    raise ValueError(f"Unknown mutation operator: {op}")
            mutation_operators = ops

        self.mutation_operators = None
        if isinstance(mutation_operators, list) and all(isinstance(op, BaseOperator) for op in mutation_operators):
            self.mutation_operators = mutation_operators
        
        if self.mutation_operators is None or len(self.mutation_operators) == 0:
            self.mutation_operators = [
                ReinsertOperator(),
                RouteEliminationOperator(),
                FlipOperator(),
                SwapWithinOperator(),
                SwapBetweenOperator(),
                TransferOperator(),
                CLSM1Operator(),
                CLSM2Operator(),
                CLSM3Operator(),
                CLSM4Operator(),
                TwoOptOperator(),
            ]

        # * Mutation Operator
        if mutation_operator is None:
            mutation_operator = 'naive'
        if isinstance(mutation_operator, str):
            if mutation_operator.lower() == 'naive':
                mutation_operator = NaiveMutation(
                    operators=self.mutation_operators,
                    max_iterations=1
                )
            else:
                raise ValueError(f"Unknown mutation operator: {mutation_operator}")
        if isinstance(mutation_operator, BaseMutation):
            self.mutation_operator = mutation_operator

        # * Local Search Operators
        if isinstance(local_search_operators, list) and all(isinstance(op, str) for op in local_search_operators):
            ops = []
            for op in local_search_operators:
                if op.lower() == 'reinsert':
                    ops.append(ReinsertOperator())
                elif op.lower() == 'reinsert_clustered':
                    ops.append(ReinsertOperator(max_attempts=5, clustered=True))
                elif op.lower() == 'reinsert_no_same_vehicle':
                    ops.append(ReinsertOperator(allow_same_vehicle=False))
                elif op.lower() == 'reinsert_no_same_vehicle_no_new_route':
                    ops.append(ReinsertOperator(allow_same_vehicle=False, allow_new_route=False))
                elif op.lower() == 'route_elimination':
                    ops.append(RouteEliminationOperator())
                elif op.lower() == 'flip':
                    ops.append(FlipOperator())
                elif op.lower() == 'flip_single':
                    ops.append(FlipOperator(single=True))
                elif op.lower() == 'swap_within':
                    ops.append(SwapWithinOperator())
                elif op.lower() == 'swap_within_single':
                    ops.append(SwapWithinOperator(single=True))
                elif op.lower() == 'swap_between':
                    ops.append(SwapBetweenOperator())
                elif op.lower() == 'transfer':
                    ops.append(TransferOperator())
                elif op.lower() == 'transfer_single':
                    ops.append(TransferOperator(single=True))
                elif op.lower() == 'cls_m1':
                    ops.append(CLSM1Operator())
                elif op.lower() == 'cls_m2':
                    ops.append(CLSM2Operator())
                elif op.lower() == 'cls_m3':
                    ops.append(CLSM3Operator())
                elif op.lower() == 'cls_m4':
                    ops.append(CLSM4Operator())
                elif op.lower() == 'two_opt':
                    ops.append(TwoOptOperator())
                else:
                    raise ValueError(f"Unknown local search operator: {op}")
            local_search_operators = ops
            
        self.local_search_operators = None
        if isinstance(local_search_operators, list) and all(isinstance(op, BaseOperator) for op in local_search_operators):
            self.local_search_operators = local_search_operators

        if self.local_search_operators is None or len(self.local_search_operators) == 0:
            self.local_search_operators = [
                ReinsertOperator(),
                ReinsertOperator(max_attempts=5, clustered=True),
                ReinsertOperator(allow_same_vehicle=False),
                
                RouteEliminationOperator(),
                TransferOperator(),
                CLSM1Operator(),
                CLSM2Operator(),
                CLSM3Operator(),
                CLSM4Operator(),
            ]

        # * Local Search Operator
        if local_search_operator is None:
            local_search_operator = 'naive'
        if isinstance(local_search_operator, str):
            if local_search_operator.lower() == 'naive':
                local_search_operator = NaiveLocalSearch(
                    operators=self.local_search_operators,
                    max_no_improvement=10,
                    max_iterations=30
                )
            elif local_search_operator.lower() == 'naive_best':
                local_search_operator = NaiveLocalSearch(
                    operators=self.local_search_operators,
                    max_no_improvement=20,
                    max_iterations=60,
                    best_improvement=True
                )
            elif local_search_operator.lower() == 'adaptive':
                local_search_operator = AdaptiveLocalSearch(
                    operators=self.local_search_operators,
                    max_no_improvement=20,
                    max_iterations=60,
                    use_centroid=use_centroid
                )
            elif local_search_operator.lower() == 'random':
                local_search_operator = RandomLocalSearch(
                    operators=self.local_search_operators,
                    max_no_improvement=20,
                    max_iterations=60
                )
            else:
                raise ValueError(f"Unknown local search operator: {local_search_operator}")
        if isinstance(local_search_operator, BaseLocalSearch):
            self.local_search_operator = local_search_operator
        
        
        if fitness_function is None:
            fitness_function = fitness
        self.fitness_function = fitness_function
        
        self.ensure_diversity_interval = ensure_diversity_interval
        self.use_centroid = use_centroid
        
        self.calculate_upper_bound_vehicles = calculate_upper_bound_vehicles
        
        self.init_with_local_search = init_with_local_search

        self.reproduction_strategy = reproduction_strategy.lower()
        # Set reproducer function pointer based on strategy
        if self.reproduction_strategy == 'sequential':
            self.reproducer = self._reproduce_sequential
        elif self.reproduction_strategy == 'sequential_with_buffer':
            self.reproducer = self._reproduce_sequential_with_buffer
        elif self.reproduction_strategy == 'binary':
            self.reproducer = self._reproduce_binary
            if self.max_generations is not None:
                self.max_generations *= self.population_size // 2
            if self.max_no_improvement is not None:
                self.max_no_improvement *= self.population_size // 2
        else:
            raise ValueError(f"Unknown reproduction strategy: {self.reproduction_strategy}")

        self.evaluation_interval = evaluation_interval
        self.evaluations = {}
        
        self.verbose = verbose
        
        self.track_convergence = track_convergence
        self.convergence = {}
        self.track_history = track_history
        if self.track_history:
            self.history = {}

    def solve(self, problem: PDPTWProblem, initial_population: list[PDPTWSolution] = None) -> PDPTWSolution:
        """Solve the PDPTW problem using the memetic algorithm.

        Args:
            problem (PDPTWProblem): The PDPTW problem instance.
            initial_population (list[PDPTWSolution], optional): Initial population to start the search. Defaults to None.

        Returns:
            PDPTWSolution: The best solution found.
        """
        # Reset tracking data
        self.evaluations = {}
        self.convergence = {}
        if self.track_history:
            self.history = {}
        
        best_solution = None
        best_fitness = float('inf')
        
        fitness_cache = {}
        current_fitnesses = [None for _ in range(self.population_size)]
        current_num_vehicles = [None for _ in range(self.population_size)]
        
        done = False
        generation = 0
        no_improvement_count = 0
        executed_backup_strategy = False
        
        start_time = time.time()
        
        if self.calculate_upper_bound_vehicles:
            original_num_vehicles = problem.num_vehicles
            if self.verbose: 
                print(f"Original number of vehicles: {original_num_vehicles}")
                print(f"Calculating upper bound on number of vehicles...")
            upper_bound_vehicles = self._calculate_upper_bound_vehicles(problem)
            if upper_bound_vehicles < problem.num_vehicles:
                if self.verbose: print(f"Setting number of vehicles to upper bound of {upper_bound_vehicles} vehicles")
                problem.num_vehicles = upper_bound_vehicles
        
        if initial_population is not None:
            if len(initial_population) != self.population_size:
                raise ValueError(f"Initial population size {len(initial_population)} does not match specified population size {self.population_size}.")
            population = initial_population
        else:
            population = self.initial_solution_generator.generate(problem, self.population_size)
            
        
        if self.init_with_local_search:
            if self.verbose: print("Improving initial population with local search...")
            for i in range(len(population)):
                population[i], current_fitnesses[i] = self.local_search_operator.search(problem, population[i])
                current_num_vehicles[i] = population[i].num_vehicles_used
                if current_fitnesses[i] < best_fitness:
                    best_solution = population[i]
                    best_fitness = current_fitnesses[i]
        else:
            for i in range(len(population)):
                current_fitnesses[i] = self.fitness_function(problem, population[i])
                current_num_vehicles[i] = population[i].num_vehicles_used
                if current_fitnesses[i] < best_fitness:
                    best_solution = population[i]
                    best_fitness = current_fitnesses[i]
        
        while not done:
            if self.verbose: 
                print(f"\n\033[4;34mGeneration {generation}:\033[0m {(time.time() - start_time):.2f} seconds, Best Fitness: {best_fitness:.2f}")
                print(f"\033[1;33mBest Solution - Number of Vehicles: {best_solution.num_vehicles_used}, Total Distance: {best_solution.total_distance:.2f}\033[0m\n")
            if self.track_history:
                self.history[generation] = {
                    'population': copy.deepcopy(population),
                    'fitnesses': current_fitnesses.copy(),
                    'best_solution': copy.deepcopy(best_solution),
                    'best_fitness': best_fitness
                }
            
            if generation % self.evaluation_interval == 0:
                if self.verbose: print(f"Evaluating population at generation {generation}")
                self._evaluate(problem, population, current_fitnesses, generation, start_time)
            if generation % self.ensure_diversity_interval == 0:
                if self.verbose: print(f"Ensuring diversity at generation {generation}")
                population, current_fitnesses, current_num_vehicles = self._ensure_diversity(problem, population, current_fitnesses, current_num_vehicles)

            # Reproduction step
            population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count = self.reproducer(
                problem, population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, generation, start_time, no_improvement_count
            )
                
            generation += 1
            
            if not executed_backup_strategy and self.max_no_improvement is not None and no_improvement_count >= self.max_no_improvement // 2:
                if self.verbose: 
                    print("\033[1;31mNo improvement for half of max_no_improvement.\033[0m")
                population, current_fitnesses, current_num_vehicles = self._execute_backup_strategy(problem, population, current_fitnesses, current_num_vehicles)
                executed_backup_strategy = True

            if self._check_if_done(generation, no_improvement_count, start_time):
                if self.verbose: print("\033[1;31mStopping criteria met.\033[0m\n")
                done = True
        
        if self.calculate_upper_bound_vehicles: problem.num_vehicles = original_num_vehicles

        return best_solution

    def _reproduce_sequential(
        self,
        problem: PDPTWProblem,
        population: list[PDPTWSolution],
        current_fitnesses: list,
        current_num_vehicles: list,
        best_fitness: float,
        best_solution: PDPTWSolution,
        generation: int,
        start_time: float,
        no_improvement_count: int
    ) -> tuple[list[PDPTWSolution], list, list, float, PDPTWSolution, int]:
        """Sequential reproduction: each individual pairs with the next one.

        Returns:
            tuple: (population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count)
        """
        no_improvement_in_generation = True
        for i in range(len(population)):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            children = self.crossover_operator.crossover(problem, parent1, parent2)
            if not children: continue
            child = random.choice(children) if children else None
            child = self.mutation_operator.mutate(problem, child)
            child, fitness = self.local_search_operator.search(problem, child)

            if compare(fitness, child.num_vehicles_used, current_fitnesses[i], current_num_vehicles[i]):
                population[i] = child
                current_fitnesses[i] = fitness
                current_num_vehicles[i] = child.num_vehicles_used

                if self.track_convergence:
                    self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}

            if compare(fitness, child.num_vehicles_used, best_fitness, best_solution.num_vehicles_used):
                if self.verbose: print(f"New best solution found with fitness {fitness:.2f} at generation {generation}")
                best_solution = child.clone()
                best_fitness = fitness

                no_improvement_count = 0
                no_improvement_in_generation = False

                if self.track_convergence:
                    self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}

        if no_improvement_in_generation:
            if self.verbose: print(f"No improvement in generation {generation}")
            no_improvement_count += 1

        return population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count
    
    def _reproduce_sequential_with_buffer(
        self,
        problem: PDPTWProblem,
        population: list[PDPTWSolution],
        current_fitnesses: list,
        current_num_vehicles: list,
        best_fitness: float,
        best_solution: PDPTWSolution,
        generation: int,
        start_time: float,
        no_improvement_count: int
    ) -> tuple[list[PDPTWSolution], list, list, float, PDPTWSolution, int]:
        """Sequential reproduction: each individual pairs with the next one.

        Returns:
            tuple: (population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count)
        """
        no_improvement_in_generation = True
        new_population = [None for _ in range(len(population))]
        new_fitnesses = [None for _ in range(len(population))]
        new_num_vehicles = [None for _ in range(len(population))]
        for i in range(len(population)):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            children = self.crossover_operator.crossover(problem, parent1, parent2)
            if not children: continue
            child = random.choice(children) if children else None
            child = self.mutation_operator.mutate(problem, child)
            child, fitness = self.local_search_operator.search(problem, child)

            if compare(fitness, child.num_vehicles_used, current_fitnesses[i], current_num_vehicles[i]):
                new_population[i] = child
                new_fitnesses[i] = fitness
                new_num_vehicles[i] = child.num_vehicles_used

                if self.track_convergence:
                    self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}

            if compare(fitness, child.num_vehicles_used, best_fitness, best_solution.num_vehicles_used):
                if self.verbose: print(f"New best solution found with fitness {fitness:.2f} at generation {generation}")
                best_solution = child.clone()
                best_fitness = fitness

                no_improvement_count = 0
                no_improvement_in_generation = False

                if self.track_convergence:
                    self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}

        if no_improvement_in_generation:
            if self.verbose: print(f"No improvement in generation {generation}")
            no_improvement_count += 1

        return new_population, new_fitnesses, new_num_vehicles, best_fitness, best_solution, no_improvement_count
    
    def _reproduce_binary(
        self,
        problem: PDPTWProblem,
        population: list[PDPTWSolution],
        current_fitnesses: list,
        current_num_vehicles: list,
        best_fitness: float,
        best_solution: PDPTWSolution,
        generation: int,
        start_time: float,
        no_improvement_count: int
    ) -> tuple[list[PDPTWSolution], list, list, float, PDPTWSolution, int]:
        """Binary reproduction.

        Returns:
            tuple: (population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count)
        """
        no_improvement_in_generation = True
        parent1, parent1_idx = self.selection_operator.select(population, current_fitnesses)
        parent2, parent2_idx = self.selection_operator.select(population, current_fitnesses)
        children = self.crossover_operator.crossover(problem, parent1, parent2)
        if not children:
            return population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count
        child = random.choice(children) if children else None
        child = self.mutation_operator.mutate(problem, child)
        child, fitness = self.local_search_operator.search(problem, child)
        
        worse_idx = parent1_idx if compare(current_fitnesses[parent1_idx], current_num_vehicles[parent1_idx], current_fitnesses[parent2_idx], current_num_vehicles[parent2_idx]) else parent2_idx
        if compare(fitness, child.num_vehicles_used, current_fitnesses[worse_idx], current_num_vehicles[worse_idx]):
            population[worse_idx] = child
            current_fitnesses[worse_idx] = fitness
            current_num_vehicles[worse_idx] = child.num_vehicles_used

            if self.track_convergence:
                self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}
        if compare(fitness, child.num_vehicles_used, best_fitness, best_solution.num_vehicles_used):
            if self.verbose: print(f"New best solution found with fitness {fitness:.2f} at generation {generation}")
            best_solution = child.clone()
            best_fitness = fitness

            no_improvement_count = 0
            no_improvement_in_generation = False

            if self.track_convergence:
                self.convergence[time.time() - start_time] = {"best_fitness": best_fitness, "num_vehicles": best_solution.num_vehicles_used, "avg_fitness": sum(current_fitnesses) / len(current_fitnesses)}
        if no_improvement_in_generation:
            if self.verbose: print(f"No improvement in generation {generation}")
            no_improvement_count += 1
        return population, current_fitnesses, current_num_vehicles, best_fitness, best_solution, no_improvement_count

    def _ensure_diversity(self, problem: PDPTWProblem, population: list[PDPTWSolution], current_fitnesses: list, current_num_vehicles: list):
        """Ensure diversity in the population by removing duplicates and using a distance metric."""
        # Remove duplicates
        seen = set()
        for i in range(len(population)):
            identifier = population[i].hashed_encoding
            if identifier in seen:
                population[i] = self.initial_solution_generator.generate(problem, 1)[0]
                current_fitnesses[i] = self.fitness_function(problem, population[i])
                current_num_vehicles[i] = population[i].num_vehicles_used
                
            else:
                seen.add(identifier)

        # TODO: use some distance metric to ensure diversity

        return population, current_fitnesses, current_num_vehicles
    
    def _execute_backup_strategy(self, problem: PDPTWProblem, population: list[PDPTWSolution], current_fitnesses: list, current_num_vehicles: list):
        """Execute a backup strategy to diversify the population.""" # TODO Implement other stategies
        for i in range(len(population)):
            if random.random() < 0.5:
                population[i] = self.initial_solution_generator.generate(problem, 1)[0]
                current_fitnesses[i] = self.fitness_function(problem, population[i])
                current_num_vehicles[i] = population[i].num_vehicles_used
        return population, current_fitnesses, current_num_vehicles
        
    def _evaluate(self, problem: PDPTWProblem, population: list[PDPTWSolution], current_fitnesses: list, iteration: int, start_time: float):
        """Evaluate the population and store statistics."""
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
        avg_fitness /= len(population) if len(population) > 0 else 0.0
        
        
        self.evaluations[iteration] = {
            'time': time.time() - start_time,
            'min': min_fitness,
            'max': max_fitness,
            'avg': avg_fitness
        }
        return # TODO: add more detailed evaluation
    
    def _calculate_upper_bound_vehicles(self, problem: PDPTWProblem) -> int:
        """Calculate an upper bound on the number of vehicles needed to serve all customers."""
        old_num_no_improvement = self.local_search_operator.max_no_improvement
        self.local_search_operator.max_no_improvement = 30
        old_num_iterations = self.local_search_operator.max_iterations
        self.local_search_operator.max_iterations = 50
        temp_problem = copy.deepcopy(problem)
        solution_found = True
        while solution_found:
            temp_problem.num_vehicles -= 1
            if temp_problem.num_vehicles <= 0:
                temp_problem.num_vehicles = 1
                break
            temp_solution = self.initial_solution_generator.generate(temp_problem, 1)[0]
            temp_solution, fitness = self.local_search_operator.search(temp_problem, temp_solution)
            if temp_solution.is_feasible:
                if self.verbose: print(f"Feasible with {temp_problem.num_vehicles} vehicles")
                continue
            else:
                if self.verbose: print(f"Infeasible with {temp_problem.num_vehicles} vehicles")
                solution_found = False
                temp_problem.num_vehicles += 1
        self.local_search_operator.max_no_improvement = old_num_no_improvement
        self.local_search_operator.max_iterations = old_num_iterations
        return temp_problem.num_vehicles
    
    def _check_if_done(self, generation: int, no_improvement_count: int, start_time: float) -> bool:
        """Check if stopping criteria are met."""
        if self.max_generations is not None and generation >= self.max_generations:
            return True
        if self.max_no_improvement is not None and no_improvement_count >= self.max_no_improvement:
            return True
        if self.max_time_seconds is not None and (time.time() - start_time) >= self.max_time_seconds:
            return True
        return False
