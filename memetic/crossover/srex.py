from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution

from memetic.crossover.base_crossover import BaseCrossover

from memetic.local_search.naive_local_search import NaiveLocalSearch
from memetic.solution_operators.reinsert import ReinsertOperator
from memetic.insertion.greedy_insertion import GreedyInsertion

class SREXCrossover(BaseCrossover):
    """Sequential Route Exchange Crossover (SREX) for PDPTW."""
    def __init__(self, n_total: int = 10, n_cross: int = 2):
        """Initialize SREXCrossover.

        Args:
            n_total (int, optional): Total number of local search iterations. Defaults to 10.
            n_cross (int, optional): Number of crossover operations to perform. Defaults to 2.
        """
        super().__init__()
        operators = [ReinsertOperator()]
        self.local_search = NaiveLocalSearch(operators=operators, max_no_improvement=1, max_iterations=10)
        self.insertion = GreedyInsertion()
        self.n_total = n_total
        self.n_cross = n_cross

    def crossover(self, problem: PDPTWProblem, parent1: PDPTWSolution, parent2: PDPTWSolution) -> list[PDPTWSolution]:
        """Perform crossover between two parent solutions.

        Args:
            problem (PDPTWProblem): The PDPTW problem instance.
            parent1 (PDPTWSolution): The first parent solution.
            parent2 (PDPTWSolution): The second parent solution.

        Returns:
            list[PDPTWSolution]: A list of offspring solutions generated from the crossover.
        """
        return self._srex_overall(problem, parent1, parent2)
    
    def _get_arcs(self, solution: PDPTWSolution) -> set:
        """Extract all arcs (edges) from solution routes."""
        arcs = set()
        for route in solution.routes:
            for i in range(len(route) - 1):
                arcs.add((route[i], route[i+1]))
        return arcs
    
    def _srex_overall(self, problem: PDPTWProblem, parent1: PDPTWSolution, parent2: PDPTWSolution) -> list[PDPTWSolution]:
        """Overall SREX procedure to generate offspring from two parents."""
        # Step 1: Execute local search n_total times
        pairs = []
        for _ in range(self.n_total):
            p1 = parent1.clone()
            p2 = parent2.clone()
            p1, _ = self.local_search.search(problem, p1)
            p2, _ = self.local_search.search(problem, p2)
            pairs.append((p1, p2))
        
        # Step 2: Eliminate duplicates and apply arc-based filter
        unique_pairs = []
        seen = set()
        
        arcs_parent_A = self._get_arcs(parent1)
        arcs_parent_B = self._get_arcs(parent2)
        arcs_B_not_A_parent = arcs_parent_B - arcs_parent_A
        threshold = len(arcs_B_not_A_parent) / 2
        
        for p1, p2 in pairs:
            key = (p1.hashed_encoding, p2.hashed_encoding)
            if key in seen:
                continue
            
            # # Filter: arcs in SB but not in SA
            # arcs_SA = self._get_arcs(p1)
            # arcs_SB = self._get_arcs(p2)
            # arcs_SB_not_SA = arcs_SB - arcs_SA
            
            # if len(arcs_SB_not_SA) > threshold:
            #     continue
                
            seen.add(key)
            unique_pairs.append((p1, p2))
        
        # Step 3: Select best n_cross pairs by |V_A\B|
        unique_pairs.sort(key=lambda pair: len(set(pair[0].visited_nodes) - set(pair[1].visited_nodes)))
        selected_pairs = unique_pairs[:self.n_cross]
        
        # Step 4: Generate offspring for each selected pair
        offspring_solutions = []
        for s_a, s_b in selected_pairs:
            offspring = self._srex_sub(problem, parent1, parent2, s_a, s_b)
            offspring_solutions.extend(offspring)  
        
        # Step 5: Return all offspring (or parent if none feasible)
        return offspring_solutions if offspring_solutions else [parent1.clone()]
        
    def _route_equals(self, route1, route2):
        """Compare routes by content"""
        return len(route1) == len(route2) and all(a == b for a, b in zip(route1, route2))

    def _srex_sub(self, problem: PDPTWProblem, parent1: PDPTWSolution, parent2: PDPTWSolution, 
                s_a: PDPTWSolution, s_b: PDPTWSolution) -> list[PDPTWSolution]:
        """Perform the SREX crossover for a selected pair of solutions."""
        
        offspring1 = parent1.clone()
        offspring2 = parent1.clone()
        
        V_B_not_A = set(s_b.visited_nodes) - set(s_a.visited_nodes)
        
        # Step 1 Type I: Remove routes that exist in SA
        routes_to_remove = s_a.routes
        offspring1.routes = [route.copy() for route in parent1.routes 
                            if not any(self._route_equals(route, r) for r in routes_to_remove)]
        # Then eject nodes V_B\A
        offspring1.routes = [[node for node in route if node not in V_B_not_A] 
                            for route in offspring1.routes]
        offspring1.routes = [route for route in offspring1.routes if route]
        
        # Step 1 Type II: Remove routes that exist in SA
        offspring2.routes = [route.copy() for route in parent1.routes 
                            if not any(self._route_equals(route, r) for r in routes_to_remove)]
        
        
        # Step 2 Type I: Insert routes SB
        offspring1.routes.extend([route.copy() for route in s_b.routes])
        
        # Step 2 Type II: Insert filtered routes from SB
        for route in s_b.routes:
            filtered_route = [node for node in route if node not in V_B_not_A]
            if filtered_route:
                offspring2.routes.append(filtered_route)
        
        # Step 3: Insert unserved requests
        offspring1 = self.insertion.insert(problem, offspring1)
        offspring2 = self.insertion.insert(problem, offspring2)
        
        # Step 4: Return all feasible solutions
        result = []
        if offspring1.is_feasible:
            result.append(offspring1)
        if offspring2.is_feasible:
            result.append(offspring2)
        
        return result