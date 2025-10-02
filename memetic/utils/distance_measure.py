from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution


class DistanceMeasure:
    def __init__(self):
        self.solution_to_arcs_cache = {}

    def edge_distance(self, solution1: PDPTWSolution, solution2: PDPTWSolution) -> float:
        """Calculates the normalized edge distance between two solutions.
        
        Uses the Jaccard distance: d = 1 - (intersection / union)
        Returns a value in [0, 1] where 0 = identical, 1 = completely different.

        Args:
            solution1 (PDPTWSolution): a PDPTW solution instance
            solution2 (PDPTWSolution): a PDPTW solution instance

        Returns:
            float: normalized edge distance in [0, 1]
        """
        arcs1 = self._get_edges(solution1)
        arcs2 = self._get_edges(solution2)
        
        # Handle edge case: both solutions are empty
        if len(arcs1) == 0 and len(arcs2) == 0:
            return 0.0
        
        # Jaccard distance: 1 - (intersection / union)
        intersection = len(arcs1.intersection(arcs2))
        union = len(arcs1.union(arcs2))
        
        return 1.0 - (intersection / union) if union > 0 else 0.0

    def _get_edges(self, solution: PDPTWSolution) -> set:
        """Extract all arcs (edges) from solution routes."""
        if solution.hashed_encoding in self.solution_to_arcs_cache:
            return self.solution_to_arcs_cache[solution.hashed_encoding]
        
        arcs = set()
        for route in solution.routes:
            for i in range(len(route) - 1):
                arcs.add((route[i], route[i+1]))
        
        self.solution_to_arcs_cache[solution.hashed_encoding] = arcs
        return arcs
    
    def node_distance(self, solution1: PDPTWSolution, solution2: PDPTWSolution) -> float:
        """Calculates the normalized node distance between two solutions.
        
        Uses the Jaccard distance: d = 1 - (intersection / union)
        Returns a value in [0, 1] where 0 = identical, 1 = completely different.

        Args:
            solution1 (PDPTWSolution): a PDPTW solution
            solution2 (PDPTWSolution): a PDPTW solution instance
        Returns:
            float: normalized node distance in [0, 1]
        """
        nodes1 = set()
        for route in solution1.routes:
            nodes1.update(route)
        
        nodes2 = set()
        for route in solution2.routes:
            nodes2.update(route)
        
        # Handle edge case: both solutions are empty
        if len(nodes1) == 0 and len(nodes2) == 0:
            return 0.0
        
        # Jaccard distance: 1 - (intersection / union)
        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    
    def clear_cache(self):
        self.solution_to_arcs_cache = {}