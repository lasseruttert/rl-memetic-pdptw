from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from collections import Counter
import numpy as np

def compute_edge_embedding(problem: PDPTWProblem, solution: PDPTWSolution) -> np.ndarray:
    """
    Calculates a binary edge frequency embedding for a PDPTW solution.
    
    Returns:
        embedding: Binary vector of length n*(n-1)/2 (only upper triangular matrix)
    """
    n = len(problem.nodes)
    edge_dim = n * (n - 1) // 2
    embedding = np.zeros(edge_dim, dtype=np.float32)
    
    def edge_to_index(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + j - i - 1
    
    for route in solution.routes:
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            idx = edge_to_index(i, j)
            embedding[idx] = 1.0
    
    return embedding


class PopulationCentroid:
    """
    Maintains the centroid of solution embeddings in the population.
    """
    def __init__(self, embedding_dim: int):
        self.full_dim = embedding_dim
        self.centroid = np.zeros(embedding_dim, dtype=np.float32)
        self.population_size = 0
        self.embeddings = []
        self.variance_mask = None
        self.reduced_centroid = None
        
    def add_solution(self, embedding: np.ndarray):
        """Adds a new solution to the population."""
        self.centroid = (self.centroid * self.population_size + embedding) / (self.population_size + 1)
        self.population_size += 1
        self.embeddings.append(embedding)
        self._update_variance_mask()
    
    def remove_solution(self, embedding: np.ndarray):
        """Removes a solution from the population."""
        if self.population_size <= 1:
            raise ValueError("Cannot remove from empty or single-element population")
        self.centroid = (self.centroid * self.population_size - embedding) / (self.population_size - 1)
        self.population_size -= 1
        if embedding.tolist() in [e.tolist() for e in self.embeddings]:
            self.embeddings.remove(embedding)
        self._update_variance_mask()
    
    def replace_solution(self, old_embedding: np.ndarray, new_embedding: np.ndarray):
        """Replaces one solution with another."""
        self.centroid = self.centroid + (new_embedding - old_embedding) / self.population_size
        # Update embeddings list
        for i, emb in enumerate(self.embeddings):
            if np.array_equal(emb, old_embedding):
                self.embeddings[i] = new_embedding
                break
        self._update_variance_mask()
    
    def _update_variance_mask(self):
        """Calculates variance mask and reduced centroid."""
        if len(self.embeddings) > 1:
            embeddings_array = np.array(self.embeddings)
            variance = np.var(embeddings_array, axis=0)
            self.variance_mask = variance > 0
            self.reduced_centroid = self.centroid[self.variance_mask]
    
    def compute_diversity(self, embedding: np.ndarray, use_reduction: bool = True) -> float:
        """
        Calculates the diversity of a solution relative to the population centroid.
        
        Args:
            embedding: The embedding of the solution to evaluate
            use_reduction: If True, only use dimensions with variance

        Returns:
            diversity: L2 distance to the centroid (higher = more diverse)
        """
        if use_reduction and self.variance_mask is not None:
            reduced_embedding = embedding[self.variance_mask]
            return np.linalg.norm(reduced_embedding - self.reduced_centroid)
        else:
            return np.linalg.norm(embedding - self.centroid)
    
    def get_statistics(self) -> dict:
        """Returns statistics about the population."""
        stats = {
            'population_size': self.population_size,
            'full_dimensions': self.full_dim,
        }
        
        if self.variance_mask is not None:
            stats['active_dimensions'] = np.sum(self.variance_mask)
            stats['sparsity'] = 1.0 - (np.sum(self.variance_mask) / self.full_dim)
            
        if len(self.embeddings) > 1:
            embeddings_array = np.array(self.embeddings)
            # Pairwise diversities
            diversities = []
            for i in range(len(self.embeddings)):
                for j in range(i+1, len(self.embeddings)):
                    if self.variance_mask is not None:
                        dist = np.linalg.norm(
                            embeddings_array[i][self.variance_mask] - 
                            embeddings_array[j][self.variance_mask]
                        )
                    else:
                        dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
                    diversities.append(dist)
            
            stats['mean_pairwise_diversity'] = np.mean(diversities)
            stats['min_pairwise_diversity'] = np.min(diversities)
            stats['max_pairwise_diversity'] = np.max(diversities)
        
        return stats


class SparseCentroid:
    """
    Sparse representation using only the utilized edges.
    """
    def __init__(self):
        self.edge_counts = Counter()
        self.population_size = 0
        self.all_edges = []
    
    def add_solution(self, edges: set):
        """Adds a solution (as an edge set) to the population."""
        self.edge_counts.update(edges)
        self.all_edges.append(edges)
        self.population_size += 1
    
    def remove_solution(self, edges: set):
        """Removes a solution."""
        if self.population_size <= 1:
            raise ValueError("Cannot remove from empty population")
        for edge in edges:
            self.edge_counts[edge] -= 1
            if self.edge_counts[edge] == 0:
                del self.edge_counts[edge]
        self.all_edges.remove(edges)
        self.population_size -= 1
    
    def replace_solution(self, old_edges: set, new_edges: set):
        """Replaces one solution with another."""
        # Remove old edges
        for edge in old_edges:
            self.edge_counts[edge] -= 1
            if self.edge_counts[edge] == 0:
                del self.edge_counts[edge]
        # Add new edges 
        self.edge_counts.update(new_edges)
        # Update list
        for i, edges in enumerate(self.all_edges):
            if edges == old_edges:
                self.all_edges[i] = new_edges
                break
    
    def compute_diversity(self, edges: set) -> float:
        """
        Calculates diversity based on the rarity of the utilized edges.
        
        Higher value = more diverse (uses rare edges)
        """
        if not edges:
            return 0.0
        
        diversity = 0.0
        for edge in edges:
            frequency = self.edge_counts.get(edge, 0) / self.population_size
            diversity += (1.0 - frequency)
        
        return diversity / len(edges)
    
    def compute_jaccard_diversity(self, edges: set) -> float:
        """
        Calculates the average Jaccard distance to all solutions.
        """
        if not self.all_edges:
            return 0.0
        
        total_distance = 0.0
        for other_edges in self.all_edges:
            intersection = len(edges & other_edges)
            union = len(edges | other_edges)
            jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0.0
            total_distance += jaccard_dist
        
        return total_distance / len(self.all_edges)


def compute_sparse_edges(solution: PDPTWSolution, include_depot: bool = False) -> set:
    """
    Extracts the edge set of a solution.
    
    Args:
        solution: PDPTWSolution instance
        include_depot: If False, edges involving the depot (node 0) are excluded.
        
    Returns:
        Set of edges as (i, j) tuples with i < j
    """
    edges = set()
    for route in solution.routes:
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            
            if not include_depot and (i == 0 or j == 0):
                continue
            
            edge = tuple(sorted([i, j]))
            edges.add(edge)
    
    return edges
