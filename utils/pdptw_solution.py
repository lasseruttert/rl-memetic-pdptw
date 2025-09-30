from dataclasses import dataclass
import numpy as np
from typing import Optional
from utils.pdptw_problem import PDPTWProblem
from copy import deepcopy
@dataclass
class PDPTWSolution:
    problem: PDPTWProblem
    """A solution to a PDPTW problem instance."""
    routes: list[list[int]]
    """A list of routes, where each route is a list of node indices starting and ending with the depot (index 0)."""
    
    def __post_init__(self):
        self._total_distance = None
        self._route_lengths = None
        self._is_feasible = None
        self._encoding = None
        self._hashed_encoding = None
        self._node_to_route = None
        
    def _clear_cache(self):
        self._total_distance = None
        self._route_lengths = None
        self._is_feasible = None
        self._encoding = None
        self._hashed_encoding = None
        self._node_to_route = None

    def __len__(self):
        return len(self.routes)
    
    def __iter__(self):
        return iter(self.routes)

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the solution. """
        return {
            'routes': self.routes,
            'total_distance': self.total_distance,
            'num_vehicles_used': self.num_vehicles_used,
            'num_customers_served': self.num_customers_served,
            'encoding': self.encoding,
            'hashed_encoding': self.hashed_encoding
        }
        
    def __str__(self) -> str:
        header = f"PDPTW Solution | Distance: {self.total_distance:.2f} | Vehicles used: {self.num_vehicles_used}"
        route_lines = []
        for i, route in enumerate(self.routes):
            if len(route) > 2:
                route_lines.append(f"  Vehicle {i}: {' -> '.join(map(str, route))}")
        return header + "\n" + "\n".join(route_lines)
    
    @property
    def total_distance(self) -> float:
        if self._total_distance is None:
            if self._route_lengths is None:
                self._route_lengths = {}
            distance_matrix = self.problem.distance_matrix
            total_distance = 0.0
            for route in self.routes:
                route_distance = 0.0
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    total_distance += distance_matrix[from_node, to_node]
                    route_distance += distance_matrix[from_node, to_node]
                self._route_lengths[self.routes.index(route)] = route_distance
            self._total_distance = total_distance
        return self._total_distance
    
    @property
    def route_lengths(self) -> np.ndarray:
        if self._route_lengths is None:
            self._route_lengths = {}
            _ = self.total_distance  # This will populate _route_lengths
        return self._route_lengths
    
    @property
    def is_feasible(self) -> bool:
        if self._is_feasible is None:
            from utils.feasibility import is_feasible
            self._is_feasible = is_feasible(self.problem, self)
        return self._is_feasible
    
    @property
    def encoding(self) -> str:
        if self._encoding is None:
            self._encoding = ";".join([",".join(map(str, route)) for route in self.routes])
        return self._encoding
    
    @property
    def hashed_encoding(self) -> int:
        if self._hashed_encoding is None:
            self._hashed_encoding = hash(self.encoding)
        return self._hashed_encoding
    
    @property
    def node_to_route(self) -> dict[int, int]:
        if self._node_to_route is None:
            mapping = {}
            for route_idx, route in enumerate(self.routes):
                for node in route:
                    if self.problem.is_pickup(node) or self.problem.is_delivery(node):
                        mapping[node] = route_idx
            self._node_to_route = mapping
        return self._node_to_route
    
    @property
    def num_vehicles_used(self) -> int:
        return sum(1 for r in self.routes if len(r) > 2)
    
    @property
    def num_customers_served(self) -> int:
        return sum(len(r) - 2 for r in self.routes if len(r) > 2)
    
    def modify_routes(self, new_routes: list[list[int]]):
        """Modifies the routes of the solution and clears cached properties."""
        self.routes = new_routes
        self._clear_cache()
        
    def check_feasibility(self) -> bool:
        """Re-evaluates and returns the feasibility of the solution."""
        from utils.feasibility import is_feasible
        self._is_feasible = is_feasible(self.problem, self, use_prints=True)
        return self._is_feasible
    
    def get_served_requests(self, problem) -> list[int]:
        """Returns a list of served customer requests based on the problem instance."""
        served = set()
        seen = set()
        for route in self.routes:
            for node in route:
                if problem.is_pickup(node):
                    seen.add(node)
                elif problem.is_delivery(node):
                    pair = problem.get_pair(node)
                    if pair[0] in seen:
                        served.add(pair)
        return list(served)
    
    def get_unserved_requests(self, problem) -> list[int]:
        """Returns a list of unserved customer requests based on the problem instance."""
        served = set()
        seen = set()
        unserved = []
        for route in self.routes:
            for node in route:
                if problem.is_pickup(node):
                    seen.add(node)
                elif problem.is_delivery(node):
                    pair = problem.get_pair(node)
                    if pair[0] in seen:
                        served.add(pair)
        for request in problem.pickups_deliveries:
            if request not in served:
                unserved.append(request)
        return unserved
    
    def clone(self) -> 'PDPTWSolution':
        routes = [route[:] for route in self.routes]
        new_solution = PDPTWSolution(self.problem, routes=routes)

        return new_solution