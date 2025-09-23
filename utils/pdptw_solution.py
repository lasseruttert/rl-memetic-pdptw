from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class PDPTWSolution:
    """A solution to a PDPTW problem instance."""
    routes: list[list[int]]
    """A list of routes, where each route is a list of node indices starting and ending with the depot (index 0)."""
    total_distance: float
    """The total distance traveled by all vehicles in the solution."""
    
    def __post_init__(self):
        self.encoding = str(self.routes)
        self.hashed_encoding = hash(self.encoding)
        self.num_vehicles_used = sum(1 for r in self.routes if len(r) > 1)
        self.num_customers_served = sum(len(r) - 2 for r in self.routes if len(r) > 2)  # ohne Start/Ende-Depot

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
            route_lines.append(f"  Vehicle {i}: {' -> '.join(map(str, route))}")
        return header + "\n" + "\n".join(route_lines)