from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    """A node in the PDPTW problem, representing a depot, pickup, or delivery location."""
    index: int 
    """A unique identifier for the node. Used to reference the node in routes and requests."""
    x: float 
    """The x coordinate of the node in a 2D plane."""
    y: float 
    """The y coordinate of the node in a 2D plane."""
    demand: int 
    """The demand at the node. Positive for pickups, negative for deliveries."""
    time_window: tuple[int, int] 
    """A tuple (earliest, latest) representing the time window for service at the node."""
    service_time: int 
    """The time required to service the node."""
    pickup_index: int  
    """An index referencing the associated pickup node. 0 if this node is a pickup or depot."""
    delivery_index: int  
    """An index referencing the associated delivery node. 0 if this node is a delivery or depot."""
    def __str__(self) -> str:
        return (f"Node(Index: {self.index}, Demand: {self.demand}, "
                f"Time Window: {self.time_window}, Service Time: {self.service_time}, "
                f"Pickup Index: {self.pickup_index}, Delivery Index: {self.delivery_index})")
    
@dataclass
class Request:
    """A pickup and delivery request in the PDPTW problem."""
    pickup: Node 
    """The pickup node of the request."""
    delivery: Node 
    """The delivery node of the request."""
    
    @property
    def demand(self) -> int:
        """The demand of the request."""
        return self.pickup.demand  # positive demand for pickup, negative for delivery
    
    @property
    def distance(self) -> float:
        """The Euclidean distance between the pickup and delivery nodes."""
        dx = self.pickup.x - self.delivery.x
        dy = self.pickup.y - self.delivery.y
        return np.hypot(dx, dy)

    def __str__(self) -> str:
        return f"Request(Pickup: {self.pickup.index}, Delivery: {self.delivery.index}, Demand: {self.demand})"

@dataclass
class PDPTWProblem:
    """A Pickup and Delivery Problem with Time Windows (PDPTW) instance."""
    num_vehicles: int 
    """The total number of vehicles available for routing."""
    vehicle_capacity: int
    """The maximum capacity of each vehicle."""
    nodes: list[Node]
    """A list of all nodes in the problem, including depot, pickups, and deliveries."""
    
    def __post_init__(self):
        n = len(self.nodes)

        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = self.nodes[i].x - self.nodes[j].x
                dy = self.nodes[i].y - self.nodes[j].y
                self.distance_matrix[i, j] = np.hypot(dx, dy)
                
        self.nodes_dict = {node.index: node for node in self.nodes}
        self.demands = np.array([node.demand for node in self.nodes])
        self.time_windows = np.array([node.time_window for node in self.nodes])
        self.service_times = np.array([node.service_time for node in self.nodes])
        self.pickups_deliveries = self._compute_pickups_deliveries()
        self.requests = [Request(self.nodes_dict[p], self.nodes_dict[d]) for p, d in self.pickups_deliveries]
        
        self.pickup_to_delivery = {p: d for p, d in self.pickups_deliveries}
        self.delivery_to_pickup = {d: p for p, d in self.pickups_deliveries}
    
    @property
    def num_locations(self) -> int:
        """The number of customer locations (excluding the depot). """
        return len(self.nodes) - 1  # ohne Depot
    
    @property
    def num_requests(self) -> int:
        """The number of pickup and delivery requests in the problem."""
        return len(self.pickups_deliveries)
    
    def _compute_pickups_deliveries(self) -> list[tuple[int, int]]:
        pairs = []
        seen = set()
        for node in self.nodes:
            if node.pickup_index == 0 and node.delivery_index != 0: 
                pairs.append((node.index, node.delivery_index))
        return pairs

    def is_pickup(self, node_index: int) -> bool:
        """Returns True if the node is a pickup node."""
        node = self.nodes_dict[node_index]
        return node.pickup_index == 0 and node.delivery_index != 0
    
    def is_delivery(self, node_index: int) -> bool:
        """Returns True if the node is a delivery node."""
        node = self.nodes_dict[node_index]
        return node.pickup_index != 0 and node.delivery_index == 0
    
    def get_pair(self, node_index: int) -> int:
        """Returns the request pair for a given pickup or delivery node."""
        node = self.nodes_dict[node_index]
        if self.is_pickup(node_index):
            return (node.index, node.delivery_index)
        elif self.is_delivery(node_index):
            return (node.pickup_index, node.index)
        else:
            raise ValueError(f"Node {node_index} is neither pickup nor delivery.")
    
    @property
    def data(self) -> dict:
        """Returns a dictionary representation of the problem data.
            
            num_vehicles: int \n
            vehicle_capacity: int \n
            depot: int (always 0) \n
            distance_matrix: np.array \n
            demands: np.array \n
            time_windows: np.array \n
            service_times: np.array \n
            pickups_deliveries: list of (pickup_index, delivery_index) tuples
        """
        return {
            'num_vehicles': self.num_vehicles,
            'vehicle_capacity': self.vehicle_capacity,
            'depot': 0,
            'distance_matrix': self.distance_matrix,
            'demands': self.demands,
            'time_windows': self.time_windows,
            'service_times': self.service_times,
            'pickups_deliveries': self.pickups_deliveries
        }
