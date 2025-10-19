from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random

class SwapWithinOperator(BaseOperator):
    def __init__(self, max_attempts: None = None, single_route: bool = False, type: str = "random"):
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        self.type = type
        self.name = f"SwapWithin-{self.type}-{'Single' if self.single_route else 'All'}-Max{self.max_attempts}"

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()

        if self.single_route:
            routes = [random.choice(new_solution.routes)]
        else:
            routes = new_solution.routes

        if self.type == "random":
            attempts = 0
            for route in routes:
                if self.max_attempts is not None and attempts >= self.max_attempts:
                    break
                if len(route) >= 6:  # Ensure there are at least two nodes to swap
                    # select two requests in the route and swap each pickup and delivery
                    nodes_in_route = route[1:-1]  # Exclude depot
                    requests = set(problem.get_pair(node) for node in nodes_in_route)
                    request1, request2 = random.sample(requests, 2)
                    pickup1, delivery1 = request1
                    pickup2, delivery2 = request2
                    idx1_pickup = route.index(pickup1)
                    idx1_delivery = route.index(delivery1)
                    idx2_pickup = route.index(pickup2)
                    idx2_delivery = route.index(delivery2)

                    # Swap the positions
                    route[idx1_pickup], route[idx2_pickup] = route[idx2_pickup], route[idx1_pickup]
                    route[idx1_delivery], route[idx2_delivery] = route[idx2_delivery], route[idx1_delivery]

            new_solution._clear_cache()

        elif self.type == "best":
            dist_matrix = problem.distance_matrix

            for route in routes:
                if len(route) < 6:  # Need at least 2 requests
                    continue

                nodes_in_route = route[1:-1]  # Exclude depot
                requests = list(set(problem.get_pair(node) for node in nodes_in_route))

                if len(requests) < 2:
                    continue

                best_improvement = 0
                best_swap = None

                # Try all pairs of requests
                for i in range(len(requests)):
                    for j in range(i + 1, len(requests)):
                        request1 = requests[i]
                        request2 = requests[j]

                        pickup1, delivery1 = request1
                        pickup2, delivery2 = request2

                        idx1_pickup = route.index(pickup1)
                        idx1_delivery = route.index(delivery1)
                        idx2_pickup = route.index(pickup2)
                        idx2_delivery = route.index(delivery2)

                        # Collect old edges 
                        old_edges = set()
                        swap_indices = {idx1_pickup, idx1_delivery, idx2_pickup, idx2_delivery}

                        for idx in swap_indices:
                            if idx > 0:
                                old_edges.add((route[idx - 1], route[idx]))
                            if idx < len(route) - 1:
                                old_edges.add((route[idx], route[idx + 1]))

                        # Calculate old cost
                        old_cost = sum(dist_matrix[edge[0], edge[1]] for edge in old_edges)

                        # Perform the swap temporarily
                        route[idx1_pickup], route[idx2_pickup] = route[idx2_pickup], route[idx1_pickup]
                        route[idx1_delivery], route[idx2_delivery] = route[idx2_delivery], route[idx1_delivery]

                        # Collect new edges
                        new_edges = set()
                        for idx in swap_indices:
                            if idx > 0:
                                new_edges.add((route[idx - 1], route[idx]))
                            if idx < len(route) - 1:
                                new_edges.add((route[idx], route[idx + 1]))

                        # Calculate new cost
                        new_cost = sum(dist_matrix[edge[0], edge[1]] for edge in new_edges)

                        # Revert the swap
                        route[idx1_pickup], route[idx2_pickup] = route[idx2_pickup], route[idx1_pickup]
                        route[idx1_delivery], route[idx2_delivery] = route[idx2_delivery], route[idx1_delivery]

                        improvement = old_cost - new_cost

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (idx1_pickup, idx1_delivery, idx2_pickup, idx2_delivery)

                # Apply best swap if found
                if best_swap is not None:
                    idx1_pickup, idx1_delivery, idx2_pickup, idx2_delivery = best_swap
                    pickup1 = route[idx1_pickup]
                    delivery1 = route[idx1_delivery]
                    pickup2 = route[idx2_pickup]
                    delivery2 = route[idx2_delivery]

                    route[idx1_pickup], route[idx2_pickup] = route[idx2_pickup], route[idx1_pickup]
                    route[idx1_delivery], route[idx2_delivery] = route[idx2_delivery], route[idx1_delivery]

            new_solution._clear_cache()

        return new_solution