from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator

import random

class ShiftOperator(BaseOperator):
    """Shift operator: relocate a segment within a route (similar to Or-opt).

    Extracts a segment of nodes and reinserts it at a different position in the same route.
    For PDPTW, checks precedence constraints after shifting.
    """

    def __init__(self, max_attempts: int = 5, single_route: bool = False, type: str = "random", segment_length: int = 4, max_shift_distance: int = 2):
        """Initialize shift operator.

        Args:
            max_attempts: Maximum number of shift attempts
            single_route: If True, only shift in one random route; if False, try all routes
            type: "random" for random shifts, "best" for exhaustive search with best improvement
            segment_length: Length of segment to shift
            max_shift_distance: Maximum distance (in positions) to shift the segment
        """
        super().__init__()
        self.max_attempts = max_attempts
        self.single_route = single_route
        self.type = type
        self.segment_length = segment_length
        self.max_shift_distance = max_shift_distance
        self.name = f"Shift-{self.type}-{'Single' if self.single_route else 'All'}-Max{self.max_attempts}-Seg{self.segment_length}-Dist{self.max_shift_distance}"

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()

        # Select routes to operate on
        if self.single_route:
            routes = [random.choice(new_solution.routes)]
        else:
            routes = new_solution.routes

        # Depot node (assumed to be 0)
        depot = 0

        if self.type == "random":
            attempts = 0
            for route in routes:
                if self.max_attempts is not None and attempts >= self.max_attempts:
                    break

                # Need at least segment_length + 2 nodes (excluding depots at both ends)
                if len(route) < self.segment_length + 2:
                    continue

                # Choose random segment start position (exclude depot at start)
                max_start = len(route) - self.segment_length - 1
                if max_start <= 1:
                    continue

                segment_start = random.randint(1, max_start)
                segment_end = segment_start + self.segment_length

                # Extract segment
                segment = route[segment_start:segment_end]

                # Calculate valid shift positions
                # Can shift left or right by max_shift_distance, but not into depot positions
                min_pos = max(1, segment_start - self.max_shift_distance)
                max_pos = min(len(route) - self.segment_length - 1, segment_start + self.max_shift_distance)

                # Choose random new position (different from current)
                valid_positions = [p for p in range(min_pos, max_pos + 1) if p != segment_start]
                if not valid_positions:
                    continue

                new_start = random.choice(valid_positions)

                # Create new route with shifted segment
                if new_start < segment_start:
                    # Shift left
                    new_route = (route[:new_start] +
                                segment +
                                route[new_start:segment_start] +
                                route[segment_end:])
                else:
                    # Shift right (adjust for removed segment)
                    insert_pos = new_start + self.segment_length
                    new_route = (route[:segment_start] +
                                route[segment_end:insert_pos] +
                                segment +
                                route[insert_pos:])

                # Check precedence constraints and depot preservation
                # if (self._is_feasible_precedence(problem, new_route) and
                #     new_route[0] == depot and new_route[-1] == depot):
                if new_route[0] == depot and new_route[-1] == depot:
                    route[:] = new_route
                    attempts += 1

            new_solution._clear_cache()

        elif self.type == "best":
            dist_matrix = problem.distance_matrix

            for route in routes:
                if len(route) < self.segment_length + 2:
                    continue

                best_improvement = 0
                best_shift = None

                # Try all segment positions
                for segment_start in range(1, len(route) - self.segment_length):
                    segment_end = segment_start + self.segment_length
                    segment = route[segment_start:segment_end]

                    # Calculate old cost (edges around segment)
                    old_cost = 0.0
                    # Edge before segment
                    old_cost += dist_matrix[route[segment_start - 1], route[segment_start]]
                    # Edges within segment
                    for i in range(segment_start, segment_end - 1):
                        old_cost += dist_matrix[route[i], route[i + 1]]
                    # Edge after segment
                    if segment_end < len(route):
                        old_cost += dist_matrix[route[segment_end - 1], route[segment_end]]
                    # Direct edge if segment is removed
                    if segment_end < len(route):
                        old_cost += dist_matrix[route[segment_start - 1], route[segment_end]]

                    # Try shift positions
                    min_pos = max(1, segment_start - self.max_shift_distance)
                    max_pos = min(len(route) - self.segment_length - 1, segment_start + self.max_shift_distance)

                    for new_start in range(min_pos, max_pos + 1):
                        if new_start == segment_start:
                            continue

                        # Create shifted route
                        if new_start < segment_start:
                            new_route = (route[:new_start] +
                                        segment +
                                        route[new_start:segment_start] +
                                        route[segment_end:])
                        else:
                            insert_pos = new_start + self.segment_length
                            new_route = (route[:segment_start] +
                                        route[segment_end:insert_pos] +
                                        segment +
                                        route[insert_pos:])

                        # Check precedence and depot preservation
                        if not (self._is_feasible_precedence(problem, new_route) and
                               new_route[0] == depot and new_route[-1] == depot):
                            continue

                        # Calculate new cost
                        if new_start < segment_start:
                            # Shifted left
                            new_cost = 0.0
                            new_cost += dist_matrix[new_route[new_start - 1], segment[0]]
                            for i in range(len(segment) - 1):
                                new_cost += dist_matrix[segment[i], segment[i + 1]]
                            new_cost += dist_matrix[segment[-1], new_route[new_start + self.segment_length]]
                            # Connection where segment was removed
                            old_segment_end = segment_start + self.segment_length
                            new_cost += dist_matrix[new_route[old_segment_end - 1], new_route[old_segment_end]]
                        else:
                            # Shifted right
                            insert_pos = new_start + self.segment_length
                            new_cost = 0.0
                            # Connection before insertion
                            new_cost += dist_matrix[new_route[insert_pos - len(segment) - 1], segment[0]]
                            for i in range(len(segment) - 1):
                                new_cost += dist_matrix[segment[i], segment[i + 1]]
                            # Connection after insertion
                            if insert_pos < len(new_route):
                                new_cost += dist_matrix[segment[-1], new_route[insert_pos]]
                            # Connection where segment was removed
                            new_cost += dist_matrix[new_route[segment_start - 1], new_route[segment_start]]

                        improvement = old_cost - new_cost

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_shift = new_route

                # Apply best shift if found
                if best_shift is not None:
                    route[:] = best_shift

            new_solution._clear_cache()

        return new_solution

    def _is_feasible_precedence(self, problem: PDPTWProblem, route: list) -> bool:
        """Check if all pickups come before their deliveries."""
        position = {}
        for idx, node in enumerate(route):
            position[node] = idx

        for pickup, delivery in problem.pickups_deliveries:
            if pickup in position and delivery in position:
                if position[pickup] >= position[delivery]:
                    return False
        return True
