from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution
from memetic.solution_operators.base_operator import BaseOperator
import random


class SwapBetweenOperator(BaseOperator):
    def __init__(self, type: str = "random"):
        super().__init__()
        self.type = type

    def apply(self, problem: PDPTWProblem, solution: PDPTWSolution) -> PDPTWSolution:
        new_solution = solution.clone()
        possible_routes = [route for route in new_solution.routes if len(route) >= 4]
        if len(possible_routes) < 2:
            return new_solution  # Not enough routes to swap between
        if self.type == "random":
            route1, route2 = random.sample(possible_routes, 2)
            requests_in_route1 = set(problem.get_pair(node) for node in route1[1:-1])
            requests_in_route2 = set(problem.get_pair(node) for node in route2[1:-1])
            if not requests_in_route1 or not requests_in_route2:
                return new_solution  # No requests to swap
            request1 = random.choice(list(requests_in_route1))
            request2 = random.choice(list(requests_in_route2))
            
            pickup1, delivery1 = request1
            pickup2, delivery2 = request2
            
            idx1_pickup = route1.index(pickup1)
            idx1_delivery = route1.index(delivery1)
            idx2_pickup = route2.index(pickup2)
            idx2_delivery = route2.index(delivery2)
            
            # Swap the positions
            route1[idx1_pickup], route2[idx2_pickup] = route2[idx2_pickup], route1[idx1_pickup]
            route1[idx1_delivery], route2[idx2_delivery] = route2[idx2_delivery], route1[idx1_delivery]
            
            new_solution._clear_cache()
        elif self.type == "best":
            route1 = random.choice(possible_routes)
            requests_in_route1 = set(problem.get_pair(node) for node in route1[1:-1])
            if not requests_in_route1:
                return new_solution  # No requests to swap
            request1 = random.choice(list(requests_in_route1))
            pickup1, delivery1 = request1
            idx1_pickup = route1.index(pickup1)
            idx1_delivery = route1.index(delivery1)

            best_improvement = 0
            best_route2 = None
            best_request2 = None
            best_idx2_pickup = None
            best_idx2_delivery = None

            dist_matrix = problem.distance_matrix

            for route2 in possible_routes:
                if route2 == route1:
                    continue
                requests_in_route2 = set(problem.get_pair(node) for node in route2[1:-1])
                for request2 in requests_in_route2:
                    pickup2, delivery2 = request2
                    idx2_pickup = route2.index(pickup2)
                    idx2_delivery = route2.index(delivery2)

                    # Calculate delta cost for route1
                    # Remove edges with pickup1 and delivery1
                    cost_removed_r1 = 0.0
                    if idx1_delivery == idx1_pickup + 1:
                        # Adjacent: remove (prev, pickup1), (pickup1, delivery1), (delivery1, next)
                        cost_removed_r1 += dist_matrix[route1[idx1_pickup - 1], pickup1]
                        cost_removed_r1 += dist_matrix[pickup1, delivery1]
                        cost_removed_r1 += dist_matrix[delivery1, route1[idx1_delivery + 1]]
                    else:
                        # Non-adjacent: remove edges around both nodes
                        cost_removed_r1 += dist_matrix[route1[idx1_pickup - 1], pickup1]
                        cost_removed_r1 += dist_matrix[pickup1, route1[idx1_pickup + 1]]
                        cost_removed_r1 += dist_matrix[route1[idx1_delivery - 1], delivery1]
                        cost_removed_r1 += dist_matrix[delivery1, route1[idx1_delivery + 1]]

                    # Add edges with pickup2 and delivery2
                    cost_added_r1 = 0.0
                    if idx1_delivery == idx1_pickup + 1:
                        # Adjacent: add (prev, pickup2), (pickup2, delivery2), (delivery2, next)
                        cost_added_r1 += dist_matrix[route1[idx1_pickup - 1], pickup2]
                        cost_added_r1 += dist_matrix[pickup2, delivery2]
                        cost_added_r1 += dist_matrix[delivery2, route1[idx1_delivery + 1]]
                    else:
                        # Non-adjacent: add edges around both nodes
                        cost_added_r1 += dist_matrix[route1[idx1_pickup - 1], pickup2]
                        cost_added_r1 += dist_matrix[pickup2, route1[idx1_pickup + 1]]
                        cost_added_r1 += dist_matrix[route1[idx1_delivery - 1], delivery2]
                        cost_added_r1 += dist_matrix[delivery2, route1[idx1_delivery + 1]]

                    # Calculate delta cost for route2
                    # Remove edges with pickup2 and delivery2
                    cost_removed_r2 = 0.0
                    if idx2_delivery == idx2_pickup + 1:
                        # Adjacent: remove (prev, pickup2), (pickup2, delivery2), (delivery2, next)
                        cost_removed_r2 += dist_matrix[route2[idx2_pickup - 1], pickup2]
                        cost_removed_r2 += dist_matrix[pickup2, delivery2]
                        cost_removed_r2 += dist_matrix[delivery2, route2[idx2_delivery + 1]]
                    else:
                        # Non-adjacent: remove edges around both nodes
                        cost_removed_r2 += dist_matrix[route2[idx2_pickup - 1], pickup2]
                        cost_removed_r2 += dist_matrix[pickup2, route2[idx2_pickup + 1]]
                        cost_removed_r2 += dist_matrix[route2[idx2_delivery - 1], delivery2]
                        cost_removed_r2 += dist_matrix[delivery2, route2[idx2_delivery + 1]]

                    # Add edges with pickup1 and delivery1
                    cost_added_r2 = 0.0
                    if idx2_delivery == idx2_pickup + 1:
                        # Adjacent: add (prev, pickup1), (pickup1, delivery1), (delivery1, next)
                        cost_added_r2 += dist_matrix[route2[idx2_pickup - 1], pickup1]
                        cost_added_r2 += dist_matrix[pickup1, delivery1]
                        cost_added_r2 += dist_matrix[delivery1, route2[idx2_delivery + 1]]
                    else:
                        # Non-adjacent: add edges around both nodes
                        cost_added_r2 += dist_matrix[route2[idx2_pickup - 1], pickup1]
                        cost_added_r2 += dist_matrix[pickup1, route2[idx2_pickup + 1]]
                        cost_added_r2 += dist_matrix[route2[idx2_delivery - 1], delivery1]
                        cost_added_r2 += dist_matrix[delivery1, route2[idx2_delivery + 1]]

                    # Total improvement (positive means cost reduction)
                    improvement = (cost_removed_r1 + cost_removed_r2) - (cost_added_r1 + cost_added_r2)

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_route2 = route2
                        best_request2 = request2
                        best_idx2_pickup = idx2_pickup
                        best_idx2_delivery = idx2_delivery

            # Apply the best swap if found
            if best_route2 is not None:
                pickup2, delivery2 = best_request2
                route1[idx1_pickup], best_route2[best_idx2_pickup] = best_route2[best_idx2_pickup], route1[idx1_pickup]
                route1[idx1_delivery], best_route2[best_idx2_delivery] = best_route2[best_idx2_delivery], route1[idx1_delivery]
                new_solution._clear_cache()
        
        return new_solution