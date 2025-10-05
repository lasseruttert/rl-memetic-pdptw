"""Utility functions for RL-based local search."""

import numpy as np
from utils.pdptw_problem import PDPTWProblem
from utils.pdptw_solution import PDPTWSolution


def calculate_constraint_violations(problem: PDPTWProblem, solution: PDPTWSolution) -> dict:
    """Calculate detailed constraint violation metrics for a solution.

    Args:
        problem: PDPTW problem instance
        solution: PDPTW solution to evaluate

    Returns:
        dict with keys:
            - capacity_violations: number of capacity violations
            - capacity_violation_magnitude: total excess capacity across all violations
            - time_window_violations: number of time window violations
            - time_window_violation_magnitude: total lateness across all violations
            - precedence_violations: number of precedence violations (delivery before pickup)
            - missing_nodes: number of nodes not served
            - total_violations: total number of violations
    """
    pickup_to_delivery = problem.pickup_to_delivery
    delivery_to_pickup = problem.delivery_to_pickup
    demands = problem.demands
    vehicle_capacity = problem.vehicle_capacity
    distance_matrix = problem.distance_matrix
    time_windows = problem.time_windows
    service_times = problem.service_times
    nodes = problem.nodes

    capacity_violations = 0
    capacity_violation_magnitude = 0.0
    time_window_violations = 0
    time_window_violation_magnitude = 0.0
    precedence_violations = 0

    seen_total = set()

    for route in solution.routes:
        load = 0
        current_time = 0.0
        seen = set()

        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]

            # Check capacity
            load += demands[to_node]
            if load < 0:
                capacity_violations += 1
                capacity_violation_magnitude += abs(load)
            elif load > vehicle_capacity:
                capacity_violations += 1
                capacity_violation_magnitude += (load - vehicle_capacity)

            # Check time windows
            travel_time = distance_matrix[from_node, to_node]
            current_time += travel_time

            tw_start, tw_end = time_windows[to_node]
            if current_time < tw_start:
                current_time = tw_start
            if current_time > tw_end:
                time_window_violations += 1
                time_window_violation_magnitude += (current_time - tw_end)

            current_time += service_times[to_node]

            # Check precedence
            if to_node in delivery_to_pickup:
                pickup = delivery_to_pickup[to_node]
                if pickup not in seen:
                    precedence_violations += 1

            seen.add(to_node)

        seen_total.update(seen)

    # Check missing nodes
    all_nodes = set(node.index for node in nodes)
    missing_nodes = len(all_nodes - seen_total)

    total_violations = (
        capacity_violations +
        time_window_violations +
        precedence_violations +
        missing_nodes
    )

    return {
        'capacity_violations': capacity_violations,
        'capacity_violation_magnitude': capacity_violation_magnitude,
        'time_window_violations': time_window_violations,
        'time_window_violation_magnitude': time_window_violation_magnitude,
        'precedence_violations': precedence_violations,
        'missing_nodes': missing_nodes,
        'total_violations': total_violations
    }


def extract_solution_features(problem: PDPTWProblem, solution: PDPTWSolution) -> np.ndarray:
    """Extract feature vector from a solution for RL state representation.

    Features include:
    - Problem features: num_requests, vehicle_capacity, avg_distance, time_window_tightness
    - Solution features: num_routes, num_customers_served, total_distance, route statistics
    - Constraint violation features: counts and magnitudes

    Args:
        problem: PDPTW problem instance
        solution: PDPTW solution

    Returns:
        np.ndarray: Feature vector
    """
    # Problem features
    num_requests = problem.num_requests
    vehicle_capacity = problem.vehicle_capacity
    num_vehicles = problem.num_vehicles
    avg_distance = np.mean(problem.distance_matrix[problem.distance_matrix > 0])

    # Time window tightness: average ratio of time window span to total time horizon
    time_window_spans = problem.time_windows[:, 1] - problem.time_windows[:, 0]
    max_time_horizon = np.max(problem.time_windows[:, 1])
    avg_tw_tightness = np.mean(time_window_spans) / max_time_horizon if max_time_horizon > 0 else 0

    # Solution features
    num_routes = solution.num_vehicles_used
    num_customers_served = solution.num_customers_served
    total_distance = solution.total_distance

    # Route statistics
    route_lengths = [len(r) - 2 for r in solution.routes if len(r) > 2]  # excluding depot
    avg_route_length = np.mean(route_lengths) if route_lengths else 0
    max_route_length = np.max(route_lengths) if route_lengths else 0
    min_route_length = np.min(route_lengths) if route_lengths else 0
    std_route_length = np.std(route_lengths) if route_lengths else 0

    # Route distance statistics
    route_distances = [solution.route_lengths[i] for i in range(len(solution.routes)) if len(solution.routes[i]) > 2]
    avg_route_distance = np.mean(route_distances) if route_distances else 0
    max_route_distance = np.max(route_distances) if route_distances else 0
    std_route_distance = np.std(route_distances) if route_distances else 0

    # Constraint violations
    violations = calculate_constraint_violations(problem, solution)

    # Normalized features
    features = np.array([
        # Problem features 
        num_requests,  
        vehicle_capacity,  
        num_vehicles,  
        avg_distance,  
        avg_tw_tightness,  

        # Solution features (normalized)
        num_routes / num_vehicles if num_vehicles > 0 else 0,
        num_customers_served / (num_requests * 2) if num_requests > 0 else 0,  # *2 for pickup+delivery
        total_distance / (avg_distance * num_requests * 2) if (avg_distance * num_requests) > 0 else 0,

        # Route statistics (normalized)
        avg_route_length / num_requests,
        max_route_length / num_requests,
        min_route_length / num_requests,
        std_route_length / num_requests,
        avg_route_distance / problem.distance_baseline,
        max_route_distance / problem.distance_baseline,
        std_route_distance / problem.distance_baseline,

        # Constraint violations (normalized by num_requests)
        violations['capacity_violations'] / num_requests if num_requests > 0 else 0,
        violations['capacity_violation_magnitude'] / (vehicle_capacity * num_requests) if (vehicle_capacity * num_requests) > 0 else 0,
        violations['time_window_violations'] / num_requests if num_requests > 0 else 0,
        violations['time_window_violation_magnitude'] / max_time_horizon if max_time_horizon > 0 else 0,
        violations['precedence_violations'] / num_requests if num_requests > 0 else 0,
        violations['missing_nodes'] / (num_requests * 2) if num_requests > 0 else 0,
        violations['total_violations'] / (num_requests * 2) if num_requests > 0 else 0,
    ], dtype=np.float32)

    return features
