# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

cpdef bint is_feasible_insertion_fast(object problem, list route):
    cdef int route_len = len(route)
    
    if route_len < 2 or route[0] != 0 or route[route_len - 1] != 0:
        return False
    
    cdef object distance_matrix = problem.distance_matrix
    cdef object demands = problem.demands
    cdef object time_windows = problem.time_windows
    cdef object service_times = problem.service_times
    cdef double vehicle_capacity = problem.vehicle_capacity
    cdef dict delivery_to_pickup = problem.delivery_to_pickup
    
    cdef double load = 0.0
    cdef double current_time = 0.0
    cdef set seen = set()  # Zurück zu cdef set
    
    cdef int i, from_node, to_node
    cdef double tw_start, tw_end, demand
    
    for i in range(route_len - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        if i == route_len - 2:
            break
        
        if to_node == 0:
            return False
        
        if to_node in delivery_to_pickup:
            if delivery_to_pickup[to_node] not in seen:
                return False
        
        if to_node in seen:
            return False
        
        current_time += distance_matrix[from_node, to_node]
        tw_start = time_windows[to_node, 0]
        tw_end = time_windows[to_node, 1]
        
        if current_time > tw_end:
            return False
        
        if current_time < tw_start:
            current_time = tw_start
        
        current_time += service_times[to_node]
        
        demand = demands[to_node]
        load += demand
        
        if load < 0 or load > vehicle_capacity:
            return False
        
        seen.add(to_node)
    
    return True