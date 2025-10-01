// insertion_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// Returns: (route_idx, pickup_pos, delivery_pos, cost_increase, new_route)
std::tuple<int, int, int, double, std::vector<int>> find_best_position_for_request(
    py::array_t<double> distance_matrix,
    py::array_t<double> time_windows,
    py::array_t<double> service_times,
    py::array_t<double> demands,
    double vehicle_capacity,
    const std::vector<std::vector<int>>& routes,
    int pickup,
    int delivery,
    const std::vector<int>& not_allowed_vehicle_idxs,
    int force_vehicle_idx
) {
    auto dist = distance_matrix.unchecked<2>();
    auto tw = time_windows.unchecked<2>();
    auto service = service_times.unchecked<1>();
    auto dem = demands.unchecked<1>();
    
    double best_increase = std::numeric_limits<double>::infinity();
    int best_route_idx = -1;
    int best_pickup_pos = -1;
    int best_delivery_pos = -1;
    std::vector<int> best_new_route;
    
    for (size_t route_idx = 0; route_idx < routes.size(); ++route_idx) {
        // Check constraints
        if (std::find(not_allowed_vehicle_idxs.begin(), not_allowed_vehicle_idxs.end(), route_idx) != not_allowed_vehicle_idxs.end()) {
            continue;
        }
        if (force_vehicle_idx != -1 && (int)route_idx != force_vehicle_idx) {
            continue;
        }
        
        const auto& route = routes[route_idx];
        
        // Try all pickup positions
        for (size_t pickup_pos = 1; pickup_pos < route.size(); ++pickup_pos) {
            // Try all delivery positions after pickup
            for (size_t delivery_pos = pickup_pos + 1; delivery_pos <= route.size(); ++delivery_pos) {
                
                // Build new route - exact Python logic:
                // route[:pickup_pos] + [pickup] + route[pickup_pos:delivery_pos] + [delivery] + route[delivery_pos:]
                std::vector<int> new_route;
                new_route.reserve(route.size() + 2);
                
                // route[:pickup_pos]
                for (size_t i = 0; i < pickup_pos; ++i) {
                    new_route.push_back(route[i]);
                }
                
                // [pickup]
                new_route.push_back(pickup);
                
                // route[pickup_pos:delivery_pos]
                for (size_t i = pickup_pos; i < delivery_pos; ++i) {
                    new_route.push_back(route[i]);
                }
                
                // [delivery]
                new_route.push_back(delivery);
                
                // route[delivery_pos:]
                for (size_t i = delivery_pos; i < route.size(); ++i) {
                    new_route.push_back(route[i]);
                }
                
                // Feasibility check
                double load = 0.0;
                double time = 0.0;
                bool feasible = true;
                
                for (size_t i = 0; i < new_route.size() - 1; ++i) {
                    int from = new_route[i];
                    int to = new_route[i + 1];
                    
                    // Skip end depot checks
                    if (i == new_route.size() - 2) {
                        break;
                    }
                    
                    // Check depot in middle
                    if (to == 0) {
                        feasible = false;
                        break;
                    }
                    
                    time += dist(from, to);
                    
                    if (time > tw(to, 1)) {
                        feasible = false;
                        break;
                    }
                    if (time < tw(to, 0)) time = tw(to, 0);
                    time += service(to);
                    
                    load += dem(to);
                    if (load > vehicle_capacity || load < 0) {
                        feasible = false;
                        break;
                    }
                }
                
                if (feasible) {
                    // Calculate cost increase
                    double old_cost = 0.0;
                    for (size_t i = 0; i < route.size() - 1; ++i) {
                        old_cost += dist(route[i], route[i+1]);
                    }
                    
                    double new_cost = 0.0;
                    for (size_t i = 0; i < new_route.size() - 1; ++i) {
                        new_cost += dist(new_route[i], new_route[i+1]);
                    }
                    
                    double increase = new_cost - old_cost;
                    if (route.size() == 2 && route[0] == 0 && route[1] == 0) {
                        increase *= 1.5; 
                    }
                    
                    if (increase < best_increase) {
                        best_increase = increase;
                        best_route_idx = route_idx;
                        best_pickup_pos = pickup_pos;
                        best_delivery_pos = delivery_pos;
                        best_new_route = new_route;
                    }
                }
            }
        }
    }
    
    return std::make_tuple(best_route_idx, best_pickup_pos, best_delivery_pos, best_increase, best_new_route);
}

PYBIND11_MODULE(insertion_core, m) {
    m.def("find_best_position_for_request", &find_best_position_for_request,
          py::arg("distance_matrix"),
          py::arg("time_windows"),
          py::arg("service_times"),
          py::arg("demands"),
          py::arg("vehicle_capacity"),
          py::arg("routes"),
          py::arg("pickup"),
          py::arg("delivery"),
          py::arg("not_allowed_vehicle_idxs") = std::vector<int>(),
          py::arg("force_vehicle_idx") = -1);
}