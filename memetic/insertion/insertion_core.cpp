// insertion_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

namespace py = pybind11;

bool is_feasible_route(
    const std::vector<int> &route,
    const py::array_t<double> &distance_matrix,
    const py::array_t<double> &time_windows,
    const py::array_t<double> &service_times,
    const py::array_t<double> &demands,
    double vehicle_capacity,
    const std::unordered_map<int, int> &delivery_to_pickup,
    const std::unordered_map<int, int> &pickup_to_delivery)
{
    if (route.size() < 2 || route.front() != 0 || route.back() != 0)
    {
        return false;
    }

    auto dist = distance_matrix.unchecked<2>();
    auto tw = time_windows.unchecked<2>();
    auto service = service_times.unchecked<1>();
    auto dem = demands.unchecked<1>();

    const auto n_nodes_py = tw.shape(0);
    size_t n_nodes = static_cast<size_t>(n_nodes_py);

    double load = 0.0;
    double current_time = 0.0;

    std::vector<char> seen(n_nodes, 0);

    for (size_t i = 0; i < route.size() - 1; ++i)
    {
        int from_node = route[i];
        int to_node = route[i + 1];

        // Check 1: Precedence with delivery_to_pickup
        auto delivery_it = delivery_to_pickup.find(to_node);
        if (delivery_it != delivery_to_pickup.end())
        {
            int pickup = delivery_it->second;
            // pickup must have been seen before visiting this delivery
            if (pickup < 0 || static_cast<size_t>(pickup) >= n_nodes || !seen[pickup])
            {
                return false;
            }
        }
        // Check 2: Node must be pickup, delivery, or depot
        else if (to_node != 0 && pickup_to_delivery.find(to_node) == pickup_to_delivery.end())
        {
            return false;
        }

        // Check 3: Time window 
        current_time += dist(from_node, to_node);
        double tw_start = tw(to_node, 0);
        double tw_end = tw(to_node, 1);
        if (current_time > tw_end)
        {
            return false;
        }
        if (current_time < tw_start)
        {
            current_time = tw_start;
        }
        current_time += service(to_node);

        // Check 4: Duplicate visit
        if (seen[to_node])
        {
            return false;
        }

        // Check 5: Depot in middle
        if (to_node == 0 && i + 1 < route.size() - 1)
        {
            return false;
        }

        // Check 6: Capacity
        load += dem(to_node);
        if (load < 0.0 || load > vehicle_capacity)
        {
            return false;
        }

        // mark visited
        seen[to_node] = 1;
    }

    return true;
}

std::tuple<int, int, int, double, std::vector<int>> find_best_position_for_request(
    py::array_t<double> distance_matrix,
    py::array_t<double> time_windows,
    py::array_t<double> service_times,
    py::array_t<double> demands,
    double vehicle_capacity,
    const std::vector<std::vector<int>> &routes,
    int pickup,
    int delivery,
    const std::vector<int> &not_allowed_vehicle_idxs,
    int force_vehicle_idx,
    const std::map<int, int> &delivery_to_pickup_map,
    const std::map<int, int> &pickup_to_delivery_map)
{
    auto dist = distance_matrix.unchecked<2>();

    std::unordered_map<int, int> delivery_to_pickup(delivery_to_pickup_map.begin(), delivery_to_pickup_map.end());
    std::unordered_map<int, int> pickup_to_delivery(pickup_to_delivery_map.begin(), pickup_to_delivery_map.end());

    std::unordered_set<int> not_allowed(not_allowed_vehicle_idxs.begin(), not_allowed_vehicle_idxs.end());

    double best_increase = std::numeric_limits<double>::infinity();
    int best_route_idx = -1;
    int best_pickup_pos = -1;
    int best_delivery_pos = -1;
    std::vector<int> best_new_route;

    for (size_t route_idx = 0; route_idx < routes.size(); ++route_idx)
    {
        int r_idx = static_cast<int>(route_idx);

        if (not_allowed.find(r_idx) != not_allowed.end())
        {
            continue;
        }
        if (force_vehicle_idx != -1 && r_idx != force_vehicle_idx)
        {
            continue;
        }

        const auto &route = routes[route_idx];

        if (route.size() == 2 && route[0] == 0 && route[1] == 0)
        {
            std::vector<int> new_route = {0, pickup, delivery, 0};

            if (!is_feasible_route(new_route, distance_matrix, time_windows,
                                   service_times, demands, vehicle_capacity,
                                   delivery_to_pickup, pickup_to_delivery))
            {
                continue;
            }

            double cost = dist(0, pickup) + dist(pickup, delivery) + dist(delivery, 0);
            cost *= 10.0; 

            if (cost < best_increase)
            {
                best_increase = cost;
                best_route_idx = r_idx;
                best_pickup_pos = 1;
                best_delivery_pos = 2;
                best_new_route = std::move(new_route);
            }
            continue;
        }

        double old_cost = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i)
        {
            old_cost += dist(route[i], route[i + 1]);
        }

        for (size_t pickup_pos = 1; pickup_pos < route.size(); ++pickup_pos)
        {
            for (size_t delivery_pos = pickup_pos; delivery_pos < route.size(); ++delivery_pos)
            {
                double removed = 0.0;
                double added = 0.0;

                if (delivery_pos > pickup_pos)
                {
                    removed = dist(route[pickup_pos - 1], route[pickup_pos]) + dist(route[delivery_pos - 1], route[delivery_pos]);
                    added = dist(route[pickup_pos - 1], pickup) + dist(pickup, route[pickup_pos])
                          + dist(route[delivery_pos - 1], delivery) + dist(delivery, route[delivery_pos]);
                }
                else
                {
                    removed = dist(route[pickup_pos - 1], route[pickup_pos]);
                    added = dist(route[pickup_pos - 1], pickup) + dist(pickup, delivery) + dist(delivery, route[pickup_pos]);
                }

                double increase = added - removed;

                if (!(increase < best_increase))
                {
                    continue;
                }

                std::vector<int> new_route;
                new_route.reserve(route.size() + 2);

                for (size_t i = 0; i < pickup_pos; ++i)
                {
                    new_route.push_back(route[i]);
                }
                new_route.push_back(pickup);
                for (size_t i = pickup_pos; i < delivery_pos; ++i)
                {
                    new_route.push_back(route[i]);
                }
                new_route.push_back(delivery);
                for (size_t i = delivery_pos; i < route.size(); ++i)
                {
                    new_route.push_back(route[i]);
                }

                if (!is_feasible_route(new_route, distance_matrix, time_windows,
                                       service_times, demands, vehicle_capacity,
                                       delivery_to_pickup, pickup_to_delivery))
                {
                    continue;
                }

                if (increase < best_increase)
                {
                    best_increase = increase;
                    best_route_idx = r_idx;
                    best_pickup_pos = static_cast<int>(pickup_pos);
                    best_delivery_pos = static_cast<int>(delivery_pos);
                    best_new_route = std::move(new_route);
                }
            }
        }
    }

    return std::make_tuple(best_route_idx, best_pickup_pos, best_delivery_pos, best_increase, best_new_route);
}

PYBIND11_MODULE(insertion_core, m)
{
    m.doc() = "Optimized insertion core (C++ / pybind11)";
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
          py::arg("force_vehicle_idx") = -1,
          py::arg("delivery_to_pickup"),
          py::arg("pickup_to_delivery"));
}
