// fitness_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

namespace py = pybind11;

double calculate_penalty(
    const std::vector<std::vector<int>> &routes,
    const py::array_t<double> &distance_matrix,
    const py::array_t<double> &time_windows,
    const py::array_t<double> &service_times,
    const py::array_t<double> &demands,
    double vehicle_capacity,
    const std::map<int, int> &delivery_to_pickup_map,
    const std::map<int, int> &pickup_to_delivery_map,
    const std::vector<int> &all_node_indices,
    double distance_baseline)
{
    auto dist = distance_matrix.unchecked<2>();
    auto tw = time_windows.unchecked<2>();
    auto service = service_times.unchecked<1>();
    auto dem = demands.unchecked<1>();

    std::unordered_map<int, int> delivery_to_pickup(delivery_to_pickup_map.begin(), delivery_to_pickup_map.end());
    std::unordered_map<int, int> pickup_to_delivery(pickup_to_delivery_map.begin(), pickup_to_delivery_map.end());

    int num_violations = 0;
    std::unordered_set<int> seen_total;

    for (const auto &route : routes)
    {
        double load = 0.0;
        double current_time = 0.0;
        std::unordered_set<int> seen;

        for (size_t i = 0; i < route.size() - 1; ++i)
        {
            int from_node = route[i];
            int to_node = route[i + 1];

            // Check if the route starts at the depot
            if (i == 0 && from_node != 0)
            {
                num_violations++;
            }

            // Check if the route ends at the depot
            if (i + 1 == route.size() - 1 && to_node != 0)
            {
                num_violations++;
            }

            // Check if depot is visited in the middle of the route
            if (i + 1 < route.size() - 1 && to_node == 0)
            {
                num_violations++;
            }

            // Check if node has already been served
            if (seen.find(to_node) != seen.end())
            {
                num_violations++;
            }

            // Check if pickup happens before delivery
            auto delivery_it = delivery_to_pickup.find(to_node);
            if (delivery_it != delivery_to_pickup.end())
            {
                int pickup = delivery_it->second;
                if (seen.find(pickup) == seen.end())
                {
                    num_violations++;
                }
            }
            // Check if node is valid (depot, pickup, or delivery)
            else
            {
                if (pickup_to_delivery.find(to_node) == pickup_to_delivery.end() && to_node != 0)
                {
                    num_violations++;
                }
            }

            // Check if vehicle capacities are respected
            load += dem(to_node);
            if (load < 0.0 || load > vehicle_capacity)
            {
                num_violations++;
            }

            // Check if time windows are respected
            double travel_time = dist(from_node, to_node);
            current_time += travel_time;

            double tw_start = tw(to_node, 0);
            double tw_end = tw(to_node, 1);
            if (current_time < tw_start)
            {
                current_time = tw_start;
            }
            if (current_time > tw_end)
            {
                num_violations++;
            }
            current_time += service(to_node);

            seen.insert(to_node);
        }

        seen_total.insert(seen.begin(), seen.end());
    }

    // Check if all nodes are served
    std::unordered_set<int> all_nodes(all_node_indices.begin(), all_node_indices.end());
    if (seen_total != all_nodes)
    {
        for (int node : all_nodes)
        {
            if (seen_total.find(node) == seen_total.end())
            {
                num_violations++;
            }
        }
    }

    if (num_violations == 0)
    {
        return 0.0;
    }
    else
    {
        return num_violations * 0.05 * distance_baseline + 1.0 * distance_baseline;
    }
}

std::tuple<double, bool> calculate_fitness(
    const std::vector<std::vector<int>> &routes,
    double total_distance,
    int num_vehicles_used,
    int num_vehicles,
    const py::array_t<double> &distance_matrix,
    const py::array_t<double> &time_windows,
    const py::array_t<double> &service_times,
    const py::array_t<double> &demands,
    double vehicle_capacity,
    const std::map<int, int> &delivery_to_pickup_map,
    const std::map<int, int> &pickup_to_delivery_map,
    const std::vector<int> &all_node_indices,
    double distance_baseline)
{
    double penalty = calculate_penalty(
        routes,
        distance_matrix,
        time_windows,
        service_times,
        demands,
        vehicle_capacity,
        delivery_to_pickup_map,
        pickup_to_delivery_map,
        all_node_indices,
        distance_baseline);

    bool is_feasible = (penalty == 0.0);

    double fitness = total_distance + penalty;
    double percent_vehicles_used = static_cast<double>(num_vehicles_used) / static_cast<double>(num_vehicles);
    fitness *= (1.0 + percent_vehicles_used);

    return std::make_tuple(fitness, is_feasible);
}

PYBIND11_MODULE(fitness_core, m)
{
    m.doc() = "Optimized fitness calculation (C++ / pybind11)";

    m.def("calculate_fitness", &calculate_fitness,
          py::arg("routes"),
          py::arg("total_distance"),
          py::arg("num_vehicles_used"),
          py::arg("num_vehicles"),
          py::arg("distance_matrix"),
          py::arg("time_windows"),
          py::arg("service_times"),
          py::arg("demands"),
          py::arg("vehicle_capacity"),
          py::arg("delivery_to_pickup"),
          py::arg("pickup_to_delivery"),
          py::arg("all_node_indices"),
          py::arg("distance_baseline"));

    m.def("calculate_penalty", &calculate_penalty,
          py::arg("routes"),
          py::arg("distance_matrix"),
          py::arg("time_windows"),
          py::arg("service_times"),
          py::arg("demands"),
          py::arg("vehicle_capacity"),
          py::arg("delivery_to_pickup"),
          py::arg("pickup_to_delivery"),
          py::arg("all_node_indices"),
          py::arg("distance_baseline"));
}
