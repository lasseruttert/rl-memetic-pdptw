import random
import numpy as np

def manhattan_dist(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))

def generate_pickup_deliveries(data, num_pairs):
    candidates = list(range(1, len(data["distance_matrix"])))
    random.shuffle(candidates)

    pickup_deliveries = []
    used = set()

    for _ in range(num_pairs):
        pickup = None
        while candidates:
            candidate = candidates.pop()
            if candidate not in used:
                pickup = candidate
                used.add(pickup)
                break
        if pickup is None:
            break

        delivery = None
        while candidates:
            candidate = candidates.pop()
            if candidate not in used and candidate != pickup:
                delivery = candidate
                used.add(delivery)
                break
        if delivery is None:
            break

        pickup_deliveries.append([pickup, delivery])

    return pickup_deliveries

def create_pdp_instance(grid_size = 200, num_locations = 20, num_pairs = 10, num_vehicles = 4, distance_function = manhattan_dist, demand = False, max_demand = 10, timewindows = False, seed = 1):
    random.seed(seed)
    locations = []
    for i in range(num_locations):
        x = random.randint(0, grid_size)
        y = random.randint(0, grid_size)
        coord = (x, y)
        locations.append(coord)

    data = {}
    data["tag"] = "PDP"
    data["locations"] = locations
    distance_matrix = np.zeros((num_locations,num_locations), dtype=int)
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i,j] = distance_function(locations[i], locations[j])
    data["distance_matrix"] = distance_matrix.tolist()

    data["pickups_deliveries"] = generate_pickup_deliveries(data, num_pairs)
    data["num_vehicles"] = num_vehicles
    data["depot"] = 0

    if demand:
        data["tag"] = "C" + data["tag"]
        demands = [0] * num_locations
        for pickup, delivery in data["pickups_deliveries"]:
            demand_value = random.randint(1, max_demand)
            demands[pickup] = demand_value
            demands[delivery] = -demand_value
            data["demands"] = demands

    if timewindows:
        data["tag"] = data["tag"] + "TW"
        size = len(distance_matrix)
        time_matrix = [
            [
                int(distance_matrix[i][j] * random.uniform(0.8, 1.2))
                for j in range(size)
            ]
            for i in range(size)
        ]

        data["time_matrix"] = time_matrix

        max_possible_distance = distance_function((0, 0), (grid_size, grid_size))
        max_timewindow = max_possible_distance * 5
        total_time = max_timewindow * num_pairs * 5

        data["time_windows"] = [(0, total_time)] * num_locations  # placeholder

        time_windows = [(0, total_time)] * num_locations

        for pickup, delivery in data["pickups_deliveries"]:
            travel_time = data["distance_matrix"][pickup][delivery]

            earliest_pickup = random.randint(0, total_time - travel_time - 10)
            latest_pickup = earliest_pickup + random.randint(max_timewindow // 3, max_timewindow)

            min_delivery_start = earliest_pickup + travel_time
            max_delivery_start = min(latest_pickup + travel_time, total_time - 1)
            earliest_delivery = random.randint(min_delivery_start, max_delivery_start)
            latest_delivery = earliest_delivery + random.randint(max_timewindow // 3, max_timewindow)

            latest_delivery = min(latest_delivery, total_time)

            time_windows[pickup] = (earliest_pickup, latest_pickup)
            time_windows[delivery] = (earliest_delivery, latest_delivery)

        data["time_windows"] = time_windows

    return data
