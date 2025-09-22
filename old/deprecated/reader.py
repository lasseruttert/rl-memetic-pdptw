import math

def li_liam_to_or(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_vehicles, vehicle_capacity, _ = map(int, lines[0].split())

    nodes = []
    for line in lines[1:]:
        parts = list(map(int, line.strip().split()))
        node = {
            'index': parts[0],
            'x': parts[1],
            'y': parts[2],
            'demand': parts[3],
            'time_window': (parts[4], parts[5]),
            'service_time': parts[6],
            'pickup_index': parts[7],
            'delivery_index': parts[8]
        }
        nodes.append(node)

    def euclidean_distance(a, b):
        return int(math.hypot(a['x'] - b['x'], a['y'] - b['y']))

    distance_matrix = [
        [euclidean_distance(a, b) for b in nodes]
        for a in nodes
    ]

    pickups_deliveries = []
    index_to_node = {node['index']: idx for idx, node in enumerate(nodes)}
    for node in nodes:
        if node['delivery_index'] > 0:
            pickup_idx = index_to_node[node['index']]
            delivery_idx = index_to_node[node['delivery_index']]
            pickups_deliveries.append([pickup_idx, delivery_idx])

    data = {
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'depot': 0,
        'distance_matrix': distance_matrix,
        'demands': [node['demand'] for node in nodes],
        'time_windows': [node['time_window'] for node in nodes],
        'service_times': [node['service_time'] for node in nodes],
        'pickups_deliveries': pickups_deliveries
    }

    return data
