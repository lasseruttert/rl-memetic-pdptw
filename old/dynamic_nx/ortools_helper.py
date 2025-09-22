class ORToolsHelper:
    """
    A class which provides utilities to be able to use the OR Tools Solver with our dynamic PDP environment
    """
    def __init__(self, env):
        self.env = env

    def get_fixed_routes(self):
        """
        Used to fixate visited locations which are part of a request and the next location for each vehicle in the next call of the OR Tools Solver
        :return: A 2D List which contains a list with fixed locations for each vehicle
        """
        fixed_routes = [[] for vehicle in self.env.get_vehicles()]
        for index, vehicle in enumerate(self.env.get_vehicles()):
            if vehicle.get_next_location() is None or vehicle.get_next_location() == self.env.get_depot():
                continue
            visited = vehicle.get_visited()
            next_location = vehicle.get_next_location()
            fixed_routes[index].append(self.env.get_depot().get_index())
            for location in visited:
                fixed_routes[index].append(location.get_index())
            if fixed_routes[-1] != next_location.get_index():
                fixed_routes[index].append(next_location.get_index())
        return fixed_routes

    def trim_routes(self, routes):
        """
        Removes entries in the routes given by the OR Tools Solver which were used to model previously visited locations
        :param routes: A 2D Array of aRoutes to be trimmed
        :return: A 2D Array of trimmed Routes
        """
        if not routes: return [[] for vehicle in self.env.get_vehicles()]
        vehicles = self.env.get_vehicles()
        for index, route in enumerate(routes):
            if route:
                route.pop(0)
                visited_indices = {location.get_index() for location in vehicles[index].get_visited()}
                routes[index] = [node for node in route if node not in visited_indices]
        return routes
    
    def normalize_time_windows(self, data):
        current_time_step = data.get("current_timestep", 0)
        time_windows = data.get("time_windows", None)
        for index, (earliest, latest) in enumerate(time_windows):
            time_windows[index] = earliest - current_time_step, latest - current_time_step
        data["time_windows"] = time_windows
        return data