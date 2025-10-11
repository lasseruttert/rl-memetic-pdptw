def compare(fitness1, num_vehicles1, fitness2, num_vehicles2):
    # * Option 1: only care for fitness
    if fitness1 < fitness2:
        return True
    
    # * Option 2: care for num vehicles first
    # if num_vehicles1 < num_vehicles2 or (num_vehicles1 == num_vehicles2 and fitness1 < fitness2):
        # return True
    return False