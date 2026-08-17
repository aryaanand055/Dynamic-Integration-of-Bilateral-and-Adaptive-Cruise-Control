"""
road.py: Contains the Road class for the traffic simulation.
"""

class Road:
    def __init__(self, length, x, y, dir_x, dir_y):
        self.length = length
        self.x = x
        self.y = y
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.cars_on_road = []

    def get_length(self):
        return self.length

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_dir_x(self):
        return self.dir_x

    def get_dir_y(self):
        return self.dir_y

    def enter_road(self, car):
        self.cars_on_road.insert(0, car)

    def exit_road(self, car):
        if self.cars_on_road and self.cars_on_road[-1] == car:
            self.cars_on_road.pop()

    def get_cars_on_road(self):
        return self.cars_on_road
