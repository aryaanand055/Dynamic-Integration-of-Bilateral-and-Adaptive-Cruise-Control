"""
car.py: Contains the Car class for the traffic simulation.
"""

class Car:
    def __init__(self, length, color, pos,min_dis, velocity, acceleration, current_road):
        self.length = length
        self.color = color
        self.pos = pos
        self.min_dis = min_dis
        self.velocity = velocity
        self.acceleration = acceleration
        self.current_road = current_road
        self.headway_time = 2
        self.mode = 'VEL' 
        self.energy_used = 0.0
        self.mass = 1800
        self.frontal_area = 2.2
        self.CoR = 0.2 # Coefficient of Restitution
        self.Cr = 0.015  # Rolling resistance coefficient
        self.Cd = 0.29 # Drag coefficient
        self.integration_factor = 1
        self.pos_history = [self.pos]
        self.vel_history = [self.velocity]
        self.acc_history = [self.acceleration]
        self.gap_history = []
        self.x_history = []
        self.switch_events = []
        self.weights = None
        self.target_speed = None

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_length(self):
        return self.length

    def get_color(self):
        return self.color

    def get_pos(self):
        return self.pos

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration

    def get_current_road(self):
        return self.current_road

    def get_pos_history(self):
        return self.pos_history

    def update(self, dt):
        # Invert position update for inverted mapping (move right as forward)
        # S = ut + 0.5at^2
        # v = u + at
        displacement = self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.pos -= displacement

        self.velocity += self.acceleration * dt


        road_length = self.current_road.length if hasattr(self, 'current_road') else 1000
        if self.pos < 0:
            self.pos += road_length
        elif self.pos >= road_length:
            self.pos -= road_length
        self.pos_history.append(self.pos)
        self.vel_history.append(self.velocity)
        self.acc_history.append(self.acceleration)


        # Energy calculation
        gravity = 9.8
        mass = self.mass
        frontal_area = self.frontal_area
        velocity = self.velocity
        acceleration = self.acceleration
        cr = self.Cr
        Cd = self.Cd
        air_density = 1.225
       
        F_inertia = mass * acceleration
        F_roll = cr * mass * gravity
        F_drag = 0.5 * Cd * air_density * frontal_area * velocity**2
        F_total = F_inertia + F_roll + F_drag

        Power = F_total * velocity

        Energy = Power * dt 
        Energy = Energy / 3600000  # Convert to kWh
        if Energy > 0:
            self.energy_used += Energy
