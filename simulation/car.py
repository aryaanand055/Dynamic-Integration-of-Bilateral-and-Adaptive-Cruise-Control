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
        self.target_speed = None
        self.is_reckless = False
        self.is_human = False
        self.driver_type = 'ALGORITHM'
        self.human_profile_name = None
        self.original_color = color

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
        # Physical vehicle kinematics with zero-velocity holding boundary
        if self.velocity <= 1e-4 and self.acceleration <= 0.0:
            # Stationary vehicle with brakes applied on level road:
            # Static friction balances braking force -> zero acceleration, zero motion
            self.acceleration = 0.0
            self.velocity = 0.0
            displacement = 0.0
        elif self.acceleration < 0.0 and (self.velocity + self.acceleration * dt) <= 0.0:
            # Stops cleanly within this timestep at t_stop
            t_stop = self.velocity / (-self.acceleration)
            displacement = 0.5 * self.velocity * t_stop
            self.velocity = 0.0
            self.acceleration = 0.0
        else:
            displacement = self.velocity * dt + 0.5 * self.acceleration * dt**2
            self.velocity += self.acceleration * dt

        self.pos -= displacement


        road_length = self.current_road.length if getattr(self, 'current_road', None) is not None else 1000
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
