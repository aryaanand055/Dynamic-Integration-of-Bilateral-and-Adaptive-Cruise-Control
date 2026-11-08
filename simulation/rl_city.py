import math
import numpy as np
from simulation.city import City

class RLCity(City):
    """
    Extended City class to support direct acceleration control via RL.
    Cars placed in 'RL-A' mode will have their acceleration determined 
    by their `rl_action` attribute, bypassing the standard heuristics.
    """
    def driver_decision(self):
        # Save previous acceleration for RL-controlled cars to calculate jerk properly
        for car in self.cars:
            if getattr(car, 'is_rl', False):
                car._prev_acc = car.acceleration

        # Let the standard heuristic compute accelerations for all cars
        super().driver_decision()

        # Overwrite the acceleration for RL-controlled cars
        dt = self.dt
        for car in self.cars:
            if getattr(car, 'is_rl', False) and hasattr(car, 'rl_action'):
                # Direct acceleration command from the RL model
                target_acc = float(car.rl_action)

                # Enforce physical constraints: min/max acceleration
                target_acc = max(self.min_a, min(self.max_a, target_acc))
                
                # Enforce physical constraints: jerk (rate of change of acceleration)
                last_acc = getattr(car, '_prev_acc', car.acceleration)
                jerk = (target_acc - last_acc) / dt
                max_jerk = 5.0
                
                if jerk > max_jerk:
                    acc = last_acc + max_jerk * dt
                elif jerk < -max_jerk:
                    acc = last_acc - max_jerk * dt
                else:
                    acc = target_acc
                
                # If car is at rest, physical holding brake gives zero net acceleration
                if car.velocity <= 1e-4 and acc <= 0.0:
                    acc = 0.0

                car.acceleration = acc

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False
            car._prev_acc = 0.0

    def handle_collisions(self):
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False

        road_length = self.roads[0].length if self.roads else 1000
        active_cars = [c for c in self.cars if not getattr(c, 'has_exited', False)]
        sorted_cars = sorted(active_cars, key=lambda c: c.pos)
        for i, car in enumerate(sorted_cars):
            next_car = sorted_cars[(i + 1) % len(sorted_cars)]
            if car == next_car:
                continue
            # car is ahead (smaller pos), next_car is behind (larger pos)
            gap_raw = (next_car.pos - car.pos) % road_length
            if gap_raw <= car.length:
                self.collided_this_step = True
                next_car.just_collided = True
                next_car.just_collided_front = True
                car.just_collided = True
                car.just_collided_rear = True

        super().handle_collisions()

