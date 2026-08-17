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
                # rl_action is already a raw acceleration value in [min_a, max_a]
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
                
                car.acceleration = acc
