import math
import numpy as np
from simulation.city import City
from simulation.supervisor import VehicleSupervisor, FSMState

class RLCity(City):
    """
    Extended City class supporting modular FSM supervisory control:
    - CAR_FOLLOWING: 6D RL model regulates gap and closing speed.
    - FREE_CRUISE: Dedicated cruise controller regulates vehicle to v_des on clear roads.
    """
    def driver_decision(self):
        # Save previous acceleration for RL-controlled cars to calculate jerk properly
        for car in self.cars:
            if getattr(car, 'is_rl', False):
                car._prev_acc = car.acceleration

        # Let the standard heuristic compute accelerations for all cars
        super().driver_decision()

        # Overwrite acceleration for RL-controlled cars via FSM supervisor
        dt = self.dt
        road_length = self.roads[0].length if self.roads else 1000
        car_states = self.get_perceived_car_states() if hasattr(self, 'get_perceived_car_states') else [(c.pos, c.velocity) for c in self.cars]

        for idx, car in enumerate(self.cars):
            if getattr(car, 'is_rl', False) and not getattr(car, 'is_human', False) and hasattr(car, 'rl_action'):
                # 1. Update FSM Supervisor state
                cars_same_road = [c for c in self.cars if c.current_road == car.current_road and c != car and not getattr(c, 'has_exited', False)]
                def gap_to(other):
                    gap = (car.pos - other.pos) % road_length
                    return gap if gap > 0 else float('inf')
                front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None

                if front_car:
                    front_idx = self.cars.index(front_car)
                    front_pos, _ = car_states[front_idx]
                    ego_pos, _ = car_states[idx]
                    front_gap = (ego_pos - front_pos - front_car.length) % road_length
                else:
                    front_gap = float('inf')

                if not hasattr(car, 'supervisor') or car.supervisor is None:
                    car.supervisor = VehicleSupervisor()

                fsm_state = car.supervisor.update(front_car, front_gap, dt)
                car.fsm_state = fsm_state.value

                # 2. Select controller based on FSM state
                if fsm_state == FSMState.FREE_CRUISE:
                    target_acc = car.supervisor.compute_free_cruise_acceleration(
                        car=car,
                        v_des=self.v_des,
                        dt=dt,
                        min_a=self.min_a,
                        max_a=self.max_a,
                        kc=self.kc,
                        max_jerk=5.0
                    )
                else:
                    # CAR_FOLLOWING: Direct acceleration command from the RL model
                    target_acc = float(car.rl_action)
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
            if not hasattr(car, 'supervisor') or car.supervisor is None:
                car.supervisor = VehicleSupervisor()
            car.fsm_state = FSMState.FREE_CRUISE.value

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

