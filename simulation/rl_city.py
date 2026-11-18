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
                    raw_acc = float(car.rl_action)
                    raw_acc = max(self.min_a, min(self.max_a, raw_acc))

                    # Actuator anti-dither deadband for micro-chatter around zero
                    if abs(raw_acc) < 0.035:
                        raw_acc = 0.0

                    # Physical first-order powertrain/brake actuator dynamics filter (tau ~= 0.15s, alpha = 0.40)
                    # Prevents discrete-time Nyquist limit-cycle chattering while maintaining AEB responsiveness
                    last_acc = getattr(car, '_prev_acc', car.acceleration)
                    if raw_acc < -2.5 and raw_acc < last_acc:
                        target_acc = raw_acc
                    else:
                        prev_cmd = getattr(car, '_filtered_cmd', raw_acc)
                        target_acc = 0.40 * raw_acc + 0.60 * prev_cmd
                    car._filtered_cmd = target_acc

                # Enforce physical constraints: jerk (rate of change of acceleration)
                last_acc = getattr(car, '_prev_acc', car.acceleration)
                jerk = (target_acc - last_acc) / dt
                
                # Dynamic AEB jerk ceiling: 25.0 m/s³ strictly for genuine emergency collision avoidance; 5.0 m/s³ for comfortable driving
                road_length = self.roads[0].length if self.roads else 1000
                front_gap = (front_car.pos - car.pos - front_car.length) % road_length if front_car else road_length
                closing_speed = car.velocity - front_car.velocity if front_car else 0.0
                is_emergency = front_car and (front_gap < 3.5 or (closing_speed > 0.5 and front_gap / max(closing_speed, 0.1) < 1.2))
                max_jerk = 25.0 if (is_emergency and target_acc < -1.5 and target_acc < last_acc) else 5.0
                
                if jerk > max_jerk:
                    acc = last_acc + max_jerk * dt
                elif jerk < -max_jerk:
                    acc = last_acc - max_jerk * dt
                else:
                    acc = target_acc
                
                # If car is at rest, physical holding brake gives zero net acceleration
                if (car.velocity <= 1e-4 and acc <= 0.0) or (car.velocity <= 0.02 and front_car and front_car.velocity <= 0.02):
                    acc = 0.0
                    car.velocity = 0.0

                car.acceleration = acc

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False
            car._prev_acc = 0.0
            car._filtered_cmd = 0.0
            if not hasattr(car, 'supervisor') or car.supervisor is None:
                car.supervisor = VehicleSupervisor()
            car.fsm_state = FSMState.FREE_CRUISE.value

    def handle_collisions(self):
        super().handle_collisions()

