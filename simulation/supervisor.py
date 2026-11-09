"""
supervisor.py: Finite-State Supervisory Machine (FSM) for autonomous vehicle control.
Decouples Free-Flow Cruising (velocity regulation) from Car-Following (spacing regulation).
"""

from enum import Enum
import math


class FSMState(Enum):
    FREE_CRUISE = "FREE_CRUISE"
    CAR_FOLLOWING = "CAR_FOLLOWING"


class VehicleSupervisor:
    """
    Supervisory state machine with radar hysteresis.
    - FREE_CRUISE: No vehicle ahead within radar range. Regulates speed to v_des.
    - CAR_FOLLOWING: Target vehicle acquired. Regulates gap and relative speed via 6D RL/BCC.
    """
    RADAR_ACQUIRE_THRESHOLD = 65.0  # Meters (enter CAR_FOLLOWING)
    RADAR_RELEASE_THRESHOLD = 70.0  # Meters (exit to FREE_CRUISE)

    def __init__(self, initial_state=FSMState.FREE_CRUISE):
        self.state = initial_state
        self.time_in_state = 0.0
        self.mode_switches = 0
        self.last_transition_time = 0.0

    def update(self, front_car, front_gap, dt):
        """
        Updates the supervisory state with hysteresis to eliminate boundary chattering.
        """
        self.time_in_state += dt
        prev_state = self.state

        if front_car is None or front_gap >= self.RADAR_RELEASE_THRESHOLD:
            self.state = FSMState.FREE_CRUISE
        elif front_gap <= self.RADAR_ACQUIRE_THRESHOLD:
            self.state = FSMState.CAR_FOLLOWING
        # If 65.0 < front_gap < 70.0, retain existing state (hysteresis band)

        if self.state != prev_state:
            self.mode_switches += 1
            self.time_in_state = 0.0

        return self.state

    def compute_free_cruise_acceleration(self, car, v_des, dt, min_a=-5.0, max_a=3.0, kc=0.4, max_jerk=5.0):
        """
        Smooth, jerk-limited free cruise velocity regulator.
        Locks vehicle to target speed v_des when no lead car is present.
        """
        # Linear speed tracking error
        target_acc = kc * (v_des - car.velocity)
        target_acc = max(min_a, min(max_a, target_acc))

        # Enforce rate-of-change (jerk) constraints
        last_acc = getattr(car, '_prev_acc', car.acceleration)
        jerk = (target_acc - last_acc) / max(1e-4, dt)

        if jerk > max_jerk:
            acc = last_acc + max_jerk * dt
        elif jerk < -max_jerk:
            acc = last_acc - max_jerk * dt
        else:
            acc = target_acc

        # Automotive zero-drift auto-hold at standstill
        if car.velocity <= 1e-4 and acc <= 0.0:
            acc = 0.0

        return acc

