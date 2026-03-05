import gymnasium as gym
from gymnasium import spaces
import numpy as np
from city import City
from car import Car

class TrafficControlEnv(gym.Env):
    """
    Traffic Control Environment for TD3 training.
    
    The agent outputs 5 weights for a car-following controller:
    - w1: Front gap error weight
    - w2: Back gap error weight  
    - w3: Front velocity matching weight
    - w4: Back velocity matching weight
    - w5: Target velocity weight
    
    Observation includes gap and velocity information for proper learning.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_cars=6):
        super(TrafficControlEnv, self).__init__()
        
        self.city = City()
        self.dt = 0.1
        self.max_steps = 500  # Shorter episodes for faster iteration
        self.current_step = 0
        self.num_cars = num_cars
        
        # Simulation parameters
        self.v_des = 15.0  # Target velocity (m/s)
        self.min_dis = 5.0  # Minimum safe distance
        self.reaction_time = 1.5
        self.road_length = 1000.0
        
        # Action Space: 5 weights, bounded to reasonable control gains
        # Using [0.1, 2.0] as typical PD gains are in this range
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.1, 0.0, 0.1], dtype=np.float32),
            high=np.array([2.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space: Normalized fleet statistics + string stability info
        # [min_gap_norm, avg_gap_norm, avg_vel_norm, vel_std_norm, avg_acc_norm, gap_rate, 
        #  min_front_gap_norm, min_rear_gap_norm, acc_amplification]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(9,), dtype=np.float32
        )
        
        # For reward normalization
        self.prev_min_gap = None
        self.consecutive_safe_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_min_gap = None
        self.consecutive_safe_steps = 0
        
        # Initialize simulation
        self.city.init(
            car_number=self.num_cars,
            kd=0.9, kv=0.6, kc=0.4,
            v_des=self.v_des,
            max_v=30.0, min_v=0.0,
            min_dis=self.min_dis,
            reaction_time=self.reaction_time,
            headway_time=2.0,
            max_a=3.0, min_a=-5.0,
            min_gap=2.0, dt=self.dt,
            model='ACC'
        )
        
        # Set initial weights to safe defaults (similar to ACC)
        default_weights = np.array([0.9, 0.0, 0.6, 0.0, 0.4], dtype=np.float32)
        for car in self.city.cars[1:]:  # Skip lead car
            car.set_weights(default_weights)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply weights to follower cars only (not lead car)
        for car in self.city.cars[1:]:
            car.set_weights(action)
        
        # Run simulation
        self.city.run(self.dt)
        
        # Get state
        obs = self._get_obs()
        min_gap = self._get_min_gap()
        
        # Check termination
        collision = min_gap < 0.5
        terminated = collision
        truncated = self.current_step >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(min_gap, collision)
        
        # Track safe steps
        if min_gap > 2.0:
            self.consecutive_safe_steps += 1
        else:
            self.consecutive_safe_steps = 0
        
        self.prev_min_gap = min_gap
        
        info = {
            "min_gap": min_gap,
            "avg_speed": np.mean([c.velocity for c in self.city.cars]),
            "collision": collision,
            "step": self.current_step,
            "safe_steps": self.consecutive_safe_steps
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Compute normalized observation vector with string stability info.
        Returns: [min_gap_norm, avg_gap_norm, avg_vel_norm, vel_std_norm, avg_acc_norm, gap_rate,
                  min_front_gap_norm, min_rear_gap_norm, acc_amplification]
        """
        cars = self.city.cars
        if len(cars) < 2:
            return np.zeros(9, dtype=np.float32)
        
        # Compute gaps using CORRECT formula
        gaps = self._compute_all_gaps()
        front_gaps, rear_gaps = self._compute_front_rear_gaps()
        
        if not gaps:
            return np.zeros(9, dtype=np.float32)
        
        min_gap = min(gaps)
        avg_gap = np.mean(gaps)
        desired_gap = self.min_dis + self.v_des * self.reaction_time  # ~27.5m
        
        # Front/rear gap info for BCC awareness
        min_front_gap = min(front_gaps) if front_gaps else desired_gap
        min_rear_gap = min(rear_gaps) if rear_gaps else desired_gap
        
        # Velocities
        velocities = [c.velocity for c in cars]
        avg_vel = np.mean(velocities)
        vel_std = np.std(velocities)
        
        # Accelerations and string stability metric
        accelerations = [c.acceleration for c in cars]
        avg_acc = np.mean(accelerations)
        
        # Acceleration amplification ratio (string stability indicator)
        # How much acceleration grows from front to back
        acc_amplification = 0.0
        if len(accelerations) >= 2 and abs(accelerations[0]) > 0.1:
            acc_amplification = abs(accelerations[-1]) / abs(accelerations[0]) - 1.0
        
        # Gap rate of change
        if self.prev_min_gap is not None:
            gap_rate = (min_gap - self.prev_min_gap) / self.dt
        else:
            gap_rate = 0.0
        
        # Normalize observations to roughly [-1, 1] range
        obs = np.array([
            (min_gap - desired_gap) / desired_gap,       # Normalized min gap error
            (avg_gap - desired_gap) / desired_gap,       # Normalized avg gap error
            (avg_vel - self.v_des) / self.v_des,         # Normalized velocity error
            vel_std / self.v_des,                         # Normalized velocity std
            avg_acc / 3.0,                                # Normalized acceleration
            np.clip(gap_rate / 5.0, -1, 1),              # Normalized gap rate
            (min_front_gap - desired_gap) / desired_gap, # Front gap awareness
            (min_rear_gap - desired_gap) / desired_gap,  # Rear gap awareness (BCC)
            np.clip(acc_amplification, -2, 2)            # String stability indicator
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)
    
    def _compute_front_rear_gaps(self):
        """Compute separate front and rear gaps for each follower car."""
        if not self.city.roads:
            return [], []
        
        road_len = self.city.roads[0].length
        cars = self.city.cars
        front_gaps = []
        rear_gaps = []
        
        # Sort by position (lower pos = ahead)
        sorted_cars = sorted(cars, key=lambda c: c.pos)
        
        for i in range(len(sorted_cars)):
            car = sorted_cars[i]
            car_ahead = sorted_cars[(i - 1) % len(sorted_cars)]
            car_behind = sorted_cars[(i + 1) % len(sorted_cars)]
            
            # Front gap: distance to car ahead
            front_gap = (car.pos - car_ahead.pos - car_ahead.length) % road_len
            # Rear gap: distance to car behind  
            rear_gap = (car_behind.pos - car.pos - car.length) % road_len
            
            if 0 < front_gap < road_len / 2:
                front_gaps.append(front_gap)
            if 0 < rear_gap < road_len / 2:
                rear_gaps.append(rear_gap)
        
        return front_gaps, rear_gaps

    def _compute_all_gaps(self):
        """Compute gaps between all consecutive cars using CORRECT formula."""
        if not self.city.roads:
            return []
        
        road_len = self.city.roads[0].length
        cars = self.city.cars
        gaps = []
        
        # Sort by position (lower pos = ahead)
        sorted_cars = sorted(cars, key=lambda c: c.pos)
        
        for i in range(len(sorted_cars)):
            car_ahead = sorted_cars[i]
            car_behind = sorted_cars[(i + 1) % len(sorted_cars)]
            
            # Gap from car_behind to car_ahead
            # = car_behind.pos - (car_ahead.pos + car_ahead.length)
            gap = (car_behind.pos - car_ahead.pos - car_ahead.length) % road_len
            
            # Sanity check - gap should be positive and reasonable
            if gap > 0 and gap < road_len / 2:
                gaps.append(gap)
        
        return gaps if gaps else [100.0]

    def _get_min_gap(self):
        """Get minimum gap in the fleet."""
        gaps = self._compute_all_gaps()
        return min(gaps) if gaps else 100.0

    def _calculate_reward(self, min_gap, collision):
        """
        Reward function designed for stable learning.
        
        Key principles:
        1. Dense rewards (every step) not just terminal
        2. Gradual penalties, not cliff edges
        3. Reward shaping to guide learning
        """
        if collision:
            return -100.0  # Terminal penalty (not too extreme)
        
        cars = self.city.cars
        
        # 1. Gap safety reward (most important)
        desired_gap = self.min_dis + self.v_des * self.reaction_time
        gap_error = (min_gap - desired_gap) / desired_gap
        
        if min_gap < 2.0:  # Danger zone
            gap_reward = -10.0 * (2.0 - min_gap)
        elif min_gap < desired_gap * 0.5:  # Too close
            gap_reward = -2.0 * abs(gap_error)
        elif min_gap > desired_gap * 1.5:  # Too far (inefficient)
            gap_reward = -0.5 * abs(gap_error)
        else:  # Good range
            gap_reward = 1.0 - abs(gap_error)
        
        # 2. Velocity tracking reward
        avg_vel = np.mean([c.velocity for c in cars])
        vel_error = abs(avg_vel - self.v_des) / self.v_des
        vel_reward = -vel_error
        
        # 3. Smoothness reward (penalize harsh accelerations)
        avg_acc = np.mean([abs(c.acceleration) for c in cars])
        smooth_reward = -0.1 * avg_acc
        
        # 4. Stability reward (penalize velocity variance)
        vel_std = np.std([c.velocity for c in cars])
        stability_reward = -0.05 * vel_std
        
        # 5. STRING STABILITY reward (key for BCC behavior)
        # Disturbances should dampen through the platoon, not amplify
        string_stability_reward = 0.0
        accelerations = [c.acceleration for c in cars]
        
        # Check if accelerations amplify through the chain
        # cars[0] is lead, cars[1:] are followers
        amplification_count = 0
        for i in range(1, len(accelerations)):
            # If follower's |acceleration| > predecessor's |acceleration|, that's amplification
            if abs(accelerations[i]) > abs(accelerations[i-1]) + 0.1:  # 0.1 tolerance
                amplification_count += 1
        
        # Penalize amplification (bad for string stability)
        string_stability_reward = -0.5 * amplification_count
        
        # Bonus for dampening: if last car has smaller |acc| than lead
        if len(accelerations) >= 2:
            lead_acc = abs(accelerations[0])
            last_acc = abs(accelerations[-1])
            if lead_acc > 0.1 and last_acc < lead_acc:
                string_stability_reward += 0.5  # Reward for dampening
        
        # 6. Survival bonus (encourage longer episodes)
        survival_bonus = 0.1
        
        total_reward = gap_reward + vel_reward + smooth_reward + stability_reward + string_stability_reward + survival_bonus
        
        return float(np.clip(total_reward, -100.0, 10.0))

    def render(self):
        pass
    
    def close(self):
        pass
