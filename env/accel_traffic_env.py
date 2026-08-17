from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from gymnasium import spaces
from simulation.rl_city import RLCity

class AccelTrafficEnv(VecEnv):
    """
    Decentralized Multi-Agent Environment wrapping RLCity.
    Uses VecEnv interface so Stable-Baselines3 naturally trains
    a single parameter-shared policy across all cars.
    
    Key design choices:
    - Lead car (index 0) is NOT RL-controlled. It follows a randomized
      velocity profile each episode so follower cars learn to react.
    - Only follower cars (indices 1..N-1) are RL-controlled.
    - Simulation parameters are domain-randomized each episode for robustness.
    - Observations are normalized to roughly [-1, 1] for stable learning.
    """
    # Normalization constants (approximate ranges for each observation feature)
    # Features: [ego_vel, ego_accel, front_gap, front_rel_vel, inverse_back_gap, back_rel_vel, v_des]
    OBS_MEANS = np.array([20.0, 0.0, 30.0, 0.0, 0.033, 0.0, 25.0], dtype=np.float32)
    OBS_STDS  = np.array([15.0, 3.0, 30.0, 10.0, 0.02, 10.0, 10.0], dtype=np.float32)

    def __init__(self):
        self.city = RLCity()
        self.dt = 0.1
        self.max_steps = 1000  # 100 seconds at dt=0.1
        self.current_step = 0
        self.num_cars = 15
        
        # Base simulation parameters (will be randomized each episode)
        self.base_params = {
            'kd': 0.9, 'kv': 0.6, 'kc': 0.4,
            'v_des': 25.0,
            'min_dis': 5.5,
            'reaction_time': 1.0,
            'max_a': 4.0,
            'min_a': -5.0,
            'max_v': 50.0,
            'min_v': 0.0,
        }
        
        # Current episode parameters (set during reset)
        self.sim_params = dict(self.base_params)
        
        # Lead car velocity profile for current episode
        self.lead_profile = []
        self.lead_profile_time = 0.0
        
        # Spaces for a SINGLE car (action = raw acceleration)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.float32(-5.0), 
            high=np.float32(4.0), 
            shape=(1,), 
            dtype=np.float32
        )
        
        # VecEnv treats each car as a separate "environment"
        # But we only train follower cars (num_cars - 1)
        self.num_rl_cars = self.num_cars - 1  # Exclude lead car
        super().__init__(self.num_rl_cars, obs_space, act_space)
        self.prev_accel = np.zeros(self.num_cars)
        # Track collision state per car so penalty is one-time, not per-step
        self.in_collision = np.zeros(self.num_rl_cars, dtype=bool)
        
        # Load ALL custom lead profiles from CSV files for training
        self.custom_profiles = []
        import csv
        import os
        project_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files = ["data/data.csv", "data/data1.csv", "data/data2.csv"]
        for csv_file in csv_files:
            csv_path = os.path.join(project_dir, csv_file)
            if os.path.exists(csv_path):
                try:
                    profile = []
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            t = float(row['time'])
                            v = float(row['velocity'])
                            profile.append((t, v))
                    if profile:
                        self.custom_profiles.append(profile)
                        print(f"Loaded custom profile from {csv_file} ({len(profile)} points, "
                              f"{profile[0][1]:.1f}-{max(v for _,v in profile):.1f} m/s)")
                except Exception as e:
                    print(f"Could not load {csv_file}: {e}")
        print(f"Total custom profiles loaded: {len(self.custom_profiles)}")
        
    def _randomize_params(self):
        """Domain randomization: vary parameters each episode."""
        rng = np.random.default_rng()
        self.sim_params = {
            'kd': rng.uniform(0.6, 1.2),
            'kv': rng.uniform(0.3, 0.9),
            'kc': rng.uniform(0.2, 0.6),
            'v_des': rng.uniform(10.0, 35.0),
            'min_dis': rng.uniform(3.0, 8.0),
            'reaction_time': rng.uniform(0.5, 1.8),
            'max_a': rng.uniform(2.5, 5.5),
            'min_a': rng.uniform(-7.0, -3.0),
            'max_v': 50.0,
            'min_v': 0.0,
        }
    
    def _generate_lead_profile(self):
        """
        Generate a random velocity profile for the lead car.
        This creates realistic disturbances: cruise, brake, accelerate.
        """
        rng = np.random.default_rng()
        profile = []
        t = 0.0
        v = self.sim_params['v_des']
        episode_duration = self.max_steps * self.dt  # 100 seconds
        
        while t < episode_duration:
            # Hold current velocity for a random duration
            hold_time = rng.uniform(5.0, 20.0)
            profile.append((t, v))
            t += hold_time
            
            if t >= episode_duration:
                break
            
            # Choose a new target velocity (realistic perturbation)
            event = rng.choice(['brake', 'accel', 'cruise'], p=[0.35, 0.30, 0.35])
            if event == 'brake':
                v_new = max(0.0, v - rng.uniform(3.0, 15.0))  # Allow complete stop
            elif event == 'accel':
                v_new = min(self.sim_params['max_v'] - 5, v + rng.uniform(3.0, 10.0))
            else:
                v_new = v  # Maintain speed with slight variation
                v_new += rng.uniform(-2.0, 2.0)
                v_new = max(0.0, min(self.sim_params['max_v'] - 5, v_new))  # Allow complete stop
            
            # Transition to new velocity over a short ramp
            ramp_time = rng.uniform(2.0, 5.0)
            profile.append((t, v))
            t += ramp_time
            v = v_new
            profile.append((t, v))
        
        # Ensure we end at the episode boundary
        profile.append((episode_duration, v))
        return profile
    
    def _apply_lead_profile(self):
        """Apply the velocity profile to the lead car via the city's built-in mechanism."""
        self.city.lead_velocity_profile = self.lead_profile
        
    def reset(self):
        self.current_step = 0
        
        # Randomize parameters for this episode
        self._randomize_params()
        
        # Mix training: 40% chance to train on a custom CSV profile (randomly chosen),
        # 60% chance to train on randomized profiles for generalization.
        rng = np.random.default_rng()
        if self.custom_profiles and rng.random() < 0.4:
            idx = rng.integers(0, len(self.custom_profiles))
            self.lead_profile = self.custom_profiles[idx]
            # When training on custom data, align v_des to the profile's max velocity
            max_v_in_profile = max(v for _, v in self.lead_profile)
            self.sim_params['v_des'] = max(max_v_in_profile, 5.0)
        else:
            self.lead_profile = self._generate_lead_profile()
        
        self.city.init(
            self.num_cars,
            self.sim_params['kd'],
            self.sim_params['kv'],
            self.sim_params['kc'],
            self.sim_params['v_des'],
            self.sim_params['max_v'],
            self.sim_params['min_v'],
            self.sim_params['min_dis'],
            self.sim_params['reaction_time'],
            2.0,  # headway_time
            self.sim_params['max_a'],
            self.sim_params['min_a'],
            2.0,  # min_gap
            self.dt,
            model='ACC'  # Base model for the lead car's heuristic fallback
        )
        
        # Lead car (index 0): NOT RL-controlled, follows velocity profile
        self.city.cars[0].is_rl = False
        
        # Follower cars (index 1..N-1): RL-controlled
        for car in self.city.cars[1:]:
            car.is_rl = True
            car.rl_action = 0.0
        
        # Apply the lead car's velocity profile
        self._apply_lead_profile()
            
        self.prev_accel = np.zeros(self.num_cars)
        self.in_collision = np.zeros(self.num_rl_cars, dtype=bool)
        return self._get_obs()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.current_step += 1
        
        acts = np.array(self.actions).flatten()
        
        # Save previous accelerations for jerk calculation in rewards
        self.prev_accel = np.array([c.acceleration for c in self.city.cars]) if self.city.cars else np.zeros(self.num_cars)
        
        # Apply RL actions to FOLLOWER cars only (indices 1..N-1)
        for i, car in enumerate(self.city.cars[1:]):
            if i < len(acts):
                car.is_rl = True
                car.rl_action = float(acts[i])
            
        self.city.run(self.dt)
        
        obs = self._get_obs()
        rewards = self._calculate_rewards()
        
        # We no longer terminate the episode early on a collision
        # to avoid punishing all cars for one car's mistake (Credit Assignment Problem).
        done = (self.current_step >= self.max_steps)
        
        global_collision = self._check_collision()
            
        dones = np.array([done] * self.num_rl_cars)
        infos = [{"collisions": 1 if global_collision else 0} for _ in range(self.num_rl_cars)]
        
        # Auto-reset logic required for VecEnv
        if done:
            terminal_obs = obs.copy()
            for i in range(self.num_rl_cars):
                infos[i]['terminal_observation'] = terminal_obs[i]
            obs = self.reset()
            
        return obs, rewards, dones, infos

    def _get_obs(self):
        """
        Build normalized observations for FOLLOWER cars only (indices 1..N-1).
        Each car gets: [ego_vel, ego_accel, front_gap, front_rel_vel, back_gap, back_rel_vel, v_des]
        All values are normalized to roughly [-1, 1].
        """
        cars = self.city.cars
        if not cars:
            return np.zeros((self.num_rl_cars, 7), dtype=np.float32)
            
        road_length = self.city.roads[0].length if self.city.roads else 1000
        obs_array = np.zeros((self.num_rl_cars, 7), dtype=np.float32)
        
        v_des = self.sim_params['v_des']
        
        # Only observe follower cars (skip index 0 = lead car)
        for obs_idx, car in enumerate(cars[1:]):
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            
            def gap_to(other):
                gap = (car.pos - other.pos) % road_length
                return gap if gap > 0 else float('inf')
            
            def gap_from(other):
                gap = (other.pos - car.pos) % road_length
                return gap if gap > 0 else float('inf')
                
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None
            
            ego_vel = car.velocity
            ego_accel = car.acceleration
            
            if front_car:
                front_gap = (car.pos - front_car.pos - front_car.length) % road_length
                front_rel_vel = front_car.velocity - ego_vel
            else:
                front_gap = road_length
                front_rel_vel = 0.0
                
            if back_car:
                back_gap = (back_car.pos - car.pos - car.length) % road_length
                back_rel_vel = back_car.velocity - ego_vel
            else:
                back_gap = road_length
                back_rel_vel = 0.0
                
            inverse_back_gap = 1.0 / max(back_gap, 1.0)
            
            raw = np.array([ego_vel, ego_accel, front_gap, front_rel_vel, inverse_back_gap, back_rel_vel, v_des], dtype=np.float32)
            # Normalize
            obs_array[obs_idx] = (raw - self.OBS_MEANS) / (self.OBS_STDS + 1e-8)
            
        return obs_array

    def _calculate_rewards(self):
        """
        Reward function for follower cars (indices 1..N-1).
        Uses continuous positive rewards for good driving behavior,
        with string stability as the highest priority.
        """
        cars = self.city.cars
        rewards = np.zeros(self.num_rl_cars, dtype=np.float32)
        if not cars: return rewards
        
        road_length = self.city.roads[0].length if self.city.roads else 1000
        v_des = self.sim_params['v_des']
        max_a = self.sim_params['max_a']
        
        for obs_idx, car in enumerate(cars[1:]):
            car_idx = obs_idx + 1  # Actual index in self.city.cars
            
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            
            def gap_to(other):
                gap = (car.pos - other.pos) % road_length
                return gap if gap > 0 else float('inf')
            def gap_from(other):
                gap = (other.pos - car.pos) % road_length
                return gap if gap > 0 else float('inf')
                
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None
            
            front_gap = (car.pos - front_car.pos - front_car.length) % road_length if front_car else road_length
            back_gap = (back_car.pos - car.pos - car.length) % road_length if back_car else road_length
            
            desired_gap = self.sim_params['min_dis'] + car.velocity * self.sim_params['reaction_time']
            
            # === 1. GAP TRACKING & SAFETY ===
            gap_ratio = front_gap / max(desired_gap, 1.0)
            if gap_ratio < 1.0:
                # Too close (safety risk)
                gap_error = 1.0 - gap_ratio
                r_safety = -10.0 * gap_error  # Steep linear penalty
                if front_gap < 5.0:
                    r_safety -= 5.0 * ((5.0 - front_gap) ** 2)  # Quadratic danger zone penalty
                r_gap = 0.0
            else:
                r_safety = 0.0
                if gap_ratio <= 1.5:
                    r_gap = 0.3 * (1.5 - gap_ratio) / 0.5  # Peaks at +0.3 when gap_ratio=1.0
                else:
                    r_gap = -0.3 * min(gap_ratio - 1.5, 10.0)  # Linear pull forward, capped at -3.0
            
            # === 2. STRING STABILITY (weight: 0.4) ===
            r_string = 0.0
            if front_car:
                front_accel_mag = abs(front_car.acceleration)
                my_accel_mag = abs(car.acceleration)
                
                # Only penalize amplification if we are close enough that it matters (gap_ratio < 1.5)
                # If we are falling behind, we *must* amplify acceleration to catch up!
                if front_accel_mag > 0.1 and gap_ratio < 1.5:
                    if my_accel_mag > front_accel_mag:
                        amplification = (my_accel_mag / front_accel_mag) - 1.0
                        r_string = -0.4 * min(amplification, 3.0)
                elif front_accel_mag <= 0.1:
                    if my_accel_mag > 0.5:
                        r_string = -0.1 * (my_accel_mag / max_a)
            
            # === 3. SPEED MATCHING (weight: 0.2) ===
            speed_ratio = abs(car.velocity - v_des) / max(v_des, 1.0)
            r_speed = 0.2 * (1.0 - min(speed_ratio, 2.0))
            
            # === 4. COMFORT & SMOOTHNESS (Option 2 Tuned) ===
            delta_a = car.acceleration - self.prev_accel[car_idx]
            # Normalizing jerk: using delta_a directly instead of (delta_a/dt)^2 to prevent mathematical explosion
            r_comfort = -2.5 * (car.acceleration ** 2 + 10.0 * delta_a ** 2) / 25.0
            r_action_delta = -1.5 * abs(car.acceleration - self.prev_accel[car_idx])
            
            # === 5. ENERGY CONSUMPTION (weight: 0.15) ===
            power_proxy = abs(car.acceleration * car.velocity)
            r_energy = -0.15 * (power_proxy / max(max_a * v_des, 1.0))
            
            # === 6. BCC AWARENESS ===
            r_bcc = 0.0
            if back_gap < desired_gap:
                back_gap_error = (desired_gap - back_gap) / max(desired_gap, 1.0)
                r_bcc = -0.15 * back_gap_error ** 2
            
            # SAFETY OVERRIDE: If tailgating heavily, disable positive speed/string rewards
            if r_safety < -0.1:
                r_speed = min(0.0, r_speed)
                r_string = min(0.0, r_string)
                
            # === 7. COLLISION — One-time + small continuous penalty to prevent staying stuck ===
            r_collision = 0.0
            currently_colliding = (front_gap < 0.5 or back_gap < 0.5)
            if currently_colliding:
                r_collision = -10.0  # Moderate continuous penalty per step of collision
                if not self.in_collision[obs_idx]:
                    r_collision -= 90.0  # Extra -90.0 (total -100.0) on collision START
            self.in_collision[obs_idx] = currently_colliding
                
            rewards[obs_idx] = r_string + r_gap + r_speed + r_comfort + r_action_delta + r_energy + r_bcc + r_safety + r_collision
            
        return rewards

    def _check_collision(self):
        road_len = self.city.roads[0].length if self.city.roads else 1000
        cars = sorted(self.city.cars, key=lambda c: c.pos)
        if len(cars) < 2: return False
        for i in range(len(cars)):
            c1 = cars[i]
            c2 = cars[(i+1)%len(cars)]
            gap = (c2.pos - c1.pos - c1.length) % road_len
            if gap < 0.5:
                return True
        return False

    # Required VecEnv methods
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None]*self.num_rl_cars
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [None]*self.num_rl_cars
    def env_is_wrapped(self, wrapper_class, indices=None): return [False]*self.num_rl_cars
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
