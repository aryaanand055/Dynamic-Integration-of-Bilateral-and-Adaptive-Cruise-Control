from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from gymnasium import spaces
from rl_city import RLCity

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
    OBS_MEANS = np.array([20.0, 0.0, 30.0, 0.0, 30.0, 0.0], dtype=np.float32)
    OBS_STDS  = np.array([15.0, 3.0, 30.0, 10.0, 30.0, 10.0], dtype=np.float32)

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
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
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
        
    def _randomize_params(self):
        """Domain randomization: vary parameters each episode."""
        rng = np.random.default_rng()
        self.sim_params = {
            'kd': rng.uniform(0.7, 1.1),
            'kv': rng.uniform(0.4, 0.8),
            'kc': rng.uniform(0.3, 0.5),
            'v_des': rng.uniform(15.0, 35.0),
            'min_dis': rng.uniform(4.0, 7.0),
            'reaction_time': rng.uniform(0.6, 1.5),
            'max_a': rng.uniform(3.0, 5.0),
            'min_a': rng.uniform(-6.0, -4.0),
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
                v_new = max(5.0, v - rng.uniform(3.0, 12.0))
            elif event == 'accel':
                v_new = min(self.sim_params['max_v'] - 5, v + rng.uniform(3.0, 10.0))
            else:
                v_new = v  # Maintain speed with slight variation
                v_new += rng.uniform(-2.0, 2.0)
                v_new = max(5.0, min(self.sim_params['max_v'] - 5, v_new))
            
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
        
        # Generate a random lead car velocity profile
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
        
        collision = self._check_collision()
        done = collision or (self.current_step >= self.max_steps)
        
        if collision:
            rewards -= 50.0  # Reduced from -1000 to avoid overwhelming per-step signal
            
        dones = np.array([done] * self.num_rl_cars)
        infos = [{"collisions": 1 if collision else 0} for _ in range(self.num_rl_cars)]
        
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
        Each car gets: [ego_vel, ego_accel, front_gap, front_rel_vel, back_gap, back_rel_vel]
        All values are normalized to roughly [-1, 1].
        """
        cars = self.city.cars
        if not cars:
            return np.zeros((self.num_rl_cars, 6), dtype=np.float32)
            
        road_length = self.city.roads[0].length if self.city.roads else 1000
        obs_array = np.zeros((self.num_rl_cars, 6), dtype=np.float32)
        
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
                front_gap = (car.pos - front_car.pos - car.length) % road_length
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
            
            raw = np.array([ego_vel, ego_accel, front_gap, front_rel_vel, back_gap, back_rel_vel], dtype=np.float32)
            # Normalize
            obs_array[obs_idx] = (raw - self.OBS_MEANS) / (self.OBS_STDS + 1e-8)
            
        return obs_array

    def _calculate_rewards(self):
        """
        Reward function for follower cars (indices 1..N-1).
        Balances safety, comfort, efficiency, energy, and string stability.
        """
        cars = self.city.cars
        rewards = np.zeros(self.num_rl_cars, dtype=np.float32)
        if not cars: return rewards
        
        road_length = self.city.roads[0].length if self.city.roads else 1000
        bcc_threshold = 25.0
        
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
            
            front_gap = (car.pos - front_car.pos - car.length) % road_length if front_car else road_length
            back_gap = (back_car.pos - car.pos - car.length) % road_length if back_car else road_length
            
            desired_gap = self.sim_params['min_dis'] + car.velocity * self.sim_params['reaction_time']
            time_gap = front_gap / max(car.velocity, 0.1)
            
            # 1. Safety: penalize being too close to the car ahead
            gap_error = max(0, desired_gap - front_gap)
            r_safety = -0.5 * (gap_error / max(desired_gap, 1.0)) ** 2
            
            # 2. Comfort: penalize large acceleration and jerk
            jerk = (car.acceleration - self.prev_accel[car_idx]) / self.dt
            r_comfort = -0.1 * (car.acceleration ** 2 + 0.5 * jerk ** 2) / 25.0
            
            # 3. Efficiency: reward maintaining desired speed
            speed_error = abs(car.velocity - self.sim_params['v_des'])
            r_efficiency = -0.3 * (speed_error / max(self.sim_params['v_des'], 1.0)) ** 2
            
            # 4. String stability: penalize amplification of acceleration from front car
            if front_car:
                accel_amplification = abs(car.acceleration) - abs(front_car.acceleration)
                r_string = -0.3 * max(0, accel_amplification) / self.sim_params['max_a']
            else:
                r_string = 0.0
            
            # 5. BCC awareness: penalize if the car behind is getting dangerously close
            r_bcc = 0.0
            if back_gap < bcc_threshold:
                back_gap_error = max(0, desired_gap - back_gap)
                r_bcc = -0.2 * (back_gap_error / max(desired_gap, 1.0)) ** 2
            
            # 6. Bonus: reward smooth cruising near desired speed with safe gaps
            if front_gap > desired_gap and speed_error < 2.0 and abs(car.acceleration) < 0.5:
                r_bonus = 0.1
            else:
                r_bonus = 0.0
                
            rewards[obs_idx] = r_safety + r_comfort + r_efficiency + r_string + r_bcc + r_bonus
            
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
