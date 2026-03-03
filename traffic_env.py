import gymnasium as gym
from gymnasium import spaces
import numpy as np
from city import City
from car import Car

class TrafficControlEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The agent controls the weights of a 5-term controller for a fleet of vehicles.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, curriculum_step=1):
        super(TrafficControlEnv, self).__init__()
        
        self.city = City()
        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0
        self.episode_count = 0
        self.collision_count = 0
        self.curriculum_step = curriculum_step
        
        # Action Space: 5 continuous weights - relaxed bounds for better exploration
        # Range [0.01, 3.0] allows agent to discover good balance
        self.action_space = spaces.Box(low=0.01, high=3.0, shape=(5,), dtype=np.float32)

        # Observation Space: [avg_vel_error, vel_std, avg_acc, spare]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Curriculum learning: start with fewer cars, gradually increase
        self.base_num_cars = 5
        self.max_num_cars = 15
        
        self.sim_params = {
            'num_cars': min(self.base_num_cars + curriculum_step, self.max_num_cars),
            'kd': 0.9, 'kv': 0.6, 'kc': 0.4,
            'v_des': 15.0,
            'min_dis': 5.0,
            'reaction_time': 1.5
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize the city simulation
        self.city.init(
            self.sim_params['num_cars'],
            self.sim_params['kd'],
            self.sim_params['kv'],
            self.sim_params['kc'],
            self.sim_params['v_des'],
            30.0, # max_v
            0.0,  # min_v
            self.sim_params['min_dis'],
            self.sim_params['reaction_time'],
            2.0,  # headway_time
            3.0,  # max_a
            -5.0, # min_a
            2.0,  # min_gap
            self.dt,
            model='ACC' 
        )
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Ensure action is numpy array and clipped to valid range
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply the weights (action) to ALL cars in the fleet
        for car in self.city.cars:
            car.set_weights(action)
            
        # Run simulation step
        self.city.run(self.dt)
        
        # Calculate Reward and Observation
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_collision()
        truncated = self.current_step >= self.max_steps
        
        if terminated:
            self.collision_count += 1
        
        info = {
            "avg_speed": np.mean([c.velocity for c in self.city.cars]),
            "min_gap": self._get_min_gap(),
            "collisions": 1 if terminated else 0,
            "episode_collision_count": self.collision_count
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        cars = self.city.cars
        if not cars:
            return np.zeros(4, dtype=np.float32)
            
        # Calculate fleet wide metrics
        gap_errors = []
        vel_errors = []
        accs = []
        energies = []
        
        for car in cars:
            # Re-calculate desired gap since it's local logic
            desired_gap = car.min_dis + car.velocity * car.headway_time
            
            # Simple assumption for neighbors (City class handles the lookup usually)
            # accessing gap history if available would be easier, but we can compute
            # current gap based on positions
            
            # Since city.py doesn't easily expose neighbors 'publicly' without search,
            # we rely on the fact that cars are sorted by position in the list usually?
            # Actually, let's just use the car's state:
            
            # Note: We need the actual gap. 'city.py' logic does this.
            # Let's approximate fleet state by standard deviation of spacing 
            # (lower std dev = better spacing)
            
            vel_errors.append(abs(car.velocity - self.sim_params['v_des']))
            accs.append(abs(car.acceleration))
            
            # Energy model is inside Car, but let's use acceleration as proxy for now
            energies.append(car.energy_used) # Accumulated

        # Gap consistency: Standard Deviation of positions (normalized) would be complex on ring road
        # Let's trust city.run updated the cars and just return aggregate Velocity Error
        
        avg_vel_error = np.mean(vel_errors)
        avg_acc = np.mean(accs)
        
        # We need a metric for gap error. 
        # Ideally we'd modify car.py to store 'last_gap_error' but let's avoid too many edits.
        # We'll use velocity variance as a proxy for instability.
        vel_std = np.std([c.velocity for c in cars])
        
        return np.array([avg_vel_error, vel_std, avg_acc, 0.0], dtype=np.float32)

    def _calculate_reward(self):
        # Improved Reward Function with gradual learning
        # Key changes: Reduced weight on velocity, gradual gap penalties, lower collision penalty
        
        cars = self.city.cars
        if not cars:
            return -100.0
        
        # 1. Velocity tracking - REDUCED weight to allow exploration
        vel_errors = [abs(c.velocity - self.sim_params['v_des']) for c in cars]
        avg_vel_error = np.mean(vel_errors)
        vel_reward = -0.5 * avg_vel_error
        
        # 2. Smoothness (acceleration should be minimal)
        avg_acc = np.mean([abs(c.acceleration) for c in cars])
        smoothness_reward = -0.05 * avg_acc
        
        # 3. Stability (velocity deviation across fleet)
        vel_std = np.std([c.velocity for c in cars])
        stability_reward = -0.02 * vel_std
        
        # 4. Gap safety - GRADUAL penalties to allow learning
        min_gap = self._get_min_gap()
        
        # Collision (hard wall)
        if min_gap < 0.5:
            return -5000.0  # Reduced from -10000 for better learning
        
        # Near-collision penalty (0.5-1.0m)
        if min_gap < 1.0:
            gap_penalty = -100.0 * (1.0 - min_gap)
        # Warning zone (1.0-2.0m)
        elif min_gap < 2.0:
            gap_penalty = -10.0 * (2.0 - min_gap)
        # Safe zone (>2.0m) - small reward
        else:
            gap_penalty = 0.1 * min(min_gap - 2.0, 5.0)
        
        # Total reward
        reward = vel_reward + smoothness_reward + stability_reward + gap_penalty
        
        # Bonus for completing episode
        if self.current_step == self.max_steps - 1:
            reward += 50.0
        
        return float(reward)

    def _get_min_gap(self):
        """Get minimum gap among all consecutive cars."""
        if not self.city.roads:
            return 1000.0
        
        road_len = self.city.roads[0].length
        cars = sorted(self.city.cars, key=lambda c: c.pos)
        min_gap = float('inf')
        
        for i in range(len(cars)):
            c1 = cars[i]
            c2 = cars[(i+1) % len(cars)]
            gap = (c2.pos - c1.pos - c1.length) % road_len
            min_gap = min(min_gap, gap)
        
        return min_gap if min_gap != float('inf') else 1000.0
    
    def _check_collision(self):
        """Collision check with safety margin. Threshold is 0.5m."""
        min_gap = self._get_min_gap()
        return min_gap < 0.5

    def render(self):
        pass
    
    def close(self):
        pass
