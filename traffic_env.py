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

    def __init__(self):
        super(TrafficControlEnv, self).__init__()
        
        self.city = City()
        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0
        
        # Action Space: 5 continuous weights for the controller terms
        # Range: [-2.0, 2.0] to allow for both positive and negative feedback if needed, 
        # though typically gains are positive. adjust as necessary.
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(5,), dtype=np.float32)

        # Observation Space: 
        # Mean Gap Error, Mean Velocity Error (Front), Mean Velocity Error (Back), Mean Energy, Collision Flag
        # Using a simple fleet-average observation for now as requested.
        # [avg_gap_error, avg_vel_error, avg_acc, avg_energy_rate]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Simulation Parameters (matching run_headless.py defaults)
        self.sim_params = {
            'num_cars': 15,
            'kd': 0.9, 'kv': 0.6, 'kc': 0.4,
            'v_des': 15.0, # m/s
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
        
        # Apply the weights (action) to ALL cars in the fleet
        weights = action
        for car in self.city.cars:
            car.set_weights(weights)
            
        # Run simulation step
        self.city.run(self.dt)
        
        # Calculate Reward and Observation
        obs = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_collision()
        truncated = self.current_step >= self.max_steps
        
        info = {
            "avg_speed": np.mean([c.velocity for c in self.city.cars]),
            "collisions": 1 if terminated else 0
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
        # Reward Function:
        # - Negative Velocity Error (encourage target speed)
        # - Negative Acceleration Magnitude (encourage smoothness/efficiency)
        # - Negative Collision penalty
        
        cars = self.city.cars
        avg_vel_error = np.mean([abs(c.velocity - self.sim_params['v_des']) for c in cars])
        avg_jerk_proxy = np.mean([abs(c.acceleration) for c in cars]) # Smoothness
        
        reward = -1.0 * avg_vel_error - 0.1 * avg_jerk_proxy
        
        if self._check_collision():
            reward -= 1000.0
            
        return reward

    def _check_collision(self):
        # Simple collision check: if any car gap < min_safe_dist
        # We need to compute gaps again or rely on City log. 
        # Since City.run doesn't return collision flag, we iterate.
        
        road_len = self.city.roads[0].length # Assuming 1 road
        cars = sorted(self.city.cars, key=lambda c: c.pos)
        
        for i in range(len(cars)):
            c1 = cars[i]
            c2 = cars[(i+1)%len(cars)]
            
            gap = (c2.pos - c1.pos - c1.length) % road_len
            if gap < 0.5: # Collision threshold
                return True
        return False

    def render(self):
        pass
    
    def close(self):
        pass
