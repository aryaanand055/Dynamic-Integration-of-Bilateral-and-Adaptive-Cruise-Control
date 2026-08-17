import sys
import os
sys.path.append('.')
from simulation.city import City
from simulation.rl_city import RLCity
from run_headless import load_velocity_profiles
from stable_baselines3 import TD3
import numpy as np

params = {
    'car_number': 15, 'kd': 0.9, 'kv': 0.5, 'kc': 0.4,
    'v_des': 15.0, 'max_v': 30.0, 'min_v': 0.0,
    'min_dis': 6.0, 'reaction_time': 0.8, 'headway_time': 1.0,
    'max_a': 3.0, 'min_a': -5.0, 'min_gap': 2.0, 'dt': 0.1
}
init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]

city_acc = City()
city_acc.init(*init_args, dt=params['dt'], model='ACC')
city_bcc = City()
city_bcc.init(*init_args, dt=params['dt'], model='BCC')
city_accbcc = City()
city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')
city_rl = RLCity()
city_rl.init(*init_args, dt=params['dt'], model='ACC')

model = TD3.load('td3_accel_agent.zip')

for i, car in enumerate(city_rl.cars):
    if i > 0: car.is_rl = True; car.rl_action = 0.0
    else: car.is_rl = False

load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl)
city_rl.lead_velocity_profile = [
    (0.0, 25.0), (10.0, 25.0), (13.0, 18.0), (20.0, 18.0), 
    (23.0, 28.0), (30.0, 28.0), (33.0, 15.0), (40.0, 15.0), 
    (44.0, 25.0), (50.0, 25.0), (53.0, 30.0), (60.0, 30.0)
]

def _get_rl_observation(city, v_des=25.0):
    OBS_MEANS = np.array([20.0, 0.0, 30.0, 0.0, 0.033, 0.0, 25.0], dtype=np.float32)
    OBS_STDS  = np.array([15.0, 3.0, 30.0, 10.0, 0.02, 10.0, 10.0], dtype=np.float32)
    road_length = city.roads[0].length
    obs_array = []
    for car in city.cars[1:]:
        cars_same_road = [c for c in city.cars if c.current_road == car.current_road and c != car]
        def gap_to(other): gap = (car.pos - other.pos) % road_length; return gap if gap > 0 else float('inf')
        def gap_from(other): gap = (other.pos - car.pos) % road_length; return gap if gap > 0 else float('inf')
        front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
        back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None
        
        front_gap = (car.pos - front_car.pos - front_car.length) % road_length if front_car else road_length
        front_rel_vel = front_car.velocity - car.velocity if front_car else 0.0
        back_gap = (back_car.pos - car.pos - car.length) % road_length if back_car else road_length
        back_rel_vel = back_car.velocity - car.velocity if back_car else 0.0
        
        inverse_back_gap = 1.0 / max(back_gap, 1.0)
        
        raw = np.array([car.velocity, car.acceleration, front_gap, front_rel_vel, inverse_back_gap, back_rel_vel, v_des], dtype=np.float32)
        obs_array.append((raw - OBS_MEANS) / (OBS_STDS + 1e-8))
    return np.array(obs_array, dtype=np.float32)



num_steps = int(60 / params['dt'])
for _ in range(num_steps):
    city_acc.run(params['dt'])
    city_bcc.run(params['dt'])
    city_accbcc.run(params['dt'])
    
    obs = _get_rl_observation(city_rl, v_des=params['v_des'])
    action, _ = model.predict(obs, deterministic=True)
    for i, car in enumerate(city_rl.cars[1:]):
        car.rl_action = float(action[i][0])
    city_rl.run(params['dt'])

print('ACC Energy:', sum(car.energy_used for car in city_acc.cars))
print('BCC Energy:', sum(car.energy_used for car in city_bcc.cars))
print('ACC+BCC Energy:', sum(car.energy_used for car in city_accbcc.cars))
print('RL Energy:', sum(car.energy_used for car in city_rl.cars))
