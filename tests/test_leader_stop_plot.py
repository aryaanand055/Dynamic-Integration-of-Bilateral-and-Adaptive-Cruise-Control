import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import PLOTS_DIR, resolve_model_path, DEFAULT_SIM_PARAMS
from env.accel_traffic_env import AccelTrafficEnv

def simulate_and_plot(model_path="td3_accel_agent.zip", output_png=None):
    """Simulates lead vehicle halt and plots 4-model comparative gap and velocity response."""
    resolved_model = resolve_model_path(model_path)
    model = TD3.load(resolved_model)
    
    dt = 0.1
    params = dict(DEFAULT_SIM_PARAMS)
    init_params = [params[k] for k in ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time", "headway_time", "max_a", "min_a", "min_gap"]]
    
    city_acc = City()
    city_acc.init(*init_params, dt=dt, model='ACC')
    city_bcc = City()
    city_bcc.init(*init_params, dt=dt, model='BCC')
    city_accbcc = City()
    city_accbcc.init(*init_params, dt=dt, model='ACC+BCC')
    city_rl = RLCity()
    city_rl.init(*init_params, dt=dt, model='ACC')
    
    for i, car in enumerate(city_rl.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False
            car.mode = 'ACC'
            
    total_time = 42.0
    steps = int(total_time / dt)
    t_stop = 4.0
    t_restart = 18.0
    
    cities = [
        ("ACC", city_acc),
        ("BCC", city_bcc),
        ("ACC+BCC Integration", city_accbcc),
        ("RL Direct Acceleration", city_rl)
    ]
    
    time_series = []
    gap_histories = {name: {i: [] for i in range(1, 15)} for name, _ in cities}
    vel_histories = {name: {i: [] for i in range(15)} for name, _ in cities}
    
    for step in range(steps):
        t = round(step * dt, 2)
        time_series.append(t)
        
        should_stop = (t >= t_stop and t < t_restart)
        for _, c in cities:
            c.set_leader_stop(should_stop)
            
        city_acc.run(dt)
        city_bcc.run(dt)
        city_accbcc.run(dt)
        
        obs_dim = model.observation_space.shape[0]
        obs = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'], obs_dim=obs_dim)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city_rl.cars[1:]):
            car.rl_action = float(act[i][0])
        city_rl.run(dt)
        
        for name, c in cities:
            road_len = c.roads[0].length
            for i, car in enumerate(c.cars):
                vel_histories[name][i].append(car.velocity)
                if i > 0:
                    lead = c.cars[i - 1]
                    gap = (car.pos - lead.pos - lead.length) % road_len
                    gap_histories[name][i].append(gap)
                    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    for row, (name, _) in enumerate(cities):
        for i in range(15):
            color = 'red' if i == 0 else ('green' if i == 1 else 'gray')
            alpha = 1.0 if i in (0, 1) else 0.4
            axes[row, 0].plot(time_series, vel_histories[name][i], color=color, alpha=alpha)
        axes[row, 0].set_title(f"{name} - Speeds")
        axes[row, 0].set_ylabel("Speed (m/s)")
        axes[row, 0].grid(True)
        
        for i in range(1, 15):
            color = 'blue' if i == 1 else 'gray'
            alpha = 1.0 if i == 1 else 0.4
            axes[row, 1].plot(time_series, gap_histories[name][i], color=color, alpha=alpha)
        axes[row, 1].axhline(y=6.0, color='orange', linestyle='--', label='min_dis=6m')
        axes[row, 1].set_title(f"{name} - Gaps")
        axes[row, 1].set_ylabel("Gap (m)")
        axes[row, 1].grid(True)
        
    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    target_plot = Path(output_png) if output_png else PLOTS_DIR / "leader_stop_restart_comparison.png"
    plt.savefig(target_plot)
    print(f"Plot saved to {target_plot}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    simulate_and_plot(target)
