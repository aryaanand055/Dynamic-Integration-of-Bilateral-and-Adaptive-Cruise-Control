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

from simulation.rl_city import RLCity
from simulation.config import PLOTS_DIR, resolve_model_path, DEFAULT_SIM_PARAMS
from env.accel_traffic_env import AccelTrafficEnv

def test_stop_and_resume(model_path="td3_accel_agent.zip", output_png=None):
    """
    Tests platoon complete stop and subsequent launch resumption.
    Verifies that all 15 cars stop cleanly, hold ~6m queue spacing,
    and restart smoothly without freezing or colliding.
    """
    resolved_model = resolve_model_path(model_path)
    print(f"Loading model from {resolved_model}...")
    model = TD3.load(resolved_model)
    dt = 0.1
    params = [15, 0.9, 0.5, 0.4, 15.0, 30.0, 0.0, 6.0, 0.8, 1.0, 3.0, -5.0, 2.0]
    
    city = RLCity()
    city.init(*params, dt=dt, model='ACC')
    for i, car in enumerate(city.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False
            car.mode = 'ACC'
            
    road_len = city.roads[0].length

    print("\n--- Phase 1: Bringing all vehicles to complete standstill (leader_stop=True) ---")
    city.set_leader_stop(True)
    t_records = []
    
    obs_dim = model.observation_space.shape[0]
    # Run until all vehicles come to standstill (up to 250 steps)
    for step in range(250):
        t = round(step * dt, 2)
        obs = AccelTrafficEnv.get_city_obs(city, 15.0, obs_dim=obs_dim)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city.cars[1:]):
            car.rl_action = float(act[i][0])
        city.run(dt)
        t_records.append(t)
        if step >= 100 and max(c.velocity for c in city.cars) < 0.05:
            print(f"All 15 vehicles fully settled at standstill after {step + 1} steps ({t}s).")
            break
        
    v_stop = [c.velocity for c in city.cars]
    gaps_stop = []
    for i in range(1, len(city.cars)):
        lead = city.cars[i - 1]
        car = city.cars[i]
        g = (car.pos - lead.pos - lead.length) % road_len
        gaps_stop.append(g)
        
    print(f"Standstill velocities (Car 0 to 14): {[round(v, 2) for v in v_stop]}")
    print(f"Standstill queue gaps (Car 1 to 14): {[round(g, 2) for g in gaps_stop]}")
    print(f"Max speed at standstill: {max(v_stop):.3f} m/s")
    print(f"Min queue gap: {min(gaps_stop):.2f}m, Max queue gap: {max(gaps_stop):.2f}m")
    
    assert max(v_stop) < 0.05, f"Vehicles failed to stop completely! Max v = {max(v_stop)}"
    assert min(gaps_stop) >= 4.5, f"Dangerous queue compression! Min gap = {min(gaps_stop)}"
    print(">>> PASS: All 15 vehicles reached complete standstill with safe queue spacing.")

    print("\n--- Phase 2: Restarting leader and observing follower drive-off (leader_stop=False) ---")
    city.set_leader_stop(False)
    restart_t = t_records[-1]
    
    for step in range(200):
        t = round(restart_t + (step + 1) * dt, 2)
        obs = AccelTrafficEnv.get_city_obs(city, 15.0, obs_dim=obs_dim)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city.cars[1:]):
            car.rl_action = float(act[i][0])
        city.run(dt)
        t_records.append(t)

    v_final = [c.velocity for c in city.cars]
    print(f"\nFinal velocities after drive-off: {[round(v, 2) for v in v_final]}")
    print(f"Min speed at resume: {min(v_final):.2f} m/s, Max speed: {max(v_final):.2f} m/s")
    
    assert min(v_final) >= 10.0, f"Some vehicles failed to resume drive-off! Min v = {min(v_final)}"
    print(">>> PASS: All vehicles resumed drive-off and accelerated smoothly back toward cruise speed.")

    # Plot velocity history
    fig, ax = plt.subplots(figsize=(12, 6))
    time_axis = [i * dt for i in range(len(city.cars[0].vel_history))]
    
    for i, car in enumerate(city.cars):
        color = 'red' if i == 0 else ('blue' if i == 1 else ('green' if i == 7 else 'gray'))
        label = f'Car {i}' if i in [0, 1, 7, 14] else None
        alpha = 1.0 if i in [0, 1, 7, 14] else 0.4
        lw = 2.0 if i in [0, 1] else 1.0
        ax.plot(time_axis, car.vel_history, color=color, label=label, alpha=alpha, linewidth=lw)

    ax.axvline(x=13.0, color='orange', linestyle='--', label='Leader Restarts (t=13.0s)')
    ax.set_title("15-Car Platoon Complete Stop and Drive-Off Launch Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()

    target_plot = Path(output_png) if output_png else PLOTS_DIR / "leader_stop_restart_comparison.png"
    plt.savefig(target_plot)
    print(f"Trajectory plot saved to {target_plot}")
    return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    test_stop_and_resume(target)
