import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

from accel_traffic_env import AccelTrafficEnv

def evaluate():
    env = AccelTrafficEnv()
    model_path = "td3_accel_agent"
    
    if not os.path.exists(model_path + ".zip"):
        print("Model not found!")
        return
        
    model = TD3.load(model_path)
    
    # Run evaluation
    obs = env.reset()
    
    steps = 500
    
    speeds = []
    accels = []
    gaps = []
    jerks = []
    rewards = []
    
    dt = env.dt
    prev_accel = np.zeros(env.sim_params['num_cars'])
    
    for _ in range(steps):
        # Predict actions for all 15 cars using parameter sharing
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards_step, dones, infos = env.step(actions)
        
        # Add sum of rewards across all cars to the step reward
        total_step_reward = np.sum(rewards_step)
        rewards.append(total_step_reward)
        
        current_speeds = []
        current_accels = []
        current_gaps = []
        current_jerks = []
        
        cars = env.city.cars
        road_len = env.city.roads[0].length if env.city.roads else 1000
        
        for i, car in enumerate(cars):
            current_speeds.append(car.velocity)
            current_accels.append(car.acceleration)
            
            # compute jerk
            j = (car.acceleration - prev_accel[i]) / dt
            current_jerks.append(j)
            
            # compute gap
            # Using gap_to just like the environment does for front car
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            def gap_to(other):
                g = (car.pos - other.pos) % road_len
                return g if g > 0 else float('inf')
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            
            gap = (car.pos - front_car.pos - car.length) % road_len if front_car else road_len
            current_gaps.append(gap)
            
        speeds.append(current_speeds)
        accels.append(current_accels)
        gaps.append(current_gaps)
        jerks.append(current_jerks)
        
        prev_accel = np.array([c.acceleration for c in cars])
        
        if any(dones):
            break
            
    speeds = np.array(speeds)
    accels = np.array(accels)
    gaps = np.array(gaps)
    jerks = np.array(jerks)
    
    # Calculate Analytics
    print("=== Analytics Report ===")
    print(f"Steps survived: {len(speeds)}")
    if len(speeds) > 0:
        print(f"Mean Speed: {np.mean(speeds):.2f} m/s (Target: {env.sim_params['v_des']})")
        print(f"Mean Acceleration: {np.mean(accels):.2f} m/s^2")
        print(f"Mean Gap: {np.mean(gaps):.2f} m")
        print(f"Mean Jerk: {np.mean(np.abs(jerks)):.2f} m/s^3")
        print(f"Total Reward: {np.sum(rewards):.2f}")
    
    # Save plots
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for i in range(min(5, speeds.shape[1])):
        plt.plot(speeds[:, i], label=f'Car {i}')
    plt.axhline(env.sim_params['v_des'], color='r', linestyle='--', label='Target Speed')
    plt.title("Vehicle Speeds Over Time")
    plt.xlabel("Step")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for i in range(min(5, accels.shape[1])):
        plt.plot(accels[:, i], label=f'Car {i}')
    plt.title("Vehicle Accelerations Over Time")
    plt.xlabel("Step")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for i in range(min(5, gaps.shape[1])):
        plt.plot(gaps[:, i], label=f'Car {i}')
    plt.title("Front Gap Over Time")
    plt.xlabel("Step")
    plt.ylabel("Gap (m)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(rewards)
    plt.title("Step Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    
    plt.tight_layout()
    plt.savefig("analytics_plot.png")
    print("Plot saved as analytics_plot.png")

if __name__ == '__main__':
    evaluate()
