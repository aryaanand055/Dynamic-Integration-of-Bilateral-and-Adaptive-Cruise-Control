import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from city import City
from rl_city import RLCity
from stable_baselines3 import TD3

def load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl):
    """Loads the velocity profiles from data files and assigns them to the cities."""
    ego_velocity_profile = []
    ego_velocity_profile_1 = []
    try:
        with open("data.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row['time'])
                velocity = float(row['velocity'])
                ego_velocity_profile.append((time, velocity))

        print("Velocity profiles loaded from data.csv")

        # Assign profiles to all city models
        city_acc.lead_velocity_profile = ego_velocity_profile
        city_acc.follower_velocity_profile = ego_velocity_profile_1

        city_bcc.lead_velocity_profile = ego_velocity_profile
        city_bcc.follower_velocity_profile = ego_velocity_profile_1

        city_accbcc.lead_velocity_profile = ego_velocity_profile
        city_accbcc.follower_velocity_profile = ego_velocity_profile_1

        city_rl.lead_velocity_profile = ego_velocity_profile
        city_rl.follower_velocity_profile = ego_velocity_profile_1

    except FileNotFoundError:
        print("Warning: data.csv not found. Running without velocity profiles.")
        city_acc.lead_velocity_profile = []
        city_acc.follower_velocity_profile = []
        city_bcc.lead_velocity_profile = []
        city_bcc.follower_velocity_profile = []
        city_accbcc.lead_velocity_profile = []
        city_accbcc.follower_velocity_profile = []
        city_rl.lead_velocity_profile = []
        city_rl.follower_velocity_profile = []

def plot_results(city_acc, city_bcc, city_accbcc, city_rl, dt, use_profiles):
    """Plots the velocity and acceleration profiles for all four models."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex='col')
    fig.suptitle('Simulation Results', fontsize=16)

    # --- Plotting function for a single model ---
    def plot_model(ax_vel, ax_acc, city, model_name):
        num_cars = len(city.cars)
        
        # Plot follower cars first
        for idx, car in enumerate(city.cars):
            if idx == 0:
                continue # Skip lead car, we'll plot it last

            time_axis = [dt * i for i in range(len(car.vel_history))]
            
            color = 'gray'
            linewidth = 0.8

            # Color car 2 green ONLY if velocity profiles are active
            if use_profiles and idx == 2:
                color = 'green'
                linewidth = 1.5

            ax_vel.plot(time_axis, car.vel_history, color=color, linewidth=linewidth)
            ax_acc.plot(time_axis, car.acc_history, color=color, linewidth=linewidth)

        # Plot the lead car (car 0) last to ensure it's on top
        if num_cars > 0:
            lead_car = city.cars[0]
            time_axis = [dt * i for i in range(len(lead_car.vel_history))]
            ax_vel.plot(time_axis, lead_car.vel_history, color='red', linewidth=1.5)
            ax_acc.plot(time_axis, lead_car.acc_history, color='red', linewidth=1.5)


        ax_vel.set_title(f"{model_name} Velocity")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.grid(True)

        ax_acc.set_title(f"{model_name} Acceleration")
        ax_acc.set_ylabel("Acceleration (m/s^2)")
        ax_acc.grid(True)



    # --- Plot each model ---
    plot_model(axes[0, 0], axes[0, 1], city_acc, "ACC")
    plot_model(axes[1, 0], axes[1, 1], city_bcc, "BCC")
    plot_model(axes[2, 0], axes[2, 1], city_accbcc, "ACC + BCC Integration")
    plot_model(axes[3, 0], axes[3, 1], city_rl, "RL Direct Acceleration")

    # Set common X-axis labels
    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    plt.savefig('simulation_results.png')
    print("Saved velocity/acceleration plots to simulation_results.png")

def plot_energy_consumption(city_acc, city_bcc, city_accbcc, city_rl):
    """Plots the total energy consumption for each model as a bar graph."""
    total_energy_acc = sum(car.energy_used for car in city_acc.cars)
    total_energy_bcc = sum(car.energy_used for car in city_bcc.cars)
    total_energy_accbcc = sum(car.energy_used for car in city_accbcc.cars)
    total_energy_rl = sum(car.energy_used for car in city_rl.cars)

    models = ['ACC', 'BCC', 'ACC+BCC', 'RL Model']
    energy_values = [total_energy_acc, total_energy_bcc, total_energy_accbcc, total_energy_rl]

    plt.figure(figsize=(7, 6))
    bars = plt.bar(models, energy_values, color=['lightblue'], width = 0.5)
    plt.ylabel('Energy Consumption (KwH)')
    plt.title('Total Energy Consumption per Model')
    plt.savefig('energy_consumption.png')
    print("Saved energy plot to energy_consumption.png")

def display_gap_statistics(city_acc, city_bcc, city_accbcc, city_rl):
    """Calculates and prints the final gap statistics for each model."""

    def safe_avg(gaps): return sum(gaps) / len(gaps) if gaps else 0

    print("\n--- Inter-vehicular Distance Statistics ---")
    
    for city, name in zip([city_acc, city_bcc, city_accbcc, city_rl], ["ACC", "BCC", "ACC+BCC", "RL"]):
        print(f"{name} Model:")
        print(f"  - Minimum Distance: {city.overall_min_gap:.2f} m")
        print(f"  - Average Distance: {safe_avg(city.all_gaps):.2f} m")
        print(f"  - Maximum Distance: {city.overall_max_gap:.2f} m")
        print("-" * 20)
    print("-------------------------------------------\n")

def get_gap_statistics(gaps):
    if not gaps:
        return None
    gaps = np.array(gaps)

    stats = {
        "min": np.min(gaps),
        "p5": np.percentile(gaps, 5),
        "p25": np.percentile(gaps, 25),
        "median": np.median(gaps),
        "mean": np.mean(gaps),
        "p75": np.percentile(gaps, 75),
        "p95": np.percentile(gaps, 95),
        "max": np.max(gaps),
        "std": np.std(gaps, ddof=1) if len(gaps)>1 else 0.0,
        "variance": np.var(gaps, ddof=1) if len(gaps)>1 else 0.0
    }
    df = pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    print(df.to_string(float_format="%.4f"))

def _get_rl_observation(city):
    """
    Build normalized observations for FOLLOWER cars only (indices 1..N-1).
    Must exactly match AccelTrafficEnv._get_obs() normalization.
    """
    OBS_MEANS = np.array([20.0, 0.0, 30.0, 0.0, 30.0, 0.0], dtype=np.float32)
    OBS_STDS  = np.array([15.0, 3.0, 30.0, 10.0, 30.0, 10.0], dtype=np.float32)
    
    cars = city.cars
    road_length = city.roads[0].length if city.roads else 1000
    obs_array = []
    
    # Only observe follower cars (skip index 0 = lead car)
    for car in cars[1:]:
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
        normalized = (raw - OBS_MEANS) / (OBS_STDS + 1e-8)
        obs_array.append(normalized)
        
    return np.array(obs_array, dtype=np.float32)

def main():
    """Main function to run the simulation without GUI."""

    # --- Control Flag ---
    USE_VELOCITY_PROFILES = True
    
    # --- Simulation Parameters ---
    simulation_duration = 60  # Run for 60 seconds
    params = {
        "car_number": 15,
        "kd": 0.9,
        "kv": 0.6,
        "kc": 0.4,
        "v_des": 30.0,
        "max_v": 50.0,
        "min_v": 0.0,
        "min_dis": 6.0,
        "reaction_time": 0.8,
        "headway_time": 2.0,
        "max_a": 4.0,
        "min_a": -5.0,
        "min_gap": 2.0,
        "dt": 0.1
    }
    dt = params["dt"]
    num_steps = int(simulation_duration / dt)

    # --- Initialize City Models ---
    city_acc = City()
    city_bcc = City()
    city_accbcc = City()
    city_rl = RLCity()

    # Unpack params dictionary to pass as arguments
    init_args = [params[k] for k in ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time", "headway_time", "max_a", "min_a", "min_gap"]]

    city_acc.init(*init_args, dt=dt, model='ACC')
    city_bcc.init(*init_args, dt=dt, model='BCC')
    city_accbcc.init(*init_args, dt=dt, model='ACC+BCC')
    city_rl.init(*init_args, dt=dt, model='ACC')

    # Load RL Model
    model_path = "td3_accel_agent.zip"
    model = None
    if os.path.exists(model_path):
        model = TD3.load(model_path)
        print("Successfully loaded RL model.")
    else:
        print("Warning: RL model not found. RL simulation will not run correctly.")

    # Set RL cars to RL-A mode (except for the lead vehicle)
    for i, car in enumerate(city_rl.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False

    # --- Load Velocity Profiles (Conditional) ---
    if USE_VELOCITY_PROFILES:
        load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl)

    # --- Run Simulation Loop ---
    print(f"Running simulation for {simulation_duration} seconds ({num_steps} steps)...")
    for step in range(num_steps):
        # Print progress every 10%
        if (step + 1) % (num_steps // 10) == 0:
            print(f"  ...Progress: {int(((step + 1) / num_steps) * 100)}%")
            
        city_acc.run(dt)
        city_bcc.run(dt)
        city_accbcc.run(dt)

        if model and city_rl.cars:
            obs = _get_rl_observation(city_rl)
            action, _ = model.predict(obs, deterministic=True)
            # Apply actions to FOLLOWER cars only (indices 1..N-1)
            for i, car in enumerate(city_rl.cars[1:]):
                if i < len(action):
                    car.rl_action = float(action[i])
        city_rl.run(dt)
    
    print("Simulation complete.")

    # --- Plot Final Results ---
    print("Generating plots...")
    plot_results(city_acc, city_bcc, city_accbcc, city_rl, dt, USE_VELOCITY_PROFILES)
    plot_energy_consumption(city_acc, city_bcc, city_accbcc, city_rl)
    display_gap_statistics(city_acc, city_bcc, city_accbcc, city_rl)
    
    print("ACC Stats:")
    get_gap_statistics(city_acc.all_gaps)
    print("\nBCC Stats:")
    get_gap_statistics(city_bcc.all_gaps)
    print("\nIntegrated Stats:")
    get_gap_statistics(city_accbcc.all_gaps)
    print("\nRL Stats:")
    get_gap_statistics(city_rl.all_gaps)

if __name__ == "__main__":
    main()
