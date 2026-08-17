import tkinter as tk
from tkinter import ttk
import numpy as np
from stable_baselines3 import TD3
import os
import matplotlib.pyplot as plt
from gui.control_window import ControlWindow
from simulation.rl_city import RLCity

class RLControlWindow(ControlWindow):
    def __init__(self, master):
        self.model = None
        self.model_loaded = False
        super().__init__(master)
        
        # Try to load the model
        model_path = "td3_accel_agent.zip"
        if os.path.exists(model_path):
            try:
                self.model = TD3.load(model_path)
                self.model_loaded = True
                print(f"Successfully loaded RL model from {model_path}")
            except Exception as e:
                print(f"Error loading RL model: {e}")
        else:
            print(f"Warning: RL model '{model_path}' not found. RL simulation will not run.")
        


    def _build_visualization_area(self):
        self.right_content.columnconfigure(0, weight=1)

        # Metrics strip
        self.metrics_strip = tk.Frame(self.right_content, bg="#111827", padx=12, pady=8)
        self.metrics_strip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.metrics_strip.columnconfigure((0, 1, 2), weight=1)

        self.metric_value_total = self._inline_metric(self.metrics_strip, 0, "Energy", "0.0000 kWh")
        self.metric_value_vel = self._inline_metric(self.metrics_strip, 1, "Avg Velocity", "0.00 m/s")
        self.metric_value_stability = self._inline_metric(self.metrics_strip, 2, "Stability", "0.0")

        self.visualizations_wrap = ttk.Frame(self.right_content, style="Pane.TFrame")
        self.visualizations_wrap.grid(row=1, column=0, sticky="nsew")

        # Create City objects
        self.city_acc = RLCity() # Standard ACC
        self.city_bcc = RLCity() # Standard BCC
        self.city_accbcc = RLCity() # Integrated ACC+BCC
        self.city_rl = RLCity()  # RL Controlled

        # Build visualization cards
        self.painter_acc, self.energy_label_acc, self.meta_acc = self._build_visualization_card("ACC", self.city_acc, height=180)
        self.painter_bcc, self.energy_label_bcc, self.meta_bcc = self._build_visualization_card("BCC", self.city_bcc, height=180)
        self.painter_accbcc, self.energy_label_accbcc, self.meta_accbcc = self._build_visualization_card("ACC+BCC Integration", self.city_accbcc, height=180)
        
        rl_title = "RL TD3 Agent (Direct Acceleration)"
        self.painter_rl, self.energy_label_rl, self.meta_rl = self._build_visualization_card(rl_title, self.city_rl, height=180)

        self.painters = [self.painter_acc, self.painter_bcc, self.painter_accbcc, self.painter_rl]

    def run_simulation(self):
        # Get parameter values from entry fields
        args = []
        param_keys = ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time","headway_time",  "max_a", "min_a", "min_gap", "dt"]
        for key in param_keys:
            val = self.entries[key].get()
            try:
                val = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except Exception:
                val = 0
            args.append(val)

        self.dt = args[-1]  # Set self.dt from user input
        init_params = args[:-1]

        # Initialize cities
        self.city_acc.init(*init_params, dt=self.dt, model='ACC')
        self.city_bcc.init(*init_params, dt=self.dt, model='BCC')
        self.city_accbcc.init(*init_params, dt=self.dt, model='ACC+BCC')
        self.city_rl.init(*init_params, dt=self.dt, model='ACC')

        # Update painters with new city elements
        self.painter_acc.set_elements(self.city_acc.roads, self.city_acc.cars)
        self.painter_bcc.set_elements(self.city_bcc.roads, self.city_bcc.cars)
        self.painter_accbcc.set_elements(self.city_accbcc.roads, self.city_accbcc.cars)
        self.painter_rl.set_elements(self.city_rl.roads, self.city_rl.cars)

        # Set cars to RL mode (except the lead car)
        for i, car in enumerate(self.city_rl.cars):
            if i > 0:
                car.is_rl = True
                car.rl_action = 0.0
            else:
                car.is_rl = False
                car.mode = 'ACC'

        # Handle velocity profiles if enabled
        if self.use_velocity_profile.get():
            self.load_velocity_profile()
            for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                city.lead_velocity_profile = self.ego_velocity_profile
                city.follower_velocity_profile = self.ego_velocity_profile_1
        else:
            for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                city.lead_velocity_profile = []
                city.follower_velocity_profile = []

        self.current_step = 0
        self.start_timer()

    def update_simulation(self):
        dt = self.dt
        self.current_step += 1
        
        # Handle standard logic for ACC and BCC
        self.city_acc.set_leader_stop(self.leader_stop)
        self.city_acc.set_follower_stop(self.follower_stop)
        self.city_bcc.set_leader_stop(self.leader_stop)
        self.city_bcc.set_follower_stop(self.follower_stop)
        self.city_accbcc.set_leader_stop(self.leader_stop)
        self.city_accbcc.set_follower_stop(self.follower_stop)
        
        self.city_acc.run(dt)
        self.city_bcc.run(dt)
        self.city_accbcc.run(dt)

        # Handle RL Agent Logic
        if self.model_loaded and self.city_rl.cars:
            self.city_rl.set_leader_stop(self.leader_stop)
            self.city_rl.set_follower_stop(self.follower_stop)
            
            try:
                # 1. Get Observations for RL cars
                obs = self._get_rl_observation(self.city_rl)
                
                # 2. Predict Acceleration
                # TD3 predicts a vector of accelerations for all cars
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 3. Apply raw actions directly to FOLLOWER cars (indices 1..N-1)
                # No EMA smoothing — rl_city.py's jerk limiter handles smoothness
                for i, car in enumerate(self.city_rl.cars[1:]):
                    if i < len(action):
                        car.rl_action = float(action[i][0])
            except Exception as e:
                if self.current_step <= 1: 
                    print(f"RL Prediction/Observation error: {e}")
            
            # 4. Run RL City Step
            self.city_rl.run(dt)

        # Redraw all painters
        for p in self.painters:
            p.repaint()

        # Update metrics and labels
        total_energy_acc = sum(car.energy_used for car in self.city_acc.cars)
        total_energy_bcc = sum(car.energy_used for car in self.city_bcc.cars)
        total_energy_accbcc = sum(car.energy_used for car in self.city_accbcc.cars)
        total_energy_rl = sum(car.energy_used for car in self.city_rl.cars)

        self.energy_label_acc.config(text=f"Energy {total_energy_acc:.4f} kWh")
        self.energy_label_bcc.config(text=f"Energy {total_energy_bcc:.4f} kWh")
        self.energy_label_accbcc.config(text=f"Energy {total_energy_accbcc:.4f} kWh")
        self.energy_label_rl.config(text=f"Energy {total_energy_rl:.4f} kWh")

        # Avg velocities
        def get_avg_v(city):
            return sum(car.velocity for car in city.cars) / len(city.cars) if city.cars else 0.0

        self.meta_acc.config(text=f"Cars {len(self.city_acc.cars):02d} | Avg v {get_avg_v(self.city_acc):05.2f} m/s")
        self.meta_bcc.config(text=f"Cars {len(self.city_bcc.cars):02d} | Avg v {get_avg_v(self.city_bcc):05.2f} m/s")
        self.meta_accbcc.config(text=f"Cars {len(self.city_accbcc.cars):02d} | Avg v {get_avg_v(self.city_accbcc):05.2f} m/s")
        self.meta_rl.config(text=f"Cars {len(self.city_rl.cars):02d} | Avg v {get_avg_v(self.city_rl):05.2f} m/s")

        # Global metrics strip update
        total_energy_all = total_energy_acc + total_energy_bcc + total_energy_accbcc + total_energy_rl
        self.metric_value_total.config(text=f"{total_energy_all:08.4f} kWh")
        self.metric_value_vel.config(text=f"{get_avg_v(self.city_rl):05.2f} m/s") # Show RL speed as primary

        # Schedule next update
        self.timer = self.master.after(int(dt*1000), self.update_simulation)

    def _get_rl_observation(self, city):
        # Features: [ego_vel, ego_accel, front_gap, front_rel_vel, inverse_back_gap, back_rel_vel, v_des]
        OBS_MEANS = np.array([20.0, 0.0, 30.0, 0.0, 0.033, 0.0, 25.0], dtype=np.float32)
        OBS_STDS  = np.array([15.0, 3.0, 30.0, 10.0, 0.02, 10.0, 10.0], dtype=np.float32)
        
        cars = city.cars
        road_length = city.roads[0].length if city.roads else 1000
        v_des = getattr(city, 'v_des', 25.0)
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
            
            raw = np.array([car.velocity, car.acceleration, front_gap, front_rel_vel, inverse_back_gap, back_rel_vel, v_des], dtype=np.float32)
            normalized = (raw - OBS_MEANS) / (OBS_STDS + 1e-8)
            obs_array.append(normalized)
            
        return np.array(obs_array, dtype=np.float32)

    def plot_vel_acc_profiles(self):
        dt = self.dt

        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')

        models = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC Integration", self.city_accbcc),
            ("RL Direct Acceleration", self.city_rl)
        ]

        for row, (name, city) in enumerate(models):
            # Velocity
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.vel_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 0].plot(time_axis, car.vel_history, color=color, linewidth=linewidth)
            
            axes[row, 0].set_title(f"{name} Velocity")
            axes[row, 0].set_ylabel("Velocity (m/s)")
            axes[row, 0].grid(True)
            if row == 3: axes[row, 0].set_xlabel("Time (s)")

            # Acceleration
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.acc_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 1].plot(time_axis, car.acc_history, color=color, linewidth=linewidth)
                
            axes[row, 1].set_title(f"{name} Acceleration")
            axes[row, 1].set_ylabel("Accel (m/s^2)")
            axes[row, 1].grid(True)
            if row == 3: axes[row, 1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("RL-Enhanced Traffic Simulation")
    app = RLControlWindow(root)
    root.mainloop()
