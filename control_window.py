'''
Control_window.py: Contains the ControlWindow class for managing the traffic simulation GUI.
'''


import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import csv
from city import City
from transportation_painter import TransportationPainter

class ControlWindow:
    def __init__(self, master):
        self.master = master

        # Create a scrollable frame inside a canvas
        self.canvas_wrapper = tk.Canvas(master)
        self.scroll_y = tk.Scrollbar(master, orient="vertical", command=self.canvas_wrapper.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas_wrapper.pack(side="left", fill="both", expand=True)

        # This frame will hold all your content
        self.scrollable_frame = tk.Frame(self.canvas_wrapper)

        # Connect canvas to the scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_wrapper.configure(
                scrollregion=self.canvas_wrapper.bbox("all")
            )
        )

        # Embed the frame inside the canvas
        self.canvas_wrapper.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_wrapper.configure(yscrollcommand=self.scroll_y.set)

        # Create a panel/frame for parameter entries and buttons
        self.panel = tk.Frame(self.scrollable_frame)
        self.panel.pack()

        # Dictionary to hold entry widgets for parameters
        self.entries = {}

        # Default values for simulation parameters
        default_values = {
            "car_number": 15,
            "kd": 0.9,
            "kv": 0.5,
            "kc": 0.4,
            "v_des": 15.0,
            "max_v": 30.0,
            "min_v": 0.0,
            "min_dis": 6.0,
            "reaction_time": 0.8,
            "headway_time": 1,
            "max_a": 3.0,
            "min_a": -5.0,
            "min_gap": 2.0,
            "dt": 0.1 
        }
        
        params = [
            ("car_number", "Number of Cars"),
            ("kd", "Gap Control Gain (kd)"),
            ("kv", "Relative Velocity Gain (kv)"),
            ("kc", "Desired Velocity Gain (kc)"),
            ("v_des", "Desired Velocity"),
            ("max_v", "Max Velocity"),
            ("min_v", "Min Velocity"),
            ("min_dis", "Min Distance Between cars (buffer distance)"),
            ("reaction_time", "Reaction Time"),
            ("headway_time", "Headway Time"),
            ("max_a", "Max Acceleration"),
            ("min_a", "Min Acceleration"),
            ("min_gap", "Minimum Gap Between Cars (for collision check)"),
            ("dt", "Simulation Time Step (dt)")  
        ]

        # Create label and entry for each parameter
        for i, (key, label) in enumerate(params):
            tk.Label(self.panel, text=label).grid(row=i, column=0)
            entry = tk.Entry(self.panel)
            entry.grid(row=i, column=1)
            entry.insert(0, str(default_values.get(key, "")))
            self.entries[key] = entry

        # Buttons for running, stopping, and resuming simulations
        self.run_button = tk.Button(self.panel, text="Run", command=self.run_simulation)
        self.run_button.grid(row=len(params), column=0)

        # Extra buttons for controlling LEAD and FOLLOWING separately
        self.stop_lead_button = tk.Button(self.panel, text="Stop Lead", command=lambda: self.stop_lead())
        self.stop_lead_button.grid(row=len(params), column=1)

        self.resume_lead_button = tk.Button(self.panel, text="Resume Lead", command=lambda: self.resume_lead())
        self.resume_lead_button.grid(row=len(params), column=2)

        self.stop_following_button = tk.Button(self.panel, text="Stop Following", command=lambda: self.stop_follower())
        self.stop_following_button.grid(row=len(params)+1, column=1)

        self.resume_following_button = tk.Button(self.panel, text="Resume Following", command=lambda: self.resume_follower())
        self.resume_following_button.grid(row=len(params)+1, column=2)

        # Btn for acceleration
        self.plot_acc_button = tk.Button(self.panel, text="Plot Vel and Acc Profiles", command=self.plot_vel_acc_profiles)
        self.plot_acc_button.grid(row=len(params)+2, column=1)
        
        # Checkbox to enable velocity profile
        self.use_velocity_profile = tk.BooleanVar(value=False)
        self.velocity_profile_checkbox = tk.Checkbutton(self.panel, text="Enable Ego Velocity Profile", variable=self.use_velocity_profile)
        self.velocity_profile_checkbox.grid(row=len(params)+3, column=1, columnspan=2, sticky="w")

        # Create a City instance for ACC 
        self.city_acc = City()
        self.city_bcc = City()
        self.city_accbcc = City()

        # Visualization for ACC (top)
        self.painter_acc = TransportationPainter(self.scrollable_frame, self.city_acc.roads, self.city_acc.cars, width=1500, height=125, bg='white')
        self.painter_acc.pack()
        tk.Label(self.scrollable_frame, text="ACC").pack()
        self.energy_label_acc = tk.Label(self.scrollable_frame, text="Total Energy : 0 KwH", font=("Arial", 10, "bold"))
        self.energy_label_acc.pack()

        # # Visualization for BCC (bottom)
        self.painter_bcc = TransportationPainter(self.scrollable_frame, self.city_bcc.roads, self.city_bcc.cars, width=1500, height=125, bg='white')
        self.painter_bcc.pack()
        tk.Label(self.scrollable_frame, text="BCC").pack()
        self.energy_label_bcc = tk.Label(self.scrollable_frame, text="Total Energy : 0 KwH", font=("Arial", 10, "bold"))
        self.energy_label_bcc.pack()

        # Visualization for ACC and BCC velocity profiles
        self.painter_accbcc = TransportationPainter(self.scrollable_frame, self.city_accbcc.roads, self.city_accbcc.cars, width=1500, height=125, bg='white')
        self.painter_accbcc.pack()
        tk.Label(self.scrollable_frame, text="ACC and BCC Combined").pack()
        self.energy_label_accbcc = tk.Label(self.scrollable_frame, text="Total Energy : 0 KwH", font=("Arial", 10, "bold"))
        self.energy_label_accbcc.pack()
        
        # Timer for simulation updates
        self.timer = None

        # Flags to control leader stop for ACC and BCC
        self.leader_stop = False
        self.follower_stop = False

        self.dt = default_values["dt"] 
    
    def load_velocity_profile(self, filename="data.csv"):
        self.ego_velocity_profile = []
        self.ego_velocity_profile_1 = []
        with open("data.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row['time'])
                velocity = float(row['velocity'])
                self.ego_velocity_profile.append((time, velocity))

        with open("data2.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row['time'])
                velocity = float(row['velocity'])
                self.ego_velocity_profile_1.append((time, velocity))
        print("Velocity profile loaded from", "data1.csv and data2.csv")

    def run_simulation(self):
       # Get parameter values from entry fields
        args = []
        for key in ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time","headway_time",  "max_a", "min_a", "min_gap", "dt"]:
            val = self.entries[key].get()
            try:
                val = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except Exception:
                val = 0
            args.append(val)

        self.dt = args[-1]  # Set self.dt from user input

        self.city_acc = City()
        self.city_bcc = City()
        self.city_accbcc = City()

        # Initialize cities with parameters for ACC and BCC models, including dt
        self.city_acc.init(*args[:-1], dt=self.dt, model='ACC')
        self.city_bcc.init(*args[:-1], dt=self.dt, model='BCC')
        self.city_accbcc.init(*args[:-1], dt=self.dt, model='ACC+BCC')

        # Update painters with new city elements
        self.painter_acc.set_elements(self.city_acc.roads, self.city_acc.cars)
        self.painter_bcc.set_elements(self.city_bcc.roads, self.city_bcc.cars)
        self.painter_accbcc.set_elements(self.city_accbcc.roads, self.city_accbcc.cars)

        # Load velocity profile from CSV file if enabled
        if self.use_velocity_profile.get():
            self.load_velocity_profile()
            self.city_acc.lead_velocity_profile = self.ego_velocity_profile
            self.city_acc.follower_velocity_profile = self.ego_velocity_profile_1

            self.city_bcc.lead_velocity_profile = self.ego_velocity_profile
            self.city_bcc.follower_velocity_profile = self.ego_velocity_profile_1

            self.city_accbcc.lead_velocity_profile = self.ego_velocity_profile
            self.city_accbcc.follower_velocity_profile = self.ego_velocity_profile_1

        else:
            self.city_acc.lead_velocity_profile = []
            self.city_acc.follower_velocity_profile = []
            self.city_bcc.lead_velocity_profile = []
            self.city_bcc.follower_velocity_profile = []
            self.city_accbcc.lead_velocity_profile = []
            self.city_accbcc.follower_velocity_profile = []

        self.master.after(60000, self.plot_vel_acc_profiles)
        self.start_timer()

    def start_timer(self):
        # Cancel previous timer if exists, then start updating simulation
        if self.timer:
            self.master.after_cancel(self.timer)
        self.update_simulation()
   
    def plot_vel_acc_profiles(self):
        dt = self.dt  # Consistent time step

        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')

        # === ACC ===
        # Velocity (left)
        for idx, car in enumerate(self.city_acc.cars):
            time_axis = [dt * i for i in range(len(car.vel_history))]
            if idx == 0:
                axes[0, 0].plot(time_axis, car.vel_history,color="red", linewidth=0.5)
            elif idx == 2:
                axes[0, 0].plot(time_axis, car.vel_history,color="green", linewidth=0.5)
            else:
                axes[0, 0].plot(time_axis, car.vel_history,color="blue", linewidth = 0.5)

        axes[0, 0].set_title("ACC Velocity")
        axes[0, 0].set_ylabel("Velocity (m/s)")
        # axes[0, 0].legend(fontsize="x-small")
        axes[0, 0].grid(True)

        # Acceleration (right)
        for idx, car in enumerate(self.city_acc.cars):
            time_axis = [dt * i for i in range(len(car.acc_history))]
            if idx == 0:
                axes[0, 1].plot(time_axis, car.acc_history, color="red", linewidth=0.5)
            elif idx == 2:
                axes[0, 1].plot(time_axis, car.acc_history, color="green", linewidth=0.5)
            else:
                axes[0, 1].plot(time_axis, car.acc_history, color="blue", linewidth = 0.5)
        axes[0, 1].set_title("ACC Acceleration")
        axes[0, 1].set_ylabel("Acceleration (m/s^2)")
        # axes[0, 1].legend(fontsize="x-small")
        axes[0, 1].grid(True)

        # === BCC ===
        # Velocity (left)
        for idx, car in enumerate(self.city_bcc.cars):
            time_axis = [dt * i for i in range(len(car.vel_history))]
            if idx == 0:
                axes[1, 0].plot(time_axis, car.vel_history, color="red", linewidth=0.5)
            elif idx == 2:
                axes[1, 0].plot(time_axis, car.vel_history, color="green", linewidth=0.5)
            else:
                axes[1, 0].plot(time_axis, car.vel_history, color="blue", linewidth = 0.5)
        axes[1, 0].set_title("BCC Velocity")
        axes[1, 0].set_ylabel("Velocity (m/s)")
        # axes[1, 0].legend(fontsize="x-small")
        axes[1, 0].grid(True)

        # Acceleration (right)
        for idx, car in enumerate(self.city_bcc.cars):
            time_axis = [dt * i for i in range(len(car.acc_history))]
            if idx == 0:
                axes[1, 1].plot(time_axis, car.acc_history, color="red", linewidth=0.5)
            elif idx == 2:
                axes[1, 1].plot(time_axis, car.acc_history, color="green", linewidth=0.5)
            else:
                axes[1, 1].plot(time_axis, car.acc_history, color="blue", linewidth = 0.5)
        axes[1, 1].set_title("BCC Acceleration")
        axes[1, 1].set_ylabel("Acceleration  (m/s^2)")
        # axes[1, 1].legend(fontsize="x-small")
        axes[1, 1].grid(True)

        # === ACC+BCC ===
        # Velocity (left)
        for idx, car in enumerate(self.city_accbcc.cars):
            time_axis = [dt * i for i in range(len(car.vel_history))]
            if idx == 0:
                axes[2, 0].plot(time_axis, car.vel_history, color="red", linewidth=0.5)
            elif idx == 2:
                axes[2, 0].plot(time_axis, car.vel_history, color="green", linewidth=0.5)
            else:
                axes[2, 0].plot(time_axis, car.vel_history, color="blue", linewidth = 0.5)
        axes[2, 0].set_title("ACC+BCC Integration Velocity")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Velocity (m/s)")
        # axes[2, 0].legend(fontsize="x-small")
        axes[2, 0].grid(True)

        # Acceleration (right)
        for idx, car in enumerate(self.city_accbcc.cars):
            time_axis = [dt * i for i in range(len(car.acc_history))]
            if idx == 0:
                axes[2, 1].plot(time_axis, car.acc_history, color="red", linewidth=0.5)
            elif idx == 2:
                axes[2, 1].plot(time_axis, car.acc_history, color="green", linewidth=0.5)
            else:
                axes[2, 1].plot(time_axis, car.acc_history, color="blue", linewidth = 0.5)
        axes[2, 1].set_title("ACC+BCC Integration Acceleration")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_ylabel("Acceleration (m/s^2)")
        # axes[2, 1].legend(fontsize="x-small")
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def update_simulation(self):
        dt = self.dt 

        # Set leader stop flags for both cities
        self.city_acc.set_leader_stop(self.leader_stop)
        self.city_acc.set_follower_stop(self.follower_stop)
        self.city_bcc.set_leader_stop(self.leader_stop)
        self.city_bcc.set_follower_stop(self.follower_stop)
        self.city_accbcc.set_leader_stop(self.leader_stop)
        self.city_accbcc.set_follower_stop(self.follower_stop)

        # Run simulation step for both cities with fixed dt
        self.city_acc.run(dt)
        self.city_bcc.run(dt)
        self.city_accbcc.run(dt)

        # Update painters with new city elements
        self.painter_acc.set_elements(self.city_acc.roads, self.city_acc.cars)
        self.painter_bcc.set_elements(self.city_bcc.roads, self.city_bcc.cars)
        self.painter_accbcc.set_elements(self.city_accbcc.roads, self.city_accbcc.cars)

        # Redraw the visualizations
        self.painter_acc.repaint()
        self.painter_bcc.repaint()
        self.painter_accbcc.repaint()

        # Calculate total energy
        total_energy_acc = sum(car.energy_used for car in self.city_acc.cars)
        total_energy_bcc = sum(car.energy_used for car in self.city_bcc.cars)
        total_energy_accbcc = sum(car.energy_used for car in self.city_accbcc.cars)
        
        # Update label texts
        self.energy_label_acc.config(text=f"Total Energy : {total_energy_acc:.4f} KwH")
        self.energy_label_bcc.config(text=f"Total Energy : {total_energy_bcc:.4f} KwH")
        self.energy_label_accbcc.config(text=f"Total Energy : {total_energy_accbcc:.4f} KwH")

        # Schedule next update for 0.1 seconds later (100 ms)
        self.timer = self.master.after(int(dt*1000), self.update_simulation)


    def stop_lead(self):
        print("Stopping lead for ACC, BCC, and ACC+BCC")
        self.leader_stop = True
    
    def resume_lead(self):
        self.leader_stop = False

    def stop_follower(self):
        self.follower_stop = True

    def resume_follower(self):
        self.follower_stop = False


if __name__ == "__main__":
    # Create main window and start the application
    root = tk.Tk()
    root.title("Traffic Simulation Control Window")
    app = ControlWindow(root)
    root.mainloop()
