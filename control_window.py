'''
Control_window.py: Contains the ControlWindow class for managing the traffic simulation GUI.
'''


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import csv
from city import City
from transportation_painter import TransportationPainter

class ControlWindow:
    def __init__(self, master):
        self.master = master

        self.master.configure(bg="#0f172a")
        self.master.minsize(1080, 720)
        self.master.title("ACC/BCC Engineering Dashboard")

        self.default_values = {
            "car_number": 15,
            "kd": 0.9,
            "kv": 0.5,
            "kc": 0.4,
            "v_des": 15.0,
            "max_v": 30.0,
            "min_v": 0.0,
            "min_dis": 6.0,
            "reaction_time": 0.8,
            "headway_time": 1.0,
            "max_a": 3.0,
            "min_a": -5.0,
            "min_gap": 2.0,
            "dt": 0.1,
        }
        self.entries = {}

        self._configure_style()
        self._build_layout()
        self._build_controls()
        self._build_visualization_area()

        # Timer for simulation updates
        self.timer = None

        # Flags to control leader/follower stop
        self.leader_stop = False
        self.follower_stop = False

        self.dt = self.default_values["dt"]

        self.master.bind("<Configure>", self._handle_resize)
        self._handle_resize()

    def _configure_style(self):
        style = ttk.Style(self.master)
        style.theme_use("clam")

        style.configure("Root.TFrame", background="#0f172a")
        style.configure("Pane.TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111827")

        style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 24, "bold"))
        style.configure("SubTitle.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#111827", foreground="#e2e8f0", font=("Segoe UI", 10, "bold"))
        style.configure("Meta.TLabel", background="#111827", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("FieldLabel.TLabel", background="#111827", foreground="#cbd5e1", font=("Segoe UI", 9))

        style.configure("TEntry", fieldbackground="#0b1220", foreground="#e2e8f0", insertcolor="#e2e8f0", borderwidth=0, padding=5)
        style.map("TEntry", fieldbackground=[("focus", "#132035")])

        style.configure("TButton", background="#1e293b", foreground="#e2e8f0", borderwidth=0, padding=8)
        style.map("TButton", background=[("active", "#2b3b4e")])

        style.configure("Primary.TButton", background="#486581", foreground="#e2e8f0", borderwidth=0, padding=9)
        style.map("Primary.TButton", background=[("active", "#5e7a96")])

        style.configure("TCheckbutton", background="#111827", foreground="#cbd5e1")

    def _build_layout(self):
        self.root = ttk.Frame(self.master, style="Root.TFrame", padding=(24, 20, 24, 20))
        self.root.pack(fill="both", expand=True)

        self.header = ttk.Frame(self.root, style="Root.TFrame")
        self.header.pack(fill="x", pady=(0, 14))
        ttk.Label(self.header, text="ACC/BCC Vehicle Simulation", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self.header,
            text="Engineering dashboard for adaptive and bilateral cruise control",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.body = ttk.Frame(self.root, style="Pane.TFrame")
        self.body.pack(fill="both", expand=True)
        self.body.columnconfigure(0, weight=22)
        self.body.columnconfigure(1, weight=78)
        self.body.rowconfigure(0, weight=1)

        # Left side has its own scroll to keep controls usable on smaller displays.
        self.left_wrapper = ttk.Frame(self.body, style="Pane.TFrame")
        self.left_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.left_wrapper.rowconfigure(0, weight=1)
        self.left_wrapper.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(self.left_wrapper, bg="#0f172a", highlightthickness=0)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")

        self.left_scroll = ttk.Scrollbar(self.left_wrapper, orient="vertical", command=self.left_canvas.yview)
        self.left_scroll.grid(row=0, column=1, sticky="ns")
        self.left_canvas.configure(yscrollcommand=self.left_scroll.set)

        self.controls_container = tk.Frame(self.left_canvas, bg="#0f172a")
        self.controls_window = self.left_canvas.create_window((0, 0), window=self.controls_container, anchor="nw")

        self.controls_container.bind("<Configure>", lambda _e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind("<Configure>", self._resize_left_canvas)

        self.right_wrapper = ttk.Frame(self.body, style="Pane.TFrame")
        self.right_wrapper.grid(row=0, column=1, sticky="nsew")
        self.right_wrapper.rowconfigure(0, weight=1)
        self.right_wrapper.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(self.right_wrapper, bg="#0f172a", highlightthickness=0)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")

        self.right_scroll = ttk.Scrollbar(self.right_wrapper, orient="vertical", command=self.right_canvas.yview)
        self.right_scroll.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=self.right_scroll.set)

        self.right_content = tk.Frame(self.right_canvas, bg="#0f172a")
        self.right_window = self.right_canvas.create_window((0, 0), window=self.right_content, anchor="nw")

        self.right_content.bind("<Configure>", lambda _e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas.bind("<Configure>", self._resize_right_canvas)

        # Use wheel/trackpad scrolling in both columns.
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)

    def _build_controls(self):
        # Create a City instance for ACC/BCC/ACC+BCC
        self.city_acc = City()
        self.city_bcc = City()
        self.city_accbcc = City()

        self.control_cards = []
        self._create_control_group(
            title="Simulation Setup",
            fields=[
                ("car_number", "Number of Cars"),
                ("dt", "Simulation Time Step (s)"),
            ],
        )
        self._create_control_group(
            title="Control Gains",
            fields=[
                ("kd", "Gap Control Gain (kd)"),
                ("kv", "Relative Velocity Gain (kv)"),
                ("kc", "Desired Velocity Gain (kc)"),
            ],
        )
        self._create_control_group(
            title="Constraints",
            fields=[
                ("v_des", "Desired Velocity"),
                ("max_v", "Max Velocity"),
                ("min_v", "Min Velocity"),
                ("max_a", "Max Acceleration"),
                ("min_a", "Min Acceleration"),
            ],
        )
        self._create_control_group(
            title="Safety",
            fields=[
                ("min_dis", "Buffer Distance"),
                ("reaction_time", "Reaction Time"),
                ("headway_time", "Headway Time"),
                ("min_gap", "Collision Check Gap"),
            ],
        )

        action_card = tk.Frame(self.controls_container, bg="#111827", padx=10, pady=10)
        action_card.pack(fill="x", pady=(0, 12))

        ttk.Label(action_card, text="Simulation Actions", style="Section.TLabel").pack(anchor="w")
        ttk.Label(action_card, text="Run, stop, and inspect behavior", style="Meta.TLabel").pack(anchor="w", pady=(0, 10))

        self.run_button = ttk.Button(action_card, text="Run Simulation", command=self.run_simulation, style="Primary.TButton")
        self.run_button.pack(fill="x", pady=(0, 8))

        controls_row_1 = ttk.Frame(action_card, style="Card.TFrame")
        controls_row_1.pack(fill="x", pady=(0, 6))
        controls_row_1.columnconfigure((0, 1), weight=1)
        self.stop_lead_button = ttk.Button(controls_row_1, text="Stop Lead", command=self.stop_lead)
        self.stop_lead_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.resume_lead_button = ttk.Button(controls_row_1, text="Resume Lead", command=self.resume_lead)
        self.resume_lead_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        controls_row_2 = ttk.Frame(action_card, style="Card.TFrame")
        controls_row_2.pack(fill="x", pady=(0, 6))
        controls_row_2.columnconfigure((0, 1), weight=1)
        self.stop_following_button = ttk.Button(controls_row_2, text="Stop Following", command=self.stop_follower)
        self.stop_following_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.resume_following_button = ttk.Button(controls_row_2, text="Resume Following", command=self.resume_follower)
        self.resume_following_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.plot_acc_button = ttk.Button(action_card, text="Plot Vel and Acc Profiles", command=self.plot_vel_acc_profiles)
        self.plot_acc_button.pack(fill="x", pady=(0, 8))

        self.use_velocity_profile = tk.BooleanVar(value=False)
        self.velocity_profile_checkbox = ttk.Checkbutton(
            action_card,
            text="Enable Ego Velocity Profile",
            variable=self.use_velocity_profile,
        )
        self.velocity_profile_checkbox.pack(anchor="w")

    def _create_control_group(self, title, fields):
        card = tk.Frame(self.controls_container, bg="#111827", padx=10, pady=10)
        card.pack(fill="x", pady=(0, 8))
        self.control_cards.append(card)

        ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(card, text="Tune parameters", style="Meta.TLabel").pack(anchor="w", pady=(0, 6))

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for idx, (key, label) in enumerate(fields):
            row = idx // 2
            col = idx % 2
            self._create_field(grid, key, label, row, col)

    def _create_field(self, parent, key, label, row, col):
        field = ttk.Frame(parent, style="Card.TFrame")
        field.grid(row=row, column=col, sticky="ew", padx=(0, 6) if col == 0 else (6, 0), pady=(0, 6))
        field.columnconfigure(0, weight=1)

        ttk.Label(field, text=label, style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 3))

        entry = ttk.Entry(field, font=("Consolas", 10), width=12)
        entry.grid(row=1, column=0, sticky="w")
        entry.insert(0, str(self.default_values[key]))
        self.entries[key] = entry

    def _build_visualization_area(self):
        self.right_content.columnconfigure(0, weight=1)

        # Integrated utility bar: metrics are docked near simulation content, not floating as separate cards.
        self.metrics_strip = tk.Frame(self.right_content, bg="#111827", padx=12, pady=8)
        self.metrics_strip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.metrics_strip.columnconfigure(0, weight=3)
        self.metrics_strip.columnconfigure(1, weight=2)
        self.metrics_strip.columnconfigure(2, weight=2)

        self.metric_value_total = self._inline_metric(self.metrics_strip, 0, "Energy", "0.0000 kWh")
        self.metric_value_vel = self._inline_metric(self.metrics_strip, 1, "Avg Velocity", "0.00 m/s")
        self.metric_value_stability = self._inline_metric(self.metrics_strip, 2, "Stability", "0.0")

        self.visualizations_wrap = ttk.Frame(self.right_content, style="Pane.TFrame")
        self.visualizations_wrap.grid(row=1, column=0, sticky="nsew")

        # Simulation panels are stacked so each model gets full horizontal width.
        self.painter_acc, self.energy_label_acc, self.meta_acc = self._build_visualization_card("ACC", self.city_acc, height=200)
        self.painter_bcc, self.energy_label_bcc, self.meta_bcc = self._build_visualization_card("BCC", self.city_bcc, height=155)
        self.painter_accbcc, self.energy_label_accbcc, self.meta_accbcc = self._build_visualization_card("Combined", self.city_accbcc, height=155)

        self.painters = [self.painter_acc, self.painter_bcc, self.painter_accbcc]

    def _inline_metric(self, parent, col, title, value_text):
        holder = tk.Frame(parent, bg="#111827")
        holder.grid(row=0, column=col, sticky="ew", padx=(0, 10) if col < 2 else (0, 0))
        tk.Label(holder, text=title, fg="#94a3b8", bg="#111827", font=("Segoe UI", 8)).pack(anchor="w")
        value_label = tk.Label(holder, text=value_text, fg="#e2e8f0", bg="#111827", font=("Consolas", 12, "bold"))
        value_label.pack(anchor="w")
        return value_label

    def _build_visualization_card(self, title, city_obj, parent=None, height=150):
        card_parent = parent if parent is not None else self.visualizations_wrap
        card = tk.Frame(card_parent, bg="#111827", padx=10, pady=9)
        card.pack(fill="x", pady=(0, 12))

        title_label = tk.Label(card, text=title, fg="#f8fafc", bg="#111827", font=("Segoe UI", 11, "bold"))
        title_label.pack(anchor="w")

        metadata_label = tk.Label(card, text="Cars 00 | Avg v 0.00 m/s", fg="#94a3b8", bg="#111827", font=("Consolas", 9))
        metadata_label.pack(anchor="w", pady=(1, 6))

        painter = TransportationPainter(
            card,
            city_obj.roads,
            city_obj.cars,
            width=950,
            height=height,
            bg="#0b1220",
            highlightthickness=0,
        )
        painter.pack(fill="x", expand=True)

        energy_label = tk.Label(
            card,
            text="Energy 0.0000 kWh",
            fg="#94a3b8",
            bg="#111827",
            font=("Consolas", 9),
        )
        energy_label.pack(anchor="w", pady=(6, 0))

        return painter, energy_label, metadata_label

    def _resize_left_canvas(self, event):
        self.left_canvas.itemconfigure(self.controls_window, width=event.width)

    def _resize_right_canvas(self, event):
        self.right_canvas.itemconfigure(self.right_window, width=event.width)

    def _is_descendant_of(self, widget, ancestor):
        while widget is not None:
            if widget == ancestor:
                return True
            widget = widget.master
        return False

    def _on_mousewheel(self, event):
        hovered_widget = self.master.winfo_containing(event.x_root, event.y_root)
        if hovered_widget is None:
            return

        scroll_amount = int(-1 * (event.delta / 120))
        if self._is_descendant_of(hovered_widget, self.left_canvas):
            self.left_canvas.yview_scroll(scroll_amount, "units")
        elif self._is_descendant_of(hovered_widget, self.right_canvas):
            self.right_canvas.yview_scroll(scroll_amount, "units")

    def _handle_resize(self, _event=None):
        width = self.master.winfo_width()

        # Keep a true 30/70 split on medium+ screens and stack gracefully on narrow screens.
        if width < 1100:
            self.body.columnconfigure(0, weight=1)
            self.body.columnconfigure(1, weight=1)
            self.left_wrapper.grid_configure(row=0, column=0, padx=(0, 0), pady=(0, 12))
            self.right_wrapper.grid_configure(row=1, column=0)
            target_width = max(640, width - 80)
        else:
            self.body.columnconfigure(0, weight=22)
            self.body.columnconfigure(1, weight=78)
            self.left_wrapper.grid_configure(row=0, column=0, padx=(0, 12), pady=(0, 0))
            self.right_wrapper.grid_configure(row=0, column=1)
            target_width = max(760, int(width * 0.69))

        # All simulation panels use full visualization width.
        full_width = target_width

        if hasattr(self, "painter_acc"):
            self.painter_acc.config(width=full_width)
            self.painter_acc.repaint()

        if hasattr(self, "painter_bcc"):
            self.painter_bcc.config(width=full_width)
            self.painter_bcc.repaint()

        if hasattr(self, "painter_accbcc"):
            self.painter_accbcc.config(width=full_width)
            self.painter_accbcc.repaint()
    
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
        
        # Update panel texts and metrics
        self.energy_label_acc.config(text=f"Energy {total_energy_acc:.4f} kWh")
        self.energy_label_bcc.config(text=f"Energy {total_energy_bcc:.4f} kWh")
        self.energy_label_accbcc.config(text=f"Energy {total_energy_accbcc:.4f} kWh")

        avg_v_acc = sum(car.velocity for car in self.city_acc.cars) / len(self.city_acc.cars) if self.city_acc.cars else 0.0
        avg_v_bcc = sum(car.velocity for car in self.city_bcc.cars) / len(self.city_bcc.cars) if self.city_bcc.cars else 0.0
        avg_v_accbcc = sum(car.velocity for car in self.city_accbcc.cars) / len(self.city_accbcc.cars) if self.city_accbcc.cars else 0.0

        self.meta_acc.config(text=f"Cars {len(self.city_acc.cars):02d} | Avg v {avg_v_acc:05.2f} m/s")
        self.meta_bcc.config(text=f"Cars {len(self.city_bcc.cars):02d} | Avg v {avg_v_bcc:05.2f} m/s")
        self.meta_accbcc.config(text=f"Cars {len(self.city_accbcc.cars):02d} | Avg v {avg_v_accbcc:05.2f} m/s")

        total_energy_all = total_energy_acc + total_energy_bcc + total_energy_accbcc
        all_cars = self.city_acc.cars + self.city_bcc.cars + self.city_accbcc.cars
        avg_velocity_all = sum(car.velocity for car in all_cars) / len(all_cars) if all_cars else 0.0
        avg_abs_acc = sum(abs(car.acceleration) for car in all_cars) / len(all_cars) if all_cars else 0.0
        stability_score = max(0.0, min(100.0, 100.0 - (avg_abs_acc * 12.0)))

        self.metric_value_total.config(text=f"{total_energy_all:08.4f} kWh")
        self.metric_value_vel.config(text=f"{avg_velocity_all:05.2f} m/s")
        self.metric_value_stability.config(text=f"{stability_score:05.1f}")

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
