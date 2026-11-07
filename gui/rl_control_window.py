import os
import sys
import json
import threading

# Ensure project root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gui.control_window import ControlWindow
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv
from simulation.config import PROJECT_ROOT, MODELS_DIR, resolve_model_path
from evaluation.evaluate_all_scenarios import get_scenarios, load_csv_profile

# ==============================================================================
# INTUITIVENESS VERDICT CRITERIA CONFIGURATION
# Named criteria dict evaluating safety, passenger comfort, and string stability
# ==============================================================================
INTUITIVENESS_CRITERIA_CONFIG = {
    'zero_collisions': {
        'title': 'Zero Collisions',
        'desc': 'Total collision steps across all benchmark scenarios must be 0',
        'eval': lambda rl, classical, idm: rl.get('raw', {}).get('collisions', 999) == 0,
        'format_val': lambda rl, overall: f"{rl.get('raw', {}).get('collisions', 0)} collisions",
        'target': '0 collisions',
    },
    'minimum_gap': {
        'title': 'Minimum Gap Safety',
        'desc': 'Worst-case minimum bumper headway margin must be >= 2.00 m',
        'eval': lambda rl, classical, idm: rl.get('raw', {}).get('min_gap', 0.0) >= 2.00,
        'format_val': lambda rl, overall: f"{rl.get('raw', {}).get('min_gap', 0.0):.2f} m",
        'target': '>= 2.00 m',
    },
    'rms_jerk': {
        'title': 'RMS Jerk Comfort',
        'desc': 'Harmonic RMS jerk must meet automotive comfort grade (<= 0.60 m/s³)',
        'eval': lambda rl, classical, idm: rl.get('raw', {}).get('rms_jerk', 99.0) <= 0.60,
        'format_val': lambda rl, overall: f"{rl.get('raw', {}).get('rms_jerk', 0.0):.2f} m/s³",
        'target': '<= 0.60 m/s³',
    },
    'dii_vs_classical': {
        'title': 'DII vs Classical Controllers',
        'desc': 'Overall DII must exceed standard ACC controller baseline',
        'eval': lambda rl, classical, idm: rl.get('DII', 0.0) >= classical.get('ACC', {}).get('DII', 0.0),
        'format_val': lambda rl, overall: f"{rl.get('DII', 0.0):.1f} (vs ACC {overall.get('ACC', {}).get('DII', 0.0):.1f})",
        'target': '>= ACC DII',
    },
    'wave_ratio': {
        'title': 'Wave Ratio Damping',
        'desc': 'Upstream deceleration amplification must not exceed shock threshold (<= 1.25x)',
        'eval': lambda rl, classical, idm: rl.get('raw', {}).get('wave_ratio', 99.0) <= 1.25,
        'format_val': lambda rl, overall: f"{rl.get('raw', {}).get('wave_ratio', 0.0):.2f}x",
        'target': '<= 1.25x',
    },
}


class RLControlWindow(ControlWindow):
    def __init__(self, master):
        self.model = None
        self.model_loaded = False
        self.active_model_path = ""
        self.current_step = 0
        self.live_tiles = {}
        self.running_intuitiveness_stats = {}
        self._eval_running = False
        super().__init__(master)
        # Load any existing intuitiveness results into Intuitiveness tab
        self._load_intuitiveness_log()

    def _configure_style(self):
        super()._configure_style()
        style = ttk.Style(self.master)
        style.configure(
            "TCombobox",
            fieldbackground="#0b1220",
            background="#1e293b",
            foreground="#e2e8f0",
            darkcolor="#1e293b",
            lightcolor="#1e293b",
            arrowcolor="#cbd5e1",
            padding=4,
        )
        self.master.option_add("*TCombobox*Listbox.background", "#0b1220")
        self.master.option_add("*TCombobox*Listbox.foreground", "#e2e8f0")
        self.master.option_add("*TCombobox*Listbox.selectBackground", "#334155")
        self.master.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        # Sleek modern notebook tab styles
        style.configure("TNotebook", background="#0f172a", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#1e293b",
            foreground="#94a3b8",
            padding=[16, 6],
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#334155"), ("active", "#253347")],
            foreground=[("selected", "#38bdf8"), ("active", "#ffffff")]
        )

        # Treeview styles for intuitiveness table
        style.configure(
            "Intuitiveness.Treeview",
            background="#0b1220",
            foreground="#e2e8f0",
            fieldbackground="#0b1220",
            rowheight=26,
            font=("Segoe UI", 9)
        )
        style.configure(
            "Intuitiveness.Treeview.Heading",
            background="#1e293b",
            foreground="#38bdf8",
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "Intuitiveness.Treeview",
            background=[("selected", "#334155")],
            foreground=[("selected", "#ffffff")]
        )

    def _build_layout(self):
        super()._build_layout()
        # Bottom GUI status bar
        self.status_bar_frame = tk.Frame(self.root, bg="#111827", padx=10, pady=4)
        self.status_bar_frame.pack(fill="x", side="bottom", pady=(6, 0))
        self.status_bar = tk.Label(
            self.status_bar_frame,
            text="System Ready",
            fg="#94a3b8",
            bg="#111827",
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.status_bar.pack(side="left", fill="x", expand=True)

    def _build_controls(self):
        super()._build_controls()

        # Benchmark Scenario Selection Card
        scenario_card = tk.Frame(self.controls_container, bg="#111827", padx=10, pady=10)
        scenario_card.pack(fill="x", pady=(0, 10))

        ttk.Label(scenario_card, text="Scenario Selection", style="Section.TLabel").pack(anchor="w")
        ttk.Label(scenario_card, text="Select operational benchmark regime (1–15)", style="Meta.TLabel").pack(anchor="w", pady=(0, 8))

        self.benchmark_scenarios = get_scenarios()
        self.scenario_options = ["0. Default Free Simulation"] + [sc['name'] for sc in self.benchmark_scenarios]
        self.selected_scenario_var = tk.StringVar(value=self.scenario_options[0])

        self.scenario_combobox = ttk.Combobox(
            scenario_card,
            textvariable=self.selected_scenario_var,
            values=self.scenario_options,
            state="readonly",
        )
        self.scenario_combobox.pack(fill="x", pady=(0, 6))
        self.scenario_combobox.bind("<<ComboboxSelected>>", self._on_scenario_selected)

        self.scenario_info_label = ttk.Label(
            scenario_card,
            text="Mode: Free simulation using parameter values",
            style="Meta.TLabel",
        )
        self.scenario_info_label.pack(anchor="w")

        # RL Model Selection Card
        model_card = tk.Frame(self.controls_container, bg="#111827", padx=10, pady=10)
        model_card.pack(fill="x", pady=(0, 10))

        ttk.Label(model_card, text="RL Model Selection", style="Section.TLabel").pack(anchor="w")
        ttk.Label(model_card, text="Select active policy checkpoint", style="Meta.TLabel").pack(anchor="w", pady=(0, 8))

        self.available_models = self._scan_available_models()
        default_choice = list(self.available_models.keys())[0] if self.available_models else ""
        self.selected_model_var = tk.StringVar(value=default_choice)

        selector_row = ttk.Frame(model_card, style="Card.TFrame")
        selector_row.pack(fill="x", pady=(0, 6))
        selector_row.columnconfigure(0, weight=1)

        self.model_combobox = ttk.Combobox(
            selector_row,
            textvariable=self.selected_model_var,
            values=list(self.available_models.keys()),
            state="readonly",
        )
        self.model_combobox.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_dropdown_changed)

        self.browse_model_button = ttk.Button(
            selector_row,
            text="Browse...",
            command=self._browse_model_file,
        )
        self.browse_model_button.grid(row=0, column=1, sticky="e")

        self.model_status_label = ttk.Label(
            model_card,
            text="Initializing model...",
            style="Meta.TLabel",
        )
        self.model_status_label.pack(anchor="w")

        # Load initial model
        if default_choice and default_choice in self.available_models:
            self.load_rl_model(self.available_models[default_choice])

        # Promote Scenario Selection, Model Selection, and Simulation Actions to the top of the controls column
        first_widget = self.control_cards[0] if hasattr(self, 'control_cards') and self.control_cards else None
        action_card = getattr(self, 'action_card', None) or (self.run_button.master if hasattr(self, 'run_button') else None)
        if first_widget:
            scenario_card.pack(fill="x", pady=(0, 10), before=first_widget)
            model_card.pack(fill="x", pady=(0, 10), before=first_widget)
            if action_card:
                action_card.pack(fill="x", pady=(0, 12), before=first_widget)

    def _on_scenario_selected(self, _event=None):
        sel_idx = self.scenario_combobox.current()
        if sel_idx > 0 and sel_idx <= len(self.benchmark_scenarios):
            sc = self.benchmark_scenarios[sel_idx - 1]
            dur = sc.get('dur', 40.0)
            v_des = sc.get('v_des', 15.0)
            self.scenario_info_label.config(
                text=f"Dur: {dur:.0f}s | Target Speed: {v_des:.1f} m/s",
                foreground="#38bdf8"
            )
            # Pre-fill v_des entry
            if 'v_des' in self.entries:
                self.entries['v_des'].delete(0, tk.END)
                self.entries['v_des'].insert(0, str(v_des))
        else:
            self.scenario_info_label.config(
                text="Mode: Free simulation using parameter values",
                foreground="#94a3b8"
            )

    def _scan_available_models(self):
        models = {}
        default_path = resolve_model_path("td3_accel_agent.zip")
        if os.path.exists(default_path):
            models["Default (td3_accel_agent.zip)"] = os.path.abspath(default_path)

        models_dir = str(MODELS_DIR)
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for f in sorted(files):
                    if f.endswith(".zip") and f != "td3_accel_agent.zip":
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, start=str(PROJECT_ROOT)).replace("\\", "/")
                        label = f"models/{f}" if root == models_dir else rel_path
                        models[label] = full_path

        return models

    def _on_model_dropdown_changed(self, _event=None):
        choice = self.selected_model_var.get()
        if choice in self.available_models:
            self.load_rl_model(self.available_models[choice])
        elif os.path.exists(choice):
            self.load_rl_model(choice)

    def _browse_model_file(self):
        filename = filedialog.askopenfilename(
            title="Select TD3 Model Checkpoint",
            filetypes=[("Zip Model Files", "*.zip"), ("All Files", "*.*")],
        )
        if filename:
            short_name = os.path.basename(filename)
            self.available_models[short_name] = filename
            self.model_combobox["values"] = list(self.available_models.keys())
            self.selected_model_var.set(short_name)
            self.load_rl_model(filename)

    def load_rl_model(self, model_path):
        if not os.path.isabs(model_path):
            abs_path = os.path.abspath(model_path)
        else:
            abs_path = model_path

        if os.path.exists(abs_path):
            try:
                self.model = TD3.load(abs_path)
                self.model_loaded = True
                self.active_model_path = abs_path
                base_name = os.path.basename(abs_path)
                print(f"Successfully loaded RL model from {abs_path}")
                if hasattr(self, "model_status_label"):
                    self.model_status_label.config(
                        text=f"✓ Active: {base_name}",
                        foreground="#4ade80",
                    )
            except Exception as e:
                self.model = None
                self.model_loaded = False
                print(f"Error loading RL model: {e}")
                if hasattr(self, "model_status_label"):
                    self.model_status_label.config(
                        text=f"✗ Error: {e}",
                        foreground="#f87171",
                    )
        else:
            self.model = None
            self.model_loaded = False
            print(f"Warning: RL model '{abs_path}' not found.")
            if hasattr(self, "model_status_label"):
                self.model_status_label.config(
                    text=f"⚠ Not found: {os.path.basename(abs_path)}",
                    foreground="#fbbf24",
                )

    def _build_visualization_area(self):
        self.right_content.columnconfigure(0, weight=1)

        # Metrics strip
        self.metrics_strip = tk.Frame(self.right_content, bg="#111827", padx=12, pady=8)
        self.metrics_strip.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.metrics_strip.columnconfigure((0, 1, 2), weight=1)

        self.metric_value_total = self._inline_metric(self.metrics_strip, 0, "Energy", "0.0000 kWh")
        self.metric_value_vel = self._inline_metric(self.metrics_strip, 1, "Avg Velocity", "0.00 m/s")
        self.metric_value_stability = self._inline_metric(self.metrics_strip, 2, "Stability", "0.0")

        # Create City objects
        self.city_acc = RLCity()
        self.city_bcc = RLCity()
        self.city_accbcc = RLCity()
        self.city_rl = RLCity()

        # Modern Notebook Tabs:
        # Tab 1 = Road Simulation View (with Live Intuitiveness Tiles)
        # Tab 2 = Live Real-Time Telemetry & Gap Profiles
        # Tab 3 = Driver Intuitiveness & Benchmark Evaluation
        self.notebook = ttk.Notebook(self.right_content)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        # Tab 1: Road Simulation View
        self.tab_simulation = ttk.Frame(self.notebook, style="Pane.TFrame")
        self.notebook.add(self.tab_simulation, text="  Road Simulation  ")

        self.visualizations_wrap = ttk.Frame(self.tab_simulation, style="Pane.TFrame")
        self.visualizations_wrap.pack(fill="both", expand=True)

        # Build visualization cards on Tab 1 with live intuitiveness tiles
        self.painter_acc, self.energy_label_acc, self.meta_acc = self._build_visualization_card("ACC", self.city_acc, parent=self.visualizations_wrap, height=140)
        self.painter_bcc, self.energy_label_bcc, self.meta_bcc = self._build_visualization_card("BCC", self.city_bcc, parent=self.visualizations_wrap, height=140)
        self.painter_accbcc, self.energy_label_accbcc, self.meta_accbcc = self._build_visualization_card("ACC+BCC Integration", self.city_accbcc, parent=self.visualizations_wrap, height=140)
        
        rl_title = "RL TD3 Agent (Direct Acceleration)"
        self.painter_rl, self.energy_label_rl, self.meta_rl = self._build_visualization_card(rl_title, self.city_rl, parent=self.visualizations_wrap, height=140)

        self.painters = [self.painter_acc, self.painter_bcc, self.painter_accbcc, self.painter_rl]

        # Tab 2: Live Real-Time Telemetry & Gap Profiles
        self.tab_telemetry = ttk.Frame(self.notebook, style="Pane.TFrame")
        self.notebook.add(self.tab_telemetry, text="  Real-Time Telemetry & Gaps  ")
        self._build_realtime_telemetry_tab()

        # Tab 3: Driver Intuitiveness & Benchmark Suite
        self.tab_intuitiveness = ttk.Frame(self.notebook, style="Pane.TFrame")
        self.notebook.add(self.tab_intuitiveness, text="  Driver Intuitiveness & Benchmark  ")
        self._build_intuitiveness_tab()

    def _build_visualization_card(self, title, city_obj, parent=None, height=140):
        painter, energy_label, meta_label = super()._build_visualization_card(
            title, city_obj, parent=parent, height=height
        )
        card = energy_label.master

        # Identify controller key
        ctrl_key = "ACC"
        if "BCC" in title and "ACC+BCC" not in title and "+" not in title:
            ctrl_key = "BCC"
        elif "ACC+BCC" in title or "Integration" in title or "Combined" in title:
            ctrl_key = "ACC+BCC"
        elif "RL" in title:
            ctrl_key = "RL"

        # Live intuitiveness tiles container strip on Road Simulation card
        tiles_strip = tk.Frame(card, bg="#0b1220", padx=8, pady=5)
        tiles_strip.pack(fill="x", pady=(5, 0))
        tiles_strip.columnconfigure((0, 1, 2, 3, 4), weight=1)

        tile_defs = [
            ("jerk", "RMS Jerk", 0),
            ("flips", "Sign Flips", 1),
            ("wave", "Wave Ratio", 2),
            ("gap", "Min Gap", 3),
            ("col", "Collisions", 4),
        ]

        if not hasattr(self, 'live_tiles'):
            self.live_tiles = {}
        if not hasattr(self, 'running_intuitiveness_stats'):
            self.running_intuitiveness_stats = {}

        self.live_tiles[ctrl_key] = {}
        self.running_intuitiveness_stats[ctrl_key] = {
            'min_gap': float('inf'),
            'collisions': 0,
        }

        for key, label_text, col_idx in tile_defs:
            tile_box = tk.Frame(tiles_strip, bg="#0b1220", padx=4, pady=2)
            tile_box.grid(row=0, column=col_idx, sticky="ew", padx=3)

            tk.Label(
                tile_box,
                text=label_text,
                fg="#64748b",
                bg="#0b1220",
                font=("Segoe UI", 7, "bold")
            ).pack(anchor="w")

            val_lbl = tk.Label(
                tile_box,
                text="--",
                fg="#38bdf8",
                bg="#0b1220",
                font=("Consolas", 9, "bold")
            )
            val_lbl.pack(anchor="w")
            self.live_tiles[ctrl_key][key] = val_lbl

        return painter, energy_label, meta_label

    def _build_realtime_telemetry_tab(self):
        telemetry_header = tk.Frame(self.tab_telemetry, bg="#111827", padx=12, pady=8)
        telemetry_header.pack(fill="x", pady=(0, 4))
        
        title_box = tk.Frame(telemetry_header, bg="#111827")
        title_box.pack(side="left")
        
        tk.Label(
            title_box, 
            text="Live Real-Time Telemetry Stream", 
            fg="#f8fafc", 
            bg="#111827", 
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        
        tk.Label(
            title_box, 
            text="Real-time rendering of Velocity, Acceleration & Inter-Vehicle Gaps across all 4 controllers", 
            fg="#94a3b8", 
            bg="#111827", 
            font=("Segoe UI", 8)
        ).pack(anchor="w")

        btn_box = tk.Frame(telemetry_header, bg="#111827")
        btn_box.pack(side="right")
        
        self.live_badge = tk.Label(
            btn_box,
            text="LIVE STREAMING",
            fg="#22c55e",
            bg="#111827",
            font=("Consolas", 9, "bold"),
            padx=8
        )
        self.live_badge.pack(side="left", padx=(0, 10))

        ttk.Button(
            btn_box,
            text="Pop-out Window",
            command=self.plot_vel_acc_profiles
        ).pack(side="left")

        # Telemetry Matplotlib Figure (4 rows x 3 columns)
        self.telemetry_fig = Figure(figsize=(13.5, 9.8), dpi=95, facecolor="#0f172a")
        self.telemetry_fig.subplots_adjust(
            left=0.06, right=0.98, top=0.95, bottom=0.06, hspace=0.38, wspace=0.22
        )
        self.telemetry_axes = self.telemetry_fig.subplots(4, 3, sharex='col')

        self.telemetry_canvas = FigureCanvasTkAgg(self.telemetry_fig, master=self.tab_telemetry)
        canvas_widget = self.telemetry_canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        self._init_telemetry_artists()

    def _init_telemetry_artists(self):
        models = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC Integration", self.city_accbcc),
            ("RL Direct Acceleration", self.city_rl)
        ]
        
        self.telemetry_lines = {}
        min_dis = getattr(self, 'min_dis', 6.0)

        for r, (name, city) in enumerate(models):
            ax_v = self.telemetry_axes[r, 0]
            ax_a = self.telemetry_axes[r, 1]
            ax_g = self.telemetry_axes[r, 2]

            for ax in (ax_v, ax_a, ax_g):
                ax.clear()
                ax.set_facecolor("#0b1220")
                ax.grid(True, color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.8)
                ax.tick_params(colors="#94a3b8", labelsize=7.5)
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")

            ax_v.set_title(f"{name} Velocity", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_v.set_ylabel("Vel (m/s)", color="#94a3b8", fontsize=8)
            ax_v.set_ylim(-1.0, 32.0)

            ax_a.set_title(f"{name} Acceleration", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_a.set_ylabel("Accel (m/s²)", color="#94a3b8", fontsize=8)
            ax_a.set_ylim(-5.5, 3.5)

            ax_g.set_title(f"{name} Inter-Vehicle Gap", color="#e2e8f0", fontsize=8.5, fontweight="bold", pad=3)
            ax_g.set_ylabel("Gap (m)", color="#94a3b8", fontsize=8)
            ax_g.set_ylim(0.0, 45.0)

            # Dashed safety min_dis line
            ax_g.axhline(y=min_dis, color="#eab308", linestyle=":", linewidth=1.1, alpha=0.85)

            if r == 3:
                ax_v.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)
                ax_a.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)
                ax_g.set_xlabel("Time (s)", color="#cbd5e1", fontsize=8)

            lines_v, lines_a, lines_g = [], [], []
            cars = city.cars if city.cars else []
            for idx, car in enumerate(cars):
                color = "#ef4444" if idx == 0 else "#22c55e" if idx == 2 else "#64748b"
                lw = 1.4 if idx in (0, 2) else 0.75
                
                lv, = ax_v.plot([], [], color=color, linewidth=lw)
                la, = ax_a.plot([], [], color=color, linewidth=lw)
                lines_v.append(lv)
                lines_a.append(la)
                
                if idx > 0:
                    lg, = ax_g.plot([], [], color=color, linewidth=lw)
                    lines_g.append(lg)

            self.telemetry_lines[r] = {'vel': lines_v, 'acc': lines_a, 'gap': lines_g}

        self.telemetry_canvas.draw_idle()

    def _update_realtime_telemetry(self):
        if not hasattr(self, 'telemetry_lines') or not self.telemetry_lines:
            return
            
        dt = getattr(self, 'dt', 0.1)
        max_time = 5.0
        max_gap_seen = 35.0
        
        models = [
            (0, self.city_acc),
            (1, self.city_bcc),
            (2, self.city_accbcc),
            (3, self.city_rl)
        ]
        
        for r, city in models:
            lines_dict = self.telemetry_lines.get(r, None)
            if not lines_dict:
                continue
                
            for idx, car in enumerate(city.cars):
                n_v = len(car.vel_history)
                if n_v == 0:
                    continue
                time_axis = [dt * i for i in range(n_v)]
                if time_axis[-1] > max_time:
                    max_time = time_axis[-1]
                    
                if idx < len(lines_dict['vel']):
                    lines_dict['vel'][idx].set_data(time_axis, car.vel_history)
                if idx < len(lines_dict['acc']):
                    lines_dict['acc'][idx].set_data(time_axis, car.acc_history)
                if idx > 0 and (idx - 1) < len(lines_dict['gap']):
                    n_g = len(car.gap_history)
                    if n_g > 0:
                        t_gap = [dt * i for i in range(n_g)]
                        lines_dict['gap'][idx - 1].set_data(t_gap, car.gap_history)
                        cur_max_g = max(car.gap_history)
                        if cur_max_g > max_gap_seen:
                            max_gap_seen = cur_max_g

        xlim_right = max(5.0, max_time + 0.5)
        for ax in self.telemetry_axes.flat:
            ax.set_xlim(0, xlim_right)
            
        if max_gap_seen > 40.0:
            for r in range(4):
                self.telemetry_axes[r, 2].set_ylim(0, max_gap_seen + 5.0)

        self.telemetry_canvas.draw_idle()

    # ==========================================================================
    # TAB 3: DRIVER INTUITIVENESS & BENCHMARK SUITE
    # ==========================================================================
    def _build_intuitiveness_tab(self):
        canvas = tk.Canvas(self.tab_intuitiveness, bg="#0f172a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_intuitiveness, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#0f172a")

        scrollable_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Header Card (Model File Name, Date, and Run Evaluation Button)
        header_card = tk.Frame(scrollable_frame, bg="#111827", padx=16, pady=12)
        header_card.pack(fill="x", padx=12, pady=(10, 10))

        header_top = tk.Frame(header_card, bg="#111827")
        header_top.pack(fill="x")

        title_box = tk.Frame(header_top, bg="#111827")
        title_box.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_box,
            text="Driver Intuitiveness Metric Suite & Human Benchmark",
            fg="#f8fafc",
            bg="#111827",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        meta_row = tk.Frame(title_box, bg="#111827")
        meta_row.pack(anchor="w", pady=(4, 0))

        self.intuitiveness_model_label = tk.Label(
            meta_row,
            text="Model: --",
            fg="#38bdf8",
            bg="#111827",
            font=("Consolas", 9, "bold")
        )
        self.intuitiveness_model_label.pack(side="left", padx=(0, 16))

        self.intuitiveness_date_label = tk.Label(
            meta_row,
            text="Evaluated: --",
            fg="#94a3b8",
            bg="#111827",
            font=("Consolas", 9)
        )
        self.intuitiveness_date_label.pack(side="left")

        btn_box = tk.Frame(header_top, bg="#111827")
        btn_box.pack(side="right")

        self.btn_run_eval = ttk.Button(
            btn_box,
            text="▶ Run Evaluation",
            command=self._trigger_background_evaluation,
            style="Primary.TButton"
        )
        self.btn_run_eval.pack(side="right")

        # 2. Per-Controller Benchmark Table Card
        table_card = tk.Frame(scrollable_frame, bg="#111827", padx=16, pady=12)
        table_card.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            table_card,
            text="15-Scenario Comprehensive Benchmark Results",
            fg="#e2e8f0",
            bg="#111827",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(0, 8))

        columns = (
            "controller", "dii", "c1", "c2", "c3", "c4", "r_decoup",
            "rms_jerk", "flips", "wave", "min_gap", "collisions"
        )
        self.intuitiveness_tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=5,
            style="Intuitiveness.Treeview"
        )

        col_configs = [
            ("controller", "Controller", 140, "w"),
            ("dii", "DII (0-100)", 85, "center"),
            ("c1", "C1 Approp", 75, "center"),
            ("c2", "C2 Smooth", 75, "center"),
            ("c3", "C3 Calm", 75, "center"),
            ("c4", "C4 Comfort", 75, "center"),
            ("r_decoup", "R-Decoup", 80, "center"),
            ("rms_jerk", "RMS Jerk", 95, "center"),
            ("flips", "Flips/min", 75, "center"),
            ("wave", "Wave Ratio", 85, "center"),
            ("min_gap", "Min Gap", 85, "center"),
            ("collisions", "Collisions", 75, "center"),
        ]

        for cid, head, w, align in col_configs:
            self.intuitiveness_tree.heading(cid, text=head)
            self.intuitiveness_tree.column(cid, width=w, anchor=align)

        self.intuitiveness_tree.pack(fill="x", expand=True)

        self.intuitiveness_tree.tag_configure("rl_row", background="#1e293b", foreground="#38bdf8")
        self.intuitiveness_tree.tag_configure("ref_row", background="#172554", foreground="#93c5fd")
        self.intuitiveness_tree.tag_configure("norm_row", background="#0b1220", foreground="#e2e8f0")

        # 3. Verdict & Named Criteria Card
        verdict_card = tk.Frame(scrollable_frame, bg="#111827", padx=16, pady=12)
        verdict_card.pack(fill="x", padx=12, pady=(0, 16))

        # Verdict Header Banner
        banner_frame = tk.Frame(verdict_card, bg="#111827")
        banner_frame.pack(fill="x", pady=(0, 10))

        vbox_left = tk.Frame(banner_frame, bg="#111827")
        vbox_left.pack(side="left")

        tk.Label(
            vbox_left,
            text="Autonomous Policy Intuitiveness Verdict",
            fg="#f8fafc",
            bg="#111827",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        self.verdict_sublabel = tk.Label(
            vbox_left,
            text="Criteria dynamically evaluated from multi-scenario benchmark logs.",
            fg="#94a3b8",
            bg="#111827",
            font=("Segoe UI", 8)
        )
        self.verdict_sublabel.pack(anchor="w", pady=(2, 0))

        self.verdict_label = tk.Label(
            banner_frame,
            text="--",
            fg="#94a3b8",
            bg="#1f2937",
            font=("Consolas", 14, "bold"),
            padx=14,
            pady=4
        )
        self.verdict_label.pack(side="right")

        # Named Criteria Grid
        crit_grid = tk.Frame(verdict_card, bg="#111827")
        crit_grid.pack(fill="x", pady=(6, 0))
        crit_grid.columnconfigure(1, weight=1)

        row_idx = 0
        for key, spec in INTUITIVENESS_CRITERIA_CONFIG.items():
            badge_lbl = tk.Label(
                crit_grid,
                text="--",
                fg="#94a3b8",
                bg="#1f2937",
                font=("Consolas", 9, "bold"),
                width=6,
                pady=2
            )
            badge_lbl.grid(row=row_idx, column=0, padx=(0, 10), pady=4, sticky="w")
            setattr(self, f"crit_badge_{key}", badge_lbl)

            info_frame = tk.Frame(crit_grid, bg="#111827")
            info_frame.grid(row=row_idx, column=1, sticky="w", pady=4)

            tk.Label(
                info_frame,
                text=f"{spec['title']} (Target: {spec['target']})",
                fg="#e2e8f0",
                bg="#111827",
                font=("Segoe UI", 9, "bold")
            ).pack(anchor="w")

            tk.Label(
                info_frame,
                text=spec['desc'],
                fg="#64748b",
                bg="#111827",
                font=("Segoe UI", 8)
            ).pack(anchor="w")

            val_lbl = tk.Label(
                crit_grid,
                text="--",
                fg="#38bdf8",
                bg="#111827",
                font=("Consolas", 9, "bold")
            )
            val_lbl.grid(row=row_idx, column=2, padx=(10, 0), pady=4, sticky="e")
            setattr(self, f"crit_val_{key}", val_lbl)

            row_idx += 1

    def _trigger_background_evaluation(self):
        if getattr(self, '_eval_running', False):
            return
        self._eval_running = True
        self.btn_run_eval.config(state="disabled", text="⏳ Running Evaluation...")
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.config(
                text="Running full 15-scenario Driver Intuitiveness evaluation suite in background...",
                fg="#38bdf8"
            )

        threading.Thread(target=self._run_evaluation_worker, daemon=True).start()

    def _run_evaluation_worker(self):
        try:
            from evaluation.driver_intuitiveness import run_driver_intuitiveness_suite
            active_model = getattr(self, 'active_model_path', '')
            if not active_model or not os.path.exists(active_model):
                active_model = resolve_model_path('td3_accel_agent.zip')
            run_driver_intuitiveness_suite(model_path=active_model)
            self.master.after(0, self._on_evaluation_success)
        except Exception as e:
            self.master.after(0, lambda err=e: self._on_evaluation_error(err))

    def _on_evaluation_success(self):
        self._eval_running = False
        self.btn_run_eval.config(state="normal", text="▶ Run Evaluation")
        self._load_intuitiveness_log()
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.config(
                text="✓ Driver Intuitiveness evaluation completed and metrics updated.",
                fg="#4ade80"
            )

    def _on_evaluation_error(self, err):
        self._eval_running = False
        self.btn_run_eval.config(state="normal", text="▶ Run Evaluation")
        self._log_error(f"Evaluation failed: {err}")

    def _load_intuitiveness_log(self):
        log_path = os.path.join(str(PROJECT_ROOT), "logs", "driver_intuitiveness.json")
        if not os.path.exists(log_path):
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("metadata", {})
            model_path = meta.get("evaluated_model", "")
            model_name = os.path.basename(model_path) if model_path else "Unknown Model"
            timestamp = meta.get("timestamp", "Unknown Date")

            if hasattr(self, "intuitiveness_model_label"):
                self.intuitiveness_model_label.config(text=f"Model: {model_name}")
            if hasattr(self, "intuitiveness_date_label"):
                self.intuitiveness_date_label.config(text=f"Evaluated: {timestamp}")

            overall = data.get("overall", {})
            if hasattr(self, "intuitiveness_tree"):
                for item in self.intuitiveness_tree.get_children():
                    self.intuitiveness_tree.delete(item)

                controllers_display = [
                    ("ACC", "ACC"),
                    ("BCC", "BCC"),
                    ("ACC+BCC", "ACC+BCC"),
                    ("RL", "RL (TD3)"),
                    ("IDM_gap_matched", "IDM (Reference)"),
                ]

                for key, disp_name in controllers_display:
                    row_data = overall.get(key)
                    if not row_data:
                        continue
                    raw = row_data.get("raw", {})
                    dii = f"{row_data.get('DII', 0.0):.1f}"
                    c1 = f"{row_data.get('C1', 0.0):.2f}"
                    c2 = f"{row_data.get('C2', 0.0):.2f}"
                    c3 = f"{row_data.get('C3', 0.0):.2f}"
                    c4 = f"{row_data.get('C4', 0.0):.2f}"
                    r_dec = f"{row_data.get('rear_decoupling', raw.get('rear_decoupling', 1.0)):.2f}"
                    rms_j = f"{raw.get('rms_jerk', 0.0):.2f} m/s³"
                    flips = f"{raw.get('sign_changes_per_min', 0.0):.1f}"
                    wave = f"{raw.get('wave_ratio', 0.0):.2f}x"
                    mg = f"{raw.get('min_gap', 0.0):.2f} m"
                    cols = f"{raw.get('collisions', 0)}"

                    tag = "rl_row" if key == "RL" else "ref_row" if "IDM" in key else "norm_row"
                    self.intuitiveness_tree.insert(
                        "", "end",
                        values=(disp_name, dii, c1, c2, c3, c4, r_dec, rms_j, flips, wave, mg, cols),
                        tags=(tag,)
                    )

            # Compute and update verdict and criteria
            self._compute_and_update_verdict(overall)

        except Exception as e:
            self._log_error(f"Error loading intuitiveness log: {e}")

    def _compute_and_update_verdict(self, overall):
        rl_data = overall.get("RL")
        if not rl_data or not hasattr(self, "verdict_label"):
            return

        all_passed = True
        for key, spec in INTUITIVENESS_CRITERIA_CONFIG.items():
            passed = bool(spec['eval'](rl_data, overall, overall.get('IDM_gap_matched', {})))
            if not passed:
                all_passed = False

            lbl_badge = getattr(self, f"crit_badge_{key}", None)
            lbl_val = getattr(self, f"crit_val_{key}", None)

            if lbl_badge:
                if passed:
                    lbl_badge.config(text="PASS", fg="#22c55e", bg="#064e3b")
                else:
                    lbl_badge.config(text="FAIL", fg="#ef4444", bg="#7f1d1d")
            if lbl_val:
                lbl_val.config(text=spec['format_val'](rl_data, overall))

        # Final verdict text: "The text must never be typed in."
        verdict_str = "INTUITIVE" if all_passed else "NOT INTUITIVE"
        if all_passed:
            self.verdict_label.config(
                text=verdict_str,
                fg="#22c55e",
                bg="#064e3b"
            )
            if hasattr(self, "verdict_sublabel"):
                self.verdict_sublabel.config(
                    text="Policy satisfies all 5 automotive comfort, safety & intuitiveness criteria.",
                    fg="#86efac"
                )
        else:
            self.verdict_label.config(
                text=verdict_str,
                fg="#ef4444",
                bg="#7f1d1d"
            )
            if hasattr(self, "verdict_sublabel"):
                self.verdict_sublabel.config(
                    text="Policy failed one or more intuitiveness / safety criteria. Model retraining required.",
                    fg="#fca5a5"
                )

    def _log_error(self, message):
        print(f"[GUI ERROR] {message}", file=sys.stderr)
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.config(text=f"⚠ {message}", fg="#f87171")

    def _on_notebook_tab_changed(self, _event=None):
        try:
            if hasattr(self, 'notebook'):
                active_tab_idx = self.notebook.index(self.notebook.select())
                if active_tab_idx == 1:
                    self._update_realtime_telemetry()
                elif active_tab_idx == 2:
                    self._load_intuitiveness_log()
        except Exception as e:
            self._log_error(f"Tab switch update failed: {e}")

    def run_simulation(self):
        # Get parameter values from entry fields
        args = []
        param_keys = ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time", "headway_time", "max_a", "min_a", "min_gap", "dt"]
        for key in param_keys:
            val = self.entries[key].get()
            try:
                val = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except Exception:
                val = 0
            args.append(val)

        self.dt = args[-1]
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

        # Refresh real-time telemetry artists with new car fleet
        if hasattr(self, 'telemetry_canvas'):
            self._init_telemetry_artists()

        # Reset live intuitiveness statistics & tiles
        if hasattr(self, 'live_tiles'):
            for name in self.live_tiles:
                self.running_intuitiveness_stats[name] = {'min_gap': float('inf'), 'collisions': 0}
                for tile in self.live_tiles[name].values():
                    tile.config(text="--")

        # Check if a benchmark scenario is selected from the dropdown
        sel_idx = self.scenario_combobox.current() if hasattr(self, 'scenario_combobox') else 0
        if sel_idx > 0 and sel_idx <= len(self.benchmark_scenarios):
            sc = self.benchmark_scenarios[sel_idx - 1]

            # Configure lead velocity profile
            lead_profile = []
            if sc['type'] == 'csv':
                lead_profile = load_csv_profile(sc['csv_file'])
            elif sc['type'] == 'synthetic':
                lead_profile = sc['profile']
            elif sc['type'] == 'wave':
                lead_profile = [
                    (t, 15.0 + 4.0 * np.sin(2 * np.pi * t / 15.0))
                    for t in np.arange(0, sc['dur'] + 1.0, 0.5)
                ]

            follower_profile = sc.get('follower_profile', None)
            rear_cutout_time = sc.get('rear_cutout_time', None)

            for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                city.lead_velocity_profile = list(lead_profile) if lead_profile else []
                city.follower_velocity_profile = list(follower_profile) if follower_profile else []
                city.rear_cutout_time = rear_cutout_time

            # Configure RL mode: car 0 is leader, car N-1 (if follower profile) is human follower
            for i, car in enumerate(self.city_rl.cars):
                if i == 0:
                    car.is_rl = False
                    car.mode = 'ACC'
                elif i == len(self.city_rl.cars) - 1 and follower_profile:
                    car.is_rl = False
                    car.mode = 'ACC'
                else:
                    car.is_rl = True
                    car.rl_action = 0.0

            if hasattr(self, 'status_bar') and self.status_bar:
                self.status_bar.config(
                    text=f"▶ Running {sc['name']} (Dur: {sc['dur']:.0f}s, Target: {sc['v_des']:.1f} m/s)",
                    fg="#38bdf8"
                )
        else:
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
                    city.rear_cutout_time = None
            else:
                for city in [self.city_acc, self.city_bcc, self.city_accbcc, self.city_rl]:
                    city.lead_velocity_profile = []
                    city.follower_velocity_profile = []
                    city.rear_cutout_time = None
            if hasattr(self, 'status_bar') and self.status_bar:
                self.status_bar.config(
                    text="▶ Running Free Manual Simulation",
                    fg="#94a3b8"
                )

        self.current_step = 0
        self.start_timer()

    def _update_live_intuitiveness_tiles(self):
        if not hasattr(self, 'live_tiles') or not self.live_tiles:
            return

        # Check data sufficiency (at least 3 steps of simulation)
        if getattr(self, 'current_step', 0) < 3:
            for name in self.live_tiles:
                for tile in self.live_tiles[name].values():
                    if tile.cget("text") != "--":
                        tile.config(text="--")
            return

        dt = getattr(self, 'dt', 0.1)
        controllers = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC", self.city_accbcc),
            ("RL", self.city_rl),
        ]

        for name, city in controllers:
            if name not in self.live_tiles or not city.cars or len(city.cars) < 2:
                continue

            tiles = self.live_tiles[name]
            stats = self.running_intuitiveness_stats.setdefault(name, {'min_gap': float('inf'), 'collisions': 0})
            road_len = city.roads[0].length if city.roads else 1000.0

            # 1. Step gaps for follower cars using the evaluation's gap calculation
            step_gaps = []
            for car in city.cars[1:]:
                cars_same = [c for c in city.cars if c.current_road == car.current_road and c != car]
                def gap_to(other):
                    g = (car.pos - other.pos) % road_len
                    return g if g > 0 else float('inf')
                front = min(cars_same, key=gap_to, default=None)
                fg = (car.pos - front.pos - front.length) % road_len if front else road_len
                step_gaps.append(fg)

            if step_gaps:
                cur_min_g = min(step_gaps)
                if cur_min_g < stats['min_gap']:
                    stats['min_gap'] = cur_min_g
                if (cur_min_g < 0.5) or getattr(city, 'collided_this_step', False):
                    stats['collisions'] += 1

            # 2. RMS Jerk across all followers (diff of acc_history / dt)
            all_jerks = []
            for car in city.cars[1:]:
                accs = np.array(car.acc_history, dtype=np.float64)
                if len(accs) > 1:
                    all_jerks.extend((np.diff(accs) / dt).tolist())
            rms_jerk = float(np.sqrt(np.mean(np.array(all_jerks) ** 2))) if all_jerks else 0.0

            # 3. Sign flips per minute (ignoring |a| <= 0.1 deadband)
            flips_list = []
            num_steps = len(city.cars[1].acc_history) if len(city.cars) > 1 else self.current_step
            for car in city.cars[1:]:
                active_signs = [np.sign(a) for a in car.acc_history if abs(a) > 0.1]
                flips = sum(1 for k in range(1, len(active_signs)) if active_signs[k] != active_signs[k - 1])
                flips_list.append(flips)
            elapsed_min = max(0.01, (num_steps - 1) * dt / 60.0)
            sign_flips_per_min = (float(np.mean(flips_list)) if flips_list else 0.0) / elapsed_min

            # 4. Wave Ratio: max follower decel / max(leader decel, 0.1)
            leader_acc = np.array(city.cars[0].acc_history, dtype=np.float64) if city.cars else np.array([])
            max_lead_decel = float(np.max(np.maximum(0.0, -leader_acc))) if len(leader_acc) > 0 else 0.0
            all_foll_acc = []
            for car in city.cars[1:]:
                all_foll_acc.extend(car.acc_history)
            max_foll_decel = float(np.max(np.maximum(0.0, -np.array(all_foll_acc)))) if all_foll_acc else 0.0
            eff_lead_decel = max(max_lead_decel, 0.1)
            wave_ratio = max_foll_decel / eff_lead_decel

            # 5. Update labels
            tiles['jerk'].config(text=f"{rms_jerk:.2f} m/s³")
            tiles['flips'].config(text=f"{sign_flips_per_min:.1f} /min")
            tiles['wave'].config(text=f"{wave_ratio:.2f}x")
            mg = stats['min_gap']
            tiles['gap'].config(text=f"{mg:.2f} m" if mg < 1e5 else "--")
            tiles['col'].config(text=f"{stats['collisions']}")

    def update_simulation(self):
        dt = getattr(self, 'dt', 0.1)
        try:
            self.current_step += 1
            
            # Handle standard logic for ACC, BCC, and ACC+BCC
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
                    obs = self._get_rl_observation(self.city_rl)
                    action, _ = self.model.predict(obs, deterministic=True)
                    for i, car in enumerate(self.city_rl.cars[1:]):
                        if i < len(action):
                            car.rl_action = float(action[i][0])
                except Exception as e:
                    if self.current_step <= 1: 
                        print(f"RL Prediction/Observation error: {e}")
                
                self.city_rl.run(dt)

            # Redraw all painters
            for p in self.painters:
                p.repaint()

            # Update metrics and energy labels
            total_energy_acc = sum(car.energy_used for car in self.city_acc.cars)
            total_energy_bcc = sum(car.energy_used for car in self.city_bcc.cars)
            total_energy_accbcc = sum(car.energy_used for car in self.city_accbcc.cars)
            total_energy_rl = sum(car.energy_used for car in self.city_rl.cars)

            self.energy_label_acc.config(text=f"Energy {total_energy_acc:.4f} kWh")
            self.energy_label_bcc.config(text=f"Energy {total_energy_bcc:.4f} kWh")
            self.energy_label_accbcc.config(text=f"Energy {total_energy_accbcc:.4f} kWh")
            self.energy_label_rl.config(text=f"Energy {total_energy_rl:.4f} kWh")

            def get_avg_v(city):
                return sum(car.velocity for car in city.cars) / len(city.cars) if city.cars else 0.0

            self.meta_acc.config(text=f"Cars {len(self.city_acc.cars):02d} | Avg v {get_avg_v(self.city_acc):05.2f} m/s")
            self.meta_bcc.config(text=f"Cars {len(self.city_bcc.cars):02d} | Avg v {get_avg_v(self.city_bcc):05.2f} m/s")
            self.meta_accbcc.config(text=f"Cars {len(self.city_accbcc.cars):02d} | Avg v {get_avg_v(self.city_accbcc):05.2f} m/s")
            self.meta_rl.config(text=f"Cars {len(self.city_rl.cars):02d} | Avg v {get_avg_v(self.city_rl):05.2f} m/s")

            # Global metrics strip update
            total_energy_all = total_energy_acc + total_energy_bcc + total_energy_accbcc + total_energy_rl
            self.metric_value_total.config(text=f"{total_energy_all:08.4f} kWh")
            self.metric_value_vel.config(text=f"{get_avg_v(self.city_rl):05.2f} m/s")

            # Capture video frame if recording is active
            if hasattr(self, "recorder") and self.recorder.is_recording:
                target = getattr(self, "root", self.master)
                self.recorder.capture_frame(target)
                if self.recorder.frame_count % 10 == 0:
                    self.record_status_label.config(text=f"🔴 Rec: {self.recorder.frame_count} frames")

            # Stream live real-time telemetry if user is viewing Tab 2
            if hasattr(self, "notebook"):
                try:
                    active_tab_idx = self.notebook.index(self.notebook.select())
                    if active_tab_idx == 1 and (self.current_step % 2 == 0):
                        self._update_realtime_telemetry()
                except Exception:
                    pass

            # Update live intuitiveness tiles on Tab 1 (guarded against exceptions)
            try:
                self._update_live_intuitiveness_tiles()
            except Exception as e_tiles:
                self._log_error(f"Live intuitiveness tiles error: {e_tiles}")

        except Exception as e:
            self._log_error(f"Simulation step error: {e}")
        finally:
            # Rule: update_simulation must always reschedule itself unconditionally
            self.timer = self.master.after(int(dt * 1000), self.update_simulation)

    def _get_rl_observation(self, city):
        v_des = getattr(city, 'v_des', 15.0)
        return AccelTrafficEnv.get_city_obs(city, v_des)

    def plot_vel_acc_profiles(self):
        dt = getattr(self, 'dt', 0.1)

        fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex='col')

        models = [
            ("ACC", self.city_acc),
            ("BCC", self.city_bcc),
            ("ACC+BCC Integration", self.city_accbcc),
            ("RL Direct Acceleration", self.city_rl)
        ]

        min_dis = getattr(self, 'min_dis', 6.0)

        for row, (name, city) in enumerate(models):
            # 1. Velocity (Col 0)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.vel_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 0].plot(time_axis, car.vel_history, color=color, linewidth=linewidth)
            
            axes[row, 0].set_title(f"{name} Velocity")
            axes[row, 0].set_ylabel("Velocity (m/s)")
            axes[row, 0].grid(True)
            if row == 3: axes[row, 0].set_xlabel("Time (s)")

            # 2. Acceleration (Col 1)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.acc_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx in (0, 2) else 0.5
                axes[row, 1].plot(time_axis, car.acc_history, color=color, linewidth=linewidth)
                
            axes[row, 1].set_title(f"{name} Acceleration")
            axes[row, 1].set_ylabel("Accel (m/s²)")
            axes[row, 1].grid(True)
            if row == 3: axes[row, 1].set_xlabel("Time (s)")

            # 3. Inter-Vehicle Gap (Col 2)
            for idx, car in enumerate(city.cars):
                if idx == 0:
                    continue
                time_axis = [dt * i for i in range(len(car.gap_history))]
                color = "green" if idx == 2 else "grey"
                linewidth = 1.5 if idx == 2 else 0.5
                axes[row, 2].plot(time_axis, car.gap_history, color=color, linewidth=linewidth)
                
            axes[row, 2].axhline(y=min_dis, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label=f"min_dis ({min_dis}m)")
            axes[row, 2].set_title(f"{name} Inter-Vehicle Gap")
            axes[row, 2].set_ylabel("Gap (m)")
            axes[row, 2].grid(True)
            if row == 3: axes[row, 2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("RL-Enhanced Traffic Simulation")
    app = RLControlWindow(root)
    root.mainloop()
