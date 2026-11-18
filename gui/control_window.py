'''
Control_window.py: Contains the ControlWindow class for managing the traffic simulation GUI.
Light theme dashboard with unified smooth-edged side panel and modern embedded title fields.
'''

import os
import sys

# Ensure project root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import csv
from PIL import Image, ImageTk, ImageDraw
from simulation.city import City
from gui.transportation_painter import TransportationPainter
from gui.video_recorder import VideoRecorder

def _make_pill_image(master, width, height, radius, fill, outline=None, outline_width=1, parent_bg="#ffffff"):
    """Generates an antialiased 9-slice image with smooth rounded edges blended seamlessly into parent_bg."""
    scale = 4
    im = Image.new("RGBA", (width * scale, height * scale), parent_bg)
    d = ImageDraw.Draw(im)
    if outline:
        d.rounded_rectangle(
            [0, 0, width * scale - 1, height * scale - 1],
            radius=radius * scale,
            fill=fill,
            outline=outline,
            width=int(outline_width * scale),
        )
    else:
        d.rounded_rectangle(
            [0, 0, width * scale - 1, height * scale - 1],
            radius=radius * scale,
            fill=fill,
        )
    im = im.resize((width, height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(im, master=master)


class AdaptiveRoundedField(tk.Canvas):
    """
    Modern parameter input field with the title positioned as small text
    at the top of the input field container itself, with smooth rounded edges
    and no borders.
    """
    def __init__(self, master, label_text, default_val="", **kwargs):
        super().__init__(master, height=48, bg="#ffffff", highlightthickness=0, **kwargs)
        self.label_text = label_text
        self.default_val = str(default_val)
        self.r = 7
        self.bg_poly = None
        self._focused = False

        self.title_lbl = tk.Label(
            self,
            text=label_text.upper(),
            bg="#f1f5f9",
            fg="#64748b",
            font=("Segoe UI", 7, "bold"),
            anchor="w"
        )
        self.title_win = self.create_window(10, 5, window=self.title_lbl, anchor="nw")

        self.entry = tk.Entry(
            self,
            bg="#f1f5f9",
            fg="#0f172a",
            font=("Consolas", 10, "bold"),
            bd=0,
            relief="flat",
            highlightthickness=0
        )
        self.entry.insert(0, self.default_val)
        self.entry_win = self.create_window(10, 23, window=self.entry, anchor="nw")

        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", lambda _e: self.entry.focus_set())
        self.title_lbl.bind("<Button-1>", lambda _e: self.entry.focus_set())

        self.entry.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.entry.bind("<FocusOut>", lambda _e: self._set_focus(False))

    def _set_focus(self, focused):
        self._focused = focused
        self._redraw()

    def _on_resize(self, event):
        self._redraw(event.width, event.height)

    def _redraw(self, w=None, h=None):
        w = w or self.winfo_width()
        h = h or self.winfo_height()
        if w <= 2 or h <= 2:
            return
        if self.bg_poly:
            self.delete(self.bg_poly)

        r = self.r
        x1, y1, x2, y2 = 1, 1, w - 2, h - 2
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        outline = "#0284c7" if self._focused else ""
        width = 1.5 if self._focused else 0
        fill = "#ffffff" if self._focused else "#f1f5f9"
        self.bg_poly = self.create_polygon(points, smooth=True, fill=fill, outline=outline, width=width)
        self.tag_lower(self.bg_poly)

        title_color = "#0284c7" if self._focused else "#64748b"
        self.title_lbl.configure(bg=fill, fg=title_color)
        self.entry.configure(bg=fill)
        self.itemconfigure(self.entry_win, width=max(20, w - 20))


class ControlWindow:
    def __init__(self, master):
        self.master = master
        self.recorder = VideoRecorder()
        self._style_images = []
        self._scroll_targets = {}
        self._scroll_animating = set()

        self.master.configure(bg="#f8fafc")
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
            "sensor_delay": 0.0,
            "sensor_noise": 0.0,
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

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.master.bind("<Configure>", self._handle_resize)
        self._handle_resize()

    def _configure_style(self):
        style = ttk.Style(self.master)
        style.theme_use("clam")

        bg_root = "#f8fafc"
        bg_pane = "#f1f5f9"
        bg_card = "#ffffff"
        fg_text = "#0f172a"
        fg_subtext = "#64748b"
        fg_field = "#334155"

        self.master.configure(bg=bg_root)

        style.configure("Root.TFrame", background=bg_root)
        style.configure("Pane.TFrame", background=bg_pane)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("SidePanel.TFrame", background=bg_card)

        style.configure("Title.TLabel", background=bg_root, foreground=fg_text, font=("Segoe UI", 22, "bold"))
        style.configure("SubTitle.TLabel", background=bg_root, foreground=fg_subtext, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=bg_card, foreground=fg_text, font=("Segoe UI", 10, "bold"))
        style.configure("Meta.TLabel", background=bg_card, foreground=fg_subtext, font=("Segoe UI", 9))
        style.configure("FieldLabel.TLabel", background=bg_card, foreground=fg_field, font=("Segoe UI", 9))

        # Register smooth rounded 9-slice elements (borderless with smooth corners)
        if "SmoothEntry.bg" not in style.element_names():
            img_entry_norm = _make_pill_image(self.master, 32, 32, 7, "#f1f5f9")
            img_entry_focus = _make_pill_image(self.master, 32, 32, 7, "#ffffff", outline="#0284c7", outline_width=1.5)
            self._style_images.extend([img_entry_norm, img_entry_focus])

            style.element_create("SmoothEntry.bg", "image", img_entry_norm, ("focus", img_entry_focus), border=8, sticky="nsew")
            style.layout("TEntry", [
                ("SmoothEntry.bg", {"sticky": "nswe", "children": [
                    ("Entry.padding", {"sticky": "nswe", "children": [
                        ("Entry.textarea", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "TEntry",
            padding=[10, 7],
            foreground=fg_text,
            insertcolor=fg_text,
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10)
        )

        if "SmoothCombobox.bg" not in style.element_names():
            img_cb_norm = _make_pill_image(self.master, 32, 32, 7, "#f1f5f9")
            img_cb_focus = _make_pill_image(self.master, 32, 32, 7, "#ffffff", outline="#0284c7", outline_width=1.5)
            self._style_images.extend([img_cb_norm, img_cb_focus])

            style.element_create("SmoothCombobox.bg", "image", img_cb_norm, ("focus", img_cb_focus), border=8, sticky="nsew")
            style.layout("TCombobox", [
                ("SmoothCombobox.bg", {"sticky": "nswe", "children": [
                    ("Combobox.rightdownarrow", {"side": "right", "sticky": "ns"}),
                    ("Combobox.padding", {"sticky": "nswe", "children": [
                        ("Combobox.textarea", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "TCombobox",
            padding=[10, 8],
            foreground=fg_text,
            background="#ffffff",
            fieldbackground="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            focusfill="#ffffff",
            arrowcolor=fg_text,
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9)
        )
        style.map(
            "TCombobox",
            background=[("active", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            fieldbackground=[("active", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            bordercolor=[("active", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            lightcolor=[("active", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            darkcolor=[("active", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
        )
        self.master.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.master.option_add("*TCombobox*Listbox.foreground", "#0f172a")
        self.master.option_add("*TCombobox*Listbox.selectBackground", "#e2e8f0")
        self.master.option_add("*TCombobox*Listbox.selectForeground", "#0f172a")
        self.master.option_add("*TCombobox*Listbox.relief", "flat")
        self.master.option_add("*TCombobox*Listbox.borderWidth", "0")

        # Smooth Buttons (rounded edges, no harsh borders)
        if "SmoothBtn.bg" not in style.element_names():
            img_btn_norm = _make_pill_image(self.master, 32, 32, 8, "#e2e8f0")
            img_btn_act = _make_pill_image(self.master, 32, 32, 8, "#cbd5e1")
            self._style_images.extend([img_btn_norm, img_btn_act])

            style.element_create("SmoothBtn.bg", "image", img_btn_norm, ("active", img_btn_act), border=9, sticky="nsew")
            style.layout("TButton", [
                ("SmoothBtn.bg", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "TButton",
            padding=[12, 8],
            foreground=fg_text,
            background="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            focuscolor="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9)
        )
        style.map(
            "TButton",
            background=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            bordercolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            lightcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            darkcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )

        if "SmoothPrimBtn.bg" not in style.element_names():
            img_prim_norm = _make_pill_image(self.master, 32, 32, 8, "#0284c7")
            img_prim_act = _make_pill_image(self.master, 32, 32, 8, "#0369a1")
            self._style_images.extend([img_prim_norm, img_prim_act])

            style.element_create("SmoothPrimBtn.bg", "image", img_prim_norm, ("active", img_prim_act), border=9, sticky="nsew")
            style.layout("Primary.TButton", [
                ("SmoothPrimBtn.bg", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "Primary.TButton",
            padding=[14, 9],
            foreground="#ffffff",
            background="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            focuscolor="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            bordercolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            lightcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            darkcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )

        if "SmoothRecBtn.bg" not in style.element_names():
            img_rec_norm = _make_pill_image(self.master, 32, 32, 8, "#fee2e2")
            img_rec_act = _make_pill_image(self.master, 32, 32, 8, "#fecaca")
            self._style_images.extend([img_rec_norm, img_rec_act])

            style.element_create("SmoothRecBtn.bg", "image", img_rec_norm, ("active", img_rec_act), border=9, sticky="nsew")
            style.layout("Record.TButton", [
                ("SmoothRecBtn.bg", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "Record.TButton",
            padding=[12, 8],
            foreground="#dc2626",
            background="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            focuscolor="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9)
        )
        style.map(
            "Record.TButton",
            background=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            bordercolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            lightcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            darkcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )

        if "SmoothRecActBtn.bg" not in style.element_names():
            img_recact_norm = _make_pill_image(self.master, 32, 32, 8, "#dc2626")
            img_recact_act = _make_pill_image(self.master, 32, 32, 8, "#b91c1c")
            self._style_images.extend([img_recact_norm, img_recact_act])

            style.element_create("SmoothRecActBtn.bg", "image", img_recact_norm, ("active", img_recact_act), border=9, sticky="nsew")
            style.layout("RecordActive.TButton", [
                ("SmoothRecActBtn.bg", {"sticky": "nswe", "children": [
                    ("Button.padding", {"sticky": "nswe", "children": [
                        ("Button.label", {"sticky": "nswe"})
                    ]})
                ]})
            ])

        style.configure(
            "RecordActive.TButton",
            padding=[12, 8],
            foreground="#ffffff",
            background="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            focuscolor="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9, "bold")
        )
        style.map(
            "RecordActive.TButton",
            background=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            bordercolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            lightcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            darkcolor=[("active", "#ffffff"), ("pressed", "#ffffff"), ("focus", "#ffffff"), ("!active", "#ffffff")],
            relief=[("pressed", "flat"), ("!pressed", "flat")]
        )

        # Smooth, bright, wide, borderless Scrollbar
        if "SmoothScroll.thumb" not in style.element_names():
            img_thumb_norm = _make_pill_image(self.master, 16, 32, 8, "#cbd5e1")
            img_thumb_act = _make_pill_image(self.master, 16, 32, 8, "#94a3b8")
            self._style_images.extend([img_thumb_norm, img_thumb_act])

            style.element_create("SmoothScroll.thumb", "image", img_thumb_norm, ("active", img_thumb_act), border=8, sticky="nswe")
            style.layout("Vertical.TScrollbar", [
                ("Vertical.Scrollbar.trough", {"sticky": "nswe", "children": [
                    ("SmoothScroll.thumb", {"sticky": "nswe"})
                ]})
            ])

        style.configure(
            "Vertical.TScrollbar",
            troughcolor="#ffffff",
            background="#ffffff",
            bordercolor="#ffffff",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            width=16,
            relief="flat",
            borderwidth=0
        )

        style.configure("TCheckbutton", background=bg_card, foreground=fg_text, focuscolor="#ffffff")
        style.map("TCheckbutton", background=[("active", bg_card), ("!active", bg_card)])

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

        # Single continuous side panel container (replaces disconnected card layout)
        self.left_wrapper = tk.Frame(
            self.body,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#e2e8f0",
            highlightcolor="#e2e8f0"
        )
        self.left_wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.left_wrapper.rowconfigure(0, weight=1)
        self.left_wrapper.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(self.left_wrapper, bg="#ffffff", highlightthickness=0, yscrollincrement=1)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")

        # Wide, bright, padded scrollbar with no borders
        self.left_scroll = ttk.Scrollbar(self.left_wrapper, orient="vertical", command=self.left_canvas.yview)
        self.left_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 6), pady=6)
        self.left_scroll.bind("<Button-1>", lambda _e: self._cancel_scroll_anim(self.left_canvas), add="+")
        self.left_canvas.configure(yscrollcommand=self.left_scroll.set)

        self.controls_container = tk.Frame(self.left_canvas, bg="#ffffff", padx=16, pady=16)
        self.controls_window = self.left_canvas.create_window((0, 0), window=self.controls_container, anchor="nw")

        self.controls_container.bind("<Configure>", lambda _e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind("<Configure>", self._resize_left_canvas)

        # Right visualization column
        self.right_wrapper = ttk.Frame(self.body, style="Pane.TFrame")
        self.right_wrapper.grid(row=0, column=1, sticky="nsew")
        self.right_wrapper.rowconfigure(0, weight=1)
        self.right_wrapper.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(self.right_wrapper, bg="#f1f5f9", highlightthickness=0, yscrollincrement=1)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")

        # Right column scrollbar with same wide, bright, padded styling
        self.right_scroll = ttk.Scrollbar(self.right_wrapper, orient="vertical", command=self.right_canvas.yview)
        self.right_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 6), pady=6)
        self.right_scroll.bind("<Button-1>", lambda _e: self._cancel_scroll_anim(self.right_canvas), add="+")
        self.right_canvas.configure(yscrollcommand=self.right_scroll.set)

        self.right_content = tk.Frame(self.right_canvas, bg="#f1f5f9")
        self.right_window = self.right_canvas.create_window((0, 0), window=self.right_content, anchor="nw")

        self.right_content.bind("<Configure>", lambda _e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas.bind("<Configure>", self._resize_right_canvas)

        self.master.bind_all("<MouseWheel>", self._on_mousewheel)
        self.master.bind_all("<Button-4>", self._on_mousewheel)
        self.master.bind_all("<Button-5>", self._on_mousewheel)

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
                ("dt", "Time Step (dt)"),
            ],
        )
        self._create_control_group(
            title="Control Gains",
            fields=[
                ("kd", "Gap Gain (kd)"),
                ("kv", "Rel Vel Gain (kv)"),
                ("kc", "Des Vel Gain (kc)"),
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
        self._create_control_group(
            title="Sensor Perception",
            fields=[
                ("sensor_delay", "Perception Delay (s)"),
                ("sensor_noise", "Sensor Noise (m)"),
            ],
        )

        self.action_card = tk.Frame(self.controls_container, bg="#ffffff")
        self.action_card.pack(fill="x", pady=(0, 16))

        ttk.Label(self.action_card, text="Simulation Actions", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        self.run_button = ttk.Button(self.action_card, text="Run Simulation", command=self.run_simulation, style="Primary.TButton")
        self.run_button.pack(fill="x", pady=(0, 8))

        controls_row_1 = tk.Frame(self.action_card, bg="#ffffff")
        controls_row_1.pack(fill="x", pady=(0, 8))
        controls_row_1.columnconfigure((0, 1), weight=1)
        self.stop_lead_button = ttk.Button(controls_row_1, text="Stop Lead", command=self.stop_lead, style="TButton")
        self.stop_lead_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.resume_lead_button = ttk.Button(controls_row_1, text="Resume Lead", command=self.resume_lead, style="TButton")
        self.resume_lead_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        controls_row_2 = tk.Frame(self.action_card, bg="#ffffff")
        controls_row_2.pack(fill="x", pady=(0, 8))
        controls_row_2.columnconfigure((0, 1), weight=1)
        self.stop_following_button = ttk.Button(controls_row_2, text="Stop Following", command=self.stop_follower, style="TButton")
        self.stop_following_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.resume_following_button = ttk.Button(controls_row_2, text="Resume Following", command=self.resume_follower, style="TButton")
        self.resume_following_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.plot_acc_button = ttk.Button(self.action_card, text="Plot Vel and Acc Profiles", command=self.plot_vel_acc_profiles, style="TButton")
        self.plot_acc_button.pack(fill="x", pady=(0, 8))

        self.record_button = ttk.Button(
            self.action_card,
            text="⏺ Record Video",
            command=self.toggle_recording,
            style="Record.TButton",
        )
        self.record_button.pack(fill="x", pady=(0, 8))

        self.record_status_label = ttk.Label(
            self.action_card,
            text="Ready to record",
            style="Meta.TLabel",
        )
        self.record_status_label.pack(anchor="w", pady=(0, 8))

        self.use_velocity_profile = tk.BooleanVar(value=False)
        self.velocity_profile_checkbox = ttk.Checkbutton(
            self.action_card,
            text="Enable Ego Velocity Profile",
            variable=self.use_velocity_profile,
        )
        self.velocity_profile_checkbox.pack(anchor="w")

    def _create_control_group(self, title, fields):
        # Section inside the single unified side panel
        section = tk.Frame(self.controls_container, bg="#ffffff")
        section.pack(fill="x", pady=(0, 16))
        self.control_cards.append(section)

        ttk.Label(section, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(section, bg="#ffffff")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for idx, (key, label) in enumerate(fields):
            row = idx // 2
            col = idx % 2
            self._create_field(grid, key, label, row, col)

    def _create_field(self, parent, key, label, row, col):
        # Field component with title docked at the top of the input field itself
        field_comp = AdaptiveRoundedField(
            parent,
            label_text=label,
            default_val=str(self.default_values[key])
        )
        field_comp.grid(row=row, column=col, sticky="ew", padx=(0, 4) if col == 0 else (4, 0), pady=4)
        self.entries[key] = field_comp.entry

    def _build_visualization_area(self):
        self.right_content.columnconfigure(0, weight=1)

        # Top metrics strip removed per user request. Null references maintained for compatibility.
        self.metric_value_total = None
        self.metric_value_vel = None
        self.metric_value_stability = None

        self.visualizations_wrap = ttk.Frame(self.right_content, style="Pane.TFrame")
        self.visualizations_wrap.grid(row=0, column=0, sticky="nsew")

        # Simulation panels
        self.painter_acc, self.energy_label_acc, self.meta_acc = self._build_visualization_card("ACC", self.city_acc, height=200)
        self.painter_bcc, self.energy_label_bcc, self.meta_bcc = self._build_visualization_card("BCC", self.city_bcc, height=155)
        self.painter_accbcc, self.energy_label_accbcc, self.meta_accbcc = self._build_visualization_card("Combined", self.city_accbcc, height=155)

        self.painters = [self.painter_acc, self.painter_bcc, self.painter_accbcc]

    def _build_visualization_card(self, title, city_obj, parent=None, height=150):
        card_parent = parent if parent is not None else self.visualizations_wrap
        card = tk.Frame(card_parent, bg="#ffffff", padx=12, pady=10)
        card.pack(fill="x", pady=(0, 10))

        header_frame = tk.Frame(card, bg="#ffffff")
        header_frame.pack(fill="x", pady=(0, 6))

        title_label = tk.Label(header_frame, text=title, fg="#0f172a", bg="#ffffff", font=("Segoe UI", 11, "bold"))
        title_label.pack(side="left")

        # Replaced car count with Avg Energy and Avg Velocity
        metadata_label = tk.Label(
            header_frame,
            text="Avg Energy: 0.0000 kWh  |  Avg Velocity: 0.00 m/s",
            fg="#64748b",
            bg="#ffffff",
            font=("Consolas", 9, "bold")
        )
        metadata_label.pack(side="right")

        painter = TransportationPainter(
            card,
            city_obj.roads,
            city_obj.cars,
            width=950,
            height=height,
            bg="#f8fafc",
            highlightthickness=0,
        )
        painter.pack(fill="x", expand=True)

        energy_label = tk.Label(card, text="Energy 0.0000 kWh", fg="#64748b", bg="#ffffff")

        return painter, energy_label, metadata_label

    def _resize_left_canvas(self, event):
        self.left_canvas.itemconfigure(self.controls_window, width=event.width)

    def _resize_right_canvas(self, event):
        self.right_canvas.itemconfigure(self.right_window, width=event.width)

    def _get_widget_at_pointer(self, x_root, y_root):
        """Safely finds the widget under the pointer without throwing on internal Tk popdown/menu widgets."""
        try:
            raw_name = self.master.tk.call("winfo", "containing", x_root, y_root)
            if not raw_name:
                return None
            # Ttk combobox popup listbox creates an unmapped '.popdown' shell in Tk
            if "popdown" in str(raw_name):
                return "popdown"
            return self.master._nametowidget(raw_name)
        except Exception:
            return None

    def _is_descendant_of(self, widget, ancestor):
        try:
            while widget is not None:
                if widget == ancestor:
                    return True
                widget = getattr(widget, "master", None)
            return False
        except Exception:
            return False

    def _cancel_scroll_anim(self, canvas):
        """Cancels active smooth scrolling on a canvas when user drags the scrollbar."""
        if hasattr(self, "_scroll_animating"):
            self._scroll_animating.discard(canvas)
        if hasattr(self, "_scroll_targets"):
            self._scroll_targets.pop(canvas, None)

    def _smooth_scroll(self, canvas, delta_pixels):
        """Initiates or updates smooth interpolated ease-out scrolling on a canvas."""
        try:
            bbox = canvas.bbox("all")
            if not bbox:
                return
            total_h = bbox[3] - bbox[1]
            vis_h = canvas.winfo_height()
            max_scroll = max(0, total_h - vis_h)
            if max_scroll <= 0:
                return

            cur_top = canvas.yview()[0] * total_h
            if canvas not in self._scroll_animating:
                self._scroll_targets[canvas] = cur_top

            self._scroll_targets[canvas] = max(
                0.0, min(float(max_scroll), self._scroll_targets[canvas] + delta_pixels)
            )

            if canvas not in self._scroll_animating:
                self._scroll_animating.add(canvas)
                self._animate_smooth_scroll(canvas, total_h, max_scroll)
        except Exception:
            pass

    def _animate_smooth_scroll(self, canvas, total_h, max_scroll):
        """High-frequency easing loop providing smooth 60-80Hz kinetic scrolling."""
        try:
            if canvas not in self._scroll_animating or canvas not in self._scroll_targets:
                self._scroll_animating.discard(canvas)
                return

            cur_top = canvas.yview()[0] * total_h
            target = self._scroll_targets[canvas]
            diff = target - cur_top

            # Settle immediately when close enough
            if abs(diff) < 0.8:
                canvas.yview_moveto(target / total_h)
                self._scroll_animating.discard(canvas)
                self._scroll_targets.pop(canvas, None)
                return

            # Smooth ease-out lerp step
            step = diff * 0.28
            new_top = cur_top + step
            canvas.yview_moveto(new_top / total_h)
            self.master.after(12, lambda: self._animate_smooth_scroll(canvas, total_h, max_scroll))
        except Exception:
            if hasattr(self, "_scroll_animating"):
                self._scroll_animating.discard(canvas)
            if hasattr(self, "_scroll_targets"):
                self._scroll_targets.pop(canvas, None)

    def _is_in_bbox(self, widget, x_root, y_root):
        """Checks if a screen coordinate falls within the physical bounding box of a widget."""
        try:
            if not widget.winfo_ismapped():
                return False
            wx = widget.winfo_rootx()
            wy = widget.winfo_rooty()
            ww = widget.winfo_width()
            wh = widget.winfo_height()
            return wx <= x_root <= wx + ww and wy <= y_root <= wy + wh
        except Exception:
            return False

    def _on_mousewheel(self, event):
        try:
            hovered_widget = self._get_widget_at_pointer(event.x_root, event.y_root)
            if hovered_widget == "popdown":
                # Let the native combobox popdown list handle its own scrolling
                return

            delta_pixels = 0.0
            if getattr(event, "delta", 0):
                # Standard mouse wheel notch (120 units) -> smooth ~72px glide
                delta_pixels = - (event.delta / 120.0) * 72.0
            elif getattr(event, "num", 0) == 4:
                delta_pixels = -72.0
            elif getattr(event, "num", 0) == 5:
                delta_pixels = 72.0

            if abs(delta_pixels) < 0.01:
                return

            # Determine target canvas by widget hierarchy or physical geometry
            target_canvas = None
            intuit_canvas = getattr(self, "intuitiveness_canvas", None)
            if intuit_canvas and (
                self._is_descendant_of(hovered_widget, intuit_canvas)
                or self._is_in_bbox(intuit_canvas, event.x_root, event.y_root)
            ):
                target_canvas = intuit_canvas
            elif self._is_descendant_of(hovered_widget, self.left_canvas) or self._is_in_bbox(self.left_canvas, event.x_root, event.y_root):
                target_canvas = self.left_canvas
            elif self._is_descendant_of(hovered_widget, self.right_canvas) or self._is_in_bbox(self.right_canvas, event.x_root, event.y_root):
                target_canvas = self.right_canvas

            if target_canvas is not None:
                self._smooth_scroll(target_canvas, delta_pixels)
        except Exception:
            pass

    def _handle_resize(self, _event=None):
        width = self.master.winfo_width()

        if width < 1100:
            self.body.columnconfigure(0, weight=1)
            self.body.columnconfigure(1, weight=1)
            self.left_wrapper.grid_configure(row=0, column=0, padx=(0, 0), pady=(0, 12))
            self.right_wrapper.grid_configure(row=1, column=0)
            target_width = max(640, width - 80)
        else:
            self.body.columnconfigure(0, weight=22)
            self.body.columnconfigure(1, weight=78)
            self.left_wrapper.grid_configure(row=0, column=0, padx=(0, 14), pady=(0, 0))
            self.right_wrapper.grid_configure(row=0, column=1)
            target_width = max(760, int(width * 0.69))

        full_width = target_width

        if hasattr(self, "painters"):
            for p in self.painters:
                if p:
                    p.config(width=full_width)
                    p.repaint()
        else:
            for attr in ("painter_acc", "painter_bcc", "painter_accbcc", "painter_rl"):
                if hasattr(self, attr):
                    p = getattr(self, attr)
                    p.config(width=full_width)
                    p.repaint()

    def load_velocity_profile(self, filename="data/data.csv"):
        self.ego_velocity_profile = []
        self.ego_velocity_profile_1 = []
        with open("data/data.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row['time'])
                velocity = float(row['velocity'])
                self.ego_velocity_profile.append((time, velocity))

        with open("data/data2.csv", 'r') as f:
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

        self.dt = args[-1]

        sensor_delay = float(self.entries['sensor_delay'].get()) if 'sensor_delay' in self.entries else 0.0
        sensor_noise = float(self.entries['sensor_noise'].get()) if 'sensor_noise' in self.entries else 0.0

        self.city_acc = City()
        self.city_bcc = City()
        self.city_accbcc = City()

        # Initialize cities with parameters for ACC and BCC models, including dt
        self.city_acc.init(*args[:-1], dt=self.dt, model='ACC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        self.city_bcc.init(*args[:-1], dt=self.dt, model='BCC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        self.city_accbcc.init(*args[:-1], dt=self.dt, model='ACC+BCC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)

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
        if self.timer:
            self.master.after_cancel(self.timer)
        self.update_simulation()
   
    def plot_vel_acc_profiles(self):
        dt = self.dt

        fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex='col')

        models = [
            ("ACC", self.city_acc, 0),
            ("BCC", self.city_bcc, 1),
            ("ACC+BCC Integration", self.city_accbcc, 2)
        ]

        min_dis = getattr(self, 'min_dis', 6.0)

        for name, city, row in models:
            # Velocity (Col 0)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.vel_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "blue"
                axes[row, 0].plot(time_axis, car.vel_history, color=color, linewidth=0.6)
            axes[row, 0].set_title(f"{name} Velocity")
            axes[row, 0].set_ylabel("Velocity (m/s)")
            axes[row, 0].grid(True)
            if row == 2: axes[row, 0].set_xlabel("Time (s)")

            # Acceleration (Col 1)
            for idx, car in enumerate(city.cars):
                time_axis = [dt * i for i in range(len(car.acc_history))]
                color = "red" if idx == 0 else "green" if idx == 2 else "blue"
                axes[row, 1].plot(time_axis, car.acc_history, color=color, linewidth=0.6)
            axes[row, 1].set_title(f"{name} Acceleration")
            axes[row, 1].set_ylabel("Accel (m/s^2)")
            axes[row, 1].grid(True)
            if row == 2: axes[row, 1].set_xlabel("Time (s)")

            # Inter-Vehicle Gap (Col 2)
            for idx, car in enumerate(city.cars):
                if idx == 0:
                    continue
                time_axis = [dt * i for i in range(len(car.gap_history))]
                color = "green" if idx == 2 else "blue"
                axes[row, 2].plot(time_axis, car.gap_history, color=color, linewidth=0.6)
            axes[row, 2].axhline(y=min_dis, color="orange", linestyle="--", linewidth=1.0, alpha=0.8, label=f"min_dis ({min_dis}m)")
            axes[row, 2].set_title(f"{name} Inter-Vehicle Gap")
            axes[row, 2].set_ylabel("Gap (m)")
            axes[row, 2].grid(True)
            if row == 2: axes[row, 2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    def update_simulation(self):
        dt = self.dt 

        self.city_acc.set_leader_stop(self.leader_stop)
        self.city_acc.set_follower_stop(self.follower_stop)
        self.city_bcc.set_leader_stop(self.leader_stop)
        self.city_bcc.set_follower_stop(self.follower_stop)
        self.city_accbcc.set_leader_stop(self.leader_stop)
        self.city_accbcc.set_follower_stop(self.follower_stop)

        self.city_acc.run(dt)
        self.city_bcc.run(dt)
        self.city_accbcc.run(dt)

        self.painter_acc.set_elements(self.city_acc.roads, self.city_acc.cars)
        self.painter_bcc.set_elements(self.city_bcc.roads, self.city_bcc.cars)
        self.painter_accbcc.set_elements(self.city_accbcc.roads, self.city_accbcc.cars)

        self.painter_acc.repaint()
        self.painter_bcc.repaint()
        self.painter_accbcc.repaint()

        # Calculate energy and average velocity
        total_energy_acc = sum(car.energy_used for car in self.city_acc.cars)
        total_energy_bcc = sum(car.energy_used for car in self.city_bcc.cars)
        total_energy_accbcc = sum(car.energy_used for car in self.city_accbcc.cars)

        avg_energy_acc = total_energy_acc / len(self.city_acc.cars) if self.city_acc.cars else 0.0
        avg_energy_bcc = total_energy_bcc / len(self.city_bcc.cars) if self.city_bcc.cars else 0.0
        avg_energy_accbcc = total_energy_accbcc / len(self.city_accbcc.cars) if self.city_accbcc.cars else 0.0

        avg_v_acc = sum(car.velocity for car in self.city_acc.cars) / len(self.city_acc.cars) if self.city_acc.cars else 0.0
        avg_v_bcc = sum(car.velocity for car in self.city_bcc.cars) / len(self.city_bcc.cars) if self.city_bcc.cars else 0.0
        avg_v_accbcc = sum(car.velocity for car in self.city_accbcc.cars) / len(self.city_accbcc.cars) if self.city_accbcc.cars else 0.0

        if hasattr(self, 'energy_label_acc') and self.energy_label_acc:
            self.energy_label_acc.config(text=f"Energy {total_energy_acc:.4f} kWh")
        if hasattr(self, 'energy_label_bcc') and self.energy_label_bcc:
            self.energy_label_bcc.config(text=f"Energy {total_energy_bcc:.4f} kWh")
        if hasattr(self, 'energy_label_accbcc') and self.energy_label_accbcc:
            self.energy_label_accbcc.config(text=f"Energy {total_energy_accbcc:.4f} kWh")

        # Per-road metadata label shows Avg Energy and Avg Velocity (no car count)
        self.meta_acc.config(text=f"Avg Energy: {avg_energy_acc:.4f} kWh  |  Avg Velocity: {avg_v_acc:.2f} m/s")
        self.meta_bcc.config(text=f"Avg Energy: {avg_energy_bcc:.4f} kWh  |  Avg Velocity: {avg_v_bcc:.2f} m/s")
        self.meta_accbcc.config(text=f"Avg Energy: {avg_energy_accbcc:.4f} kWh  |  Avg Velocity: {avg_v_accbcc:.2f} m/s")

        # Global metrics update if widgets are present
        if hasattr(self, 'metric_value_total') and self.metric_value_total is not None:
            total_energy_all = total_energy_acc + total_energy_bcc + total_energy_accbcc
            all_cars = self.city_acc.cars + self.city_bcc.cars + self.city_accbcc.cars
            avg_velocity_all = sum(car.velocity for car in all_cars) / len(all_cars) if all_cars else 0.0
            avg_abs_acc = sum(abs(car.acceleration) for car in all_cars) / len(all_cars) if all_cars else 0.0
            stability_score = max(0.0, min(100.0, 100.0 - (avg_abs_acc * 12.0)))
            self.metric_value_total.config(text=f"{total_energy_all:08.4f} kWh")
            if hasattr(self, 'metric_value_vel') and self.metric_value_vel is not None:
                self.metric_value_vel.config(text=f"{avg_velocity_all:05.2f} m/s")
            if hasattr(self, 'metric_value_stability') and self.metric_value_stability is not None:
                self.metric_value_stability.config(text=f"{stability_score:05.1f}")

        # Capture video frame if recording is active
        if hasattr(self, "recorder") and self.recorder.is_recording:
            target = getattr(self, "root", self.master)
            self.recorder.capture_frame(target)
            if self.recorder.frame_count % 10 == 0:
                self.record_status_label.config(text=f"🔴 Rec: {self.recorder.frame_count} frames")

        self.timer = self.master.after(int(dt * 1000), self.update_simulation)

    def toggle_recording(self):
        if not self.recorder.is_recording:
            if not hasattr(self, "city_acc") or self.city_acc is None:
                self.run_simulation()

            fps = max(10, min(60, int(round(1.0 / self.dt)))) if hasattr(self, "dt") and self.dt > 0 else 20
            target = getattr(self, "root", self.master)
            path = self.recorder.start_recording(target, fps=fps)
            if path:
                self.record_button.config(text="⏹ Stop Recording", style="RecordActive.TButton")
                self.record_status_label.config(text=f"🔴 Recording: {os.path.basename(path)}")
        else:
            path, count = self.recorder.stop_recording()
            self.record_button.config(text="⏺ Record Video", style="Record.TButton")
            if path:
                self.record_status_label.config(text=f"Saved {count} frames ({os.path.basename(path)})")

    def stop_lead(self):
        print("Stopping lead for ACC, BCC, and ACC+BCC")
        self.leader_stop = True
    
    def resume_lead(self):
        self.leader_stop = False

    def stop_follower(self):
        self.follower_stop = True

    def resume_follower(self):
        self.follower_stop = False

    def _on_closing(self):
        if hasattr(self, "_scroll_animating"):
            self._scroll_animating.clear()
        if hasattr(self, "_scroll_targets"):
            self._scroll_targets.clear()
        if hasattr(self, "recorder") and self.recorder.is_recording:
            try:
                self.recorder.stop_recording()
            except Exception:
                pass
        if self.timer:
            try:
                self.master.after_cancel(self.timer)
            except Exception:
                pass
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Traffic Simulation Control Window")
    app = ControlWindow(root)
    root.mainloop()
