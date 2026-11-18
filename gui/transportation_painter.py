"""
transportation_painter.py: Visualization for the simulation (using tkinter).
Light theme road canvas with color-encoded vehicles and interactive hover tooltips.
"""

import tkinter as tk

class TransportationPainter(tk.Canvas):
    # Initialize canvas with roads and cars
    def __init__(self, master, roads, cars, *args, **kwargs):
        if "bg" not in kwargs:
            kwargs["bg"] = "#f8fafc"
        if "highlightthickness" not in kwargs:
            kwargs["highlightthickness"] = 0
        super().__init__(master, *args, **kwargs)
        self.roads = roads  # Store roads for rendering
        self.cars = cars  # Store cars for rendering
        self.counter = 0  # Track simulation steps
        self.hovered_car = None
        self.hovered_car_idx = None
        self.car_hitboxes = []

        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Leave>", self._on_mouse_leave)

    # Update roads and cars references
    def set_elements(self, roads, cars):
        self.roads = roads
        self.cars = cars
        if self.hovered_car and self.hovered_car not in self.cars:
            self.hovered_car = None
            self.hovered_car_idx = None

    # Increment simulation step counter
    def increase_counter(self):
        self.counter += 1

    # Initialize/reset visualization
    def init(self):
        self.counter = 0
        self.hovered_car = None
        self.hovered_car_idx = None
        self.repaint()

    # Clear and redraw canvas
    def repaint(self):
        self.delete('all')  # Clear all items
        self.paint()  # Redraw elements

    def _on_mouse_move(self, event):
        hit = None
        hit_idx = None
        for (x1, y1, x2, y2, car, idx) in self.car_hitboxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                hit = car
                hit_idx = idx
                break

        if hit != self.hovered_car:
            self.hovered_car = hit
            self.hovered_car_idx = hit_idx
            self.delete("tooltip")
            if hit is not None:
                self._draw_tooltip_for_car(hit, hit_idx)
        elif hit is not None:
            self.delete("tooltip")
            self._draw_tooltip_for_car(hit, hit_idx)

    def _on_mouse_leave(self, _event=None):
        self.hovered_car = None
        self.hovered_car_idx = None
        self.delete("tooltip")

    def _draw_tooltip_for_car(self, car, idx):
        if not self.roads:
            return
        canvas_width = int(self.winfo_width() or self["width"])
        canvas_height = int(self.winfo_height() or self["height"])
        margin = max(30, int(canvas_width * 0.04))
        road_left = margin
        road_right = canvas_width - margin
        road_y = max(60, canvas_height // 2)
        road_length = self.roads[0].length if self.roads else 1000.0

        x = road_right - (car.pos / road_length) * (road_right - road_left)
        y = road_y

        is_human = getattr(car, 'is_human', False) or getattr(car, 'is_reckless', False)
        is_reckless = getattr(car, 'is_reckless', False)
        is_rl = getattr(car, 'is_rl', False)
        p_name = getattr(car, 'human_profile_name', None)
        v_curr = getattr(car, 'velocity', 0.0)
        acc = getattr(car, 'acceleration', 0.0)
        energy = getattr(car, 'energy_used', 0.0)

        # Determine readable role title
        if idx == 0:
            role_text = "Lead Vehicle (Autonomous)"
        elif is_rl or (idx == 1 and not is_human):
            role_text = f"Car #{idx + 1} (RL TD3 Ego Agent)"
        elif is_human:
            role_text = f"Car #{idx + 1} (Human Driver - {p_name})" if p_name else f"Car #{idx + 1} (Human Driver)"
        else:
            mode = getattr(car, 'mode', 'ACC')
            role_text = f"Car #{idx + 1} (Platoon - {mode})"

        # Determine maneuver / state badge
        if acc < -2.0:
            status_text = "⚠️ HARD BRAKING"
            status_color = "#f87171"
        elif acc < -0.8:
            status_text = "Braking"
            status_color = "#fbbf24"
        elif is_reckless and (acc > 2.5 or v_curr > 22.0):
            status_text = "⚡ SURGE"
            status_color = "#f87171"
        elif acc > 0.8:
            status_text = "Accelerating"
            status_color = "#60a5fa"
        else:
            status_text = "Cruising"
            status_color = "#4ade80"

        gap_str = ""
        if hasattr(car, 'gap_history') and car.gap_history:
            gap_str = f"Gap: {car.gap_history[-1]:.2f} m  |  "

        lines = [
            (role_text, "#ffffff", ("Segoe UI", 8, "bold")),
            (f"Status: {status_text}", status_color, ("Segoe UI", 8, "bold")),
            (f"Speed: {v_curr:.2f} m/s ({v_curr * 3.6:.1f} km/h)", "#e2e8f0", ("Consolas", 8)),
            (f"{gap_str}Accel: {acc:+.2f} m/s²", "#cbd5e1", ("Consolas", 8)),
            (f"Energy Used: {energy:.4f} kWh", "#94a3b8", ("Consolas", 8)),
        ]

        tip_w = 236
        tip_h = len(lines) * 16 + 14

        # Position above or below car
        tip_y2 = y - 14
        tip_y1 = tip_y2 - tip_h
        if tip_y1 < 6:
            tip_y1 = y + 16
            tip_y2 = tip_y1 + tip_h

        tip_x1 = max(8, min(canvas_width - tip_w - 8, x - tip_w / 2))
        tip_x2 = tip_x1 + tip_w

        # Draw floating popover card
        self.create_rectangle(
            tip_x1, tip_y1, tip_x2, tip_y2,
            fill="#0f172a", outline="#334155", width=1,
            tags="tooltip"
        )

        curr_y = tip_y1 + 10
        for text, color, font in lines:
            self.create_text(
                tip_x1 + 10, curr_y,
                text=text, anchor="w",
                font=font, fill=color,
                tags="tooltip"
            )
            curr_y += 16

    # Render roads and cars on canvas
    def paint(self):
        canvas_width = int(self.winfo_width() or self["width"])
        canvas_height = int(self.winfo_height() or self["height"])
        margin = max(30, int(canvas_width * 0.04))
        road_left = margin
        road_right = canvas_width - margin
        road_y = max(60, canvas_height // 2)

        # Road body and shoulders (Asphalt Slate in light theme)
        self.create_rectangle(road_left, road_y - 15, road_right, road_y + 15, fill="#334155", outline="")
        self.create_line(road_left, road_y - 15, road_right, road_y - 15, width=1, fill="#475569")
        self.create_line(road_left, road_y + 15, road_right, road_y + 15, width=1, fill="#475569")

        # Dashed lane markings
        dash_count = 16
        segment = max(20, (road_right - road_left) // (dash_count * 2))
        gap = segment
        for i in range(dash_count):
            x1 = road_left + i * (segment + gap)
            x2 = min(x1 + segment, road_right)
            self.create_line(x1, road_y, x2, road_y, width=2, fill="#e2e8f0")

        self.car_hitboxes = []

        # Draw cars sorted by position
        if self.cars:
            sorted_cars = sorted(self.cars, key=lambda c: c.pos)
            l = len(sorted_cars)
            for i in range(l - 1, -1, -1):
                car = sorted_cars[i]
                road_length = self.roads[0].length if self.roads else 1000.0
                # Map car position to canvas x-coordinate (inverted)
                x = road_right - (car.pos / road_length) * (road_right - road_left)
                y = road_y

                # Determine human / reckless driver state and visual styling
                is_human = getattr(car, 'is_human', False) or getattr(car, 'is_reckless', False)

                # Color encoding
                if is_human:
                    p_name = getattr(car, 'human_profile_name', '')
                    if p_name == 'CAUTIOUS':
                        fill_color = '#eab308'
                        outline_color = "#fef08a"
                    elif p_name == 'NORMAL':
                        fill_color = '#94a3b8'
                        outline_color = "#e2e8f0"
                    elif getattr(car, 'color', '') == 'orange':
                        fill_color = 'orange'
                        outline_color = "#fed7aa"
                    else:
                        fill_color = '#ef4444'
                        outline_color = "#fee2e2"
                    outline_width = 1.5
                else:
                    fill_color = getattr(car, 'color', '#3b82f6')
                    outline_color = "#0f172a"
                    outline_width = 1

                car_len = getattr(car, 'length', 5.0)
                half_len = max(5.0, car_len / 2)

                # Draw vehicle body (pure color encoding, no cluttered text)
                self.create_rectangle(
                    x - half_len,
                    y - 7,
                    x + half_len,
                    y + 7,
                    fill=fill_color,
                    outline=outline_color,
                    width=outline_width
                )

                # Register hitbox for hover tooltip detection
                orig_idx = self.cars.index(car) if car in self.cars else i
                self.car_hitboxes.append((x - half_len - 3, y - 10, x + half_len + 3, y + 10, car, orig_idx))

        # If a car is currently hovered, re-render its tooltip live
        if self.hovered_car and self.hovered_car in self.cars:
            self._draw_tooltip_for_car(self.hovered_car, self.hovered_car_idx)
