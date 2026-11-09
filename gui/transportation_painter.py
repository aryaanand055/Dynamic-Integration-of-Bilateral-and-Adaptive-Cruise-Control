"""
transportation_painter.py: Visualization for the simulation (using tkinter).
"""

import tkinter as tk

class TransportationPainter(tk.Canvas):
    # Initialize canvas with roads and cars
    def __init__(self, master, roads, cars, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.roads = roads  # Store roads for rendering
        self.cars = cars  # Store cars for rendering
        self.counter = 0  # Track simulation steps

    # Update roads and cars references
    def set_elements(self, roads, cars):
        self.roads = roads
        self.cars = cars

    # Increment simulation step counter
    def increase_counter(self):
        self.counter += 1

    # Initialize/reset visualization
    def init(self):
        self.counter = 0
        self.repaint()

    # Clear and redraw canvas
    def repaint(self):
        self.delete('all')  # Clear all items
        self.paint()  # Redraw elements


    # Render roads and cars on canvas
    def paint(self):
        # Draw road with subtle lane treatment for a cleaner dashboard look.
        canvas_width = int(self["width"])
        canvas_height = int(self["height"])
        margin = max(30, int(canvas_width * 0.04))
        road_left = margin
        road_right = canvas_width - margin
        road_y = max(60, canvas_height // 2)

        # Road body and shoulders.
        self.create_rectangle(road_left, road_y - 18, road_right, road_y + 18, fill="#1e293b", outline="")
        self.create_line(road_left, road_y - 18, road_right, road_y - 18, width=1, fill="#334155")
        self.create_line(road_left, road_y + 18, road_right, road_y + 18, width=1, fill="#334155")

        dash_count = 14
        segment = max(20, (road_right - road_left) // (dash_count * 2))
        gap = segment
        for i in range(dash_count):
            x1 = road_left + i * (segment + gap)
            x2 = min(x1 + segment, road_right)
            self.create_line(x1, road_y, x2, road_y, width=2, fill="#64748b")
            
        # Draw cars sorted by position
        if self.cars:
            sorted_cars = sorted(self.cars, key=lambda c: c.pos)
            l = len(sorted_cars)
            for i in range(l-1, -1, -1):
                car = sorted_cars[i]
                road_length = self.roads[0].length if self.roads else 1000
                # Map car position to canvas x-coordinate (inverted)
                x = road_right - (car.pos / road_length) * (road_right - road_left)
                y = road_y
                # Determine human / reckless driver state and visual styling
                is_human = getattr(car, 'is_human', False) or getattr(car, 'is_reckless', False)
                is_reckless = getattr(car, 'is_reckless', False)
                
                # Human car styling (supports specific profile colors like Cautious Amber or Aggressive Red)
                if is_human:
                    p_name = getattr(car, 'human_profile_name', '')
                    if p_name == 'CAUTIOUS':
                        fill_color = '#eab308'
                        outline_color = "#fef08a"
                    elif p_name == 'NORMAL':
                        fill_color = '#94a3b8'
                        outline_color = "#e2e8f0"
                    elif car.color == 'orange':
                        fill_color = 'orange'
                        outline_color = "#fed7aa"
                    else:
                        fill_color = '#ef4444'
                        outline_color = "#fee2e2"
                    outline_width = 2
                else:
                    fill_color = car.color
                    outline_color = ""
                    outline_width = 0

                # Draw car rectangle
                self.create_rectangle(
                    x - car.length / 2,
                    y - 9,
                    x + car.length / 2,
                    y + 9,
                    fill=fill_color,
                    outline=outline_color,
                    width=outline_width
                )

                # Draw prominent top alert tag and bottom mode badge
                v_curr = getattr(car, 'velocity', 0.0)
                if is_human:
                    acc = getattr(car, 'acceleration', 0.0)
                    p_name = getattr(car, 'human_profile_name', None)
                    p_prefix = f"👤 {p_name}" if p_name else "👤 HUMAN"
                    if acc < -2.0:
                        tag_text = f"{p_prefix} (BRAKING) {v_curr:.1f} m/s"
                        tag_color = "#f87171"
                    elif is_reckless and (acc > 2.5 or v_curr > 22.0):
                        tag_text = f"{p_prefix} (SURGE) {v_curr:.1f} m/s"
                        tag_color = "#f87171"
                    else:
                        tag_text = f"{p_prefix} {v_curr:.1f} m/s"
                        tag_color = "#fca5a5" if fill_color == '#ef4444' else ("#fde047" if fill_color == '#eab308' else "#cbd5e1")

                    self.create_text(
                        x,
                        y - 18,
                        text=tag_text,
                        font=("Segoe UI", 7, "bold"),
                        fill=tag_color
                    )
                    mode_label = p_name if p_name else "HUMAN"
                    badge_color = fill_color
                    badge_font = ("Consolas", 8, "bold")
                elif getattr(car, 'is_rl', False):
                    mode_label = "RL"
                    badge_color = "#4ade80"
                    badge_font = ("Consolas", 7, "bold")
                elif getattr(car, 'mode', 'ACC') == 'ACC':
                    mode_label = "ACC"
                    badge_color = "#94a3b8"
                    badge_font = ("Consolas", 7)
                elif getattr(car, 'mode', 'BCC') == 'BCC':
                    mode_label = "BCC"
                    badge_color = "#94a3b8"
                    badge_font = ("Consolas", 7)
                elif getattr(car, 'mode', 'INTEGRATED') == 'INTEGRATED':
                    mode_label = f"{car.integration_factor:.2f}".split(".")[1]
                    badge_color = "#94a3b8"
                    badge_font = ("Consolas", 7)
                elif getattr(car, 'mode', "SWITCH") == "SWITCH":
                    mode_label = "SW"
                    badge_color = "#94a3b8"
                    badge_font = ("Consolas", 7)
                else:
                    mode_label = "V"
                    badge_color = "#94a3b8"
                    badge_font = ("Consolas", 7)

                # Mode badge just below each vehicle
                self.create_text(x, y + 20, text=mode_label, font=badge_font, fill=badge_color)

                # Real-time speed readout below mode badge
                v_badge_color = "#fca5a5" if is_human else ("#4ade80" if getattr(car, 'is_rl', False) else "#cbd5e1")
                self.create_text(
                    x,
                    y + 31,
                    text=f"{v_curr:.1f} m/s",
                    font=("Consolas", 7, "bold" if is_human else "normal"),
                    fill=v_badge_color
                )
   
