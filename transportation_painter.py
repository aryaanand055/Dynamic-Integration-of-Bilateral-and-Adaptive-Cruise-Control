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
                # Draw car rectangle
                self.create_rectangle(
                    x-car.length/2,
                    y-9,
                    x+car.length/2,
                    y+9,
                    fill=car.color,
                    outline='',
                    width=0
                )
                # Determine mode label (A=ACC, B=BCC, S=SWITCH)
                mode_label = 'A' if getattr(car, 'mode', 'ACC') == 'ACC' else 'B' if getattr(car, 'mode', 'BCC') == 'BCC' else f"{car.integration_factor:.2f}".split(".")[1] if getattr(car, 'mode', 'INTEGRATED') == 'INTEGRATED' else "S" if getattr(car, 'mode', "SWITCH") == "SWITCH" else 'V'
                # Minimal mode badge just below each vehicle.
                self.create_text(x, y + 20, text=mode_label, font=("Consolas", 7), fill="#94a3b8")
   
