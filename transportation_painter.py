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
        # Draw horizontal road lines (positioned nearer the top)
        canvas_width = int(self["width"])
        road_left = 50
        road_right = canvas_width - 50
        road_y = max(60, int(self["height"]) // 3)  # pull road closer to top

        for road in self.roads:
            self.create_line(road_left, road_y, road_right, road_y, width=8, fill='gray')
            
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
                self.create_rectangle(x-car.length/2, y-10, x+car.length/2, y+10, fill=car.color, outline='black')
                # Display velocity above car
                self.create_text(x, y-25, text=f"{car.velocity:.1f}")
                # Determine mode label (A=ACC, B=BCC, S=SWITCH)
                mode_label = 'A' if getattr(car, 'mode', 'ACC') == 'ACC' else 'B' if getattr(car, 'mode', 'BCC') == 'BCC' else f"{car.integration_factor:.2f}".split(".")[1] if getattr(car, 'mode', 'INTEGRATED') == 'INTEGRATED' else "S" if getattr(car, 'mode', "SWITCH") == "SWITCH" else 'V'
                # Display mode label below car
                self.create_text(x, y + 20, text=mode_label, font=("Arial", 8))

   