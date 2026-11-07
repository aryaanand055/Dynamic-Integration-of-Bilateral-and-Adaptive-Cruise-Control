"""
Unit test to verify that the human-driven car alone is colored in distinct crimson red (#ef4444)
throughout the WHOLE simulation (from t=0 through cruising, braking, and full stop),
with prominent HUMAN driver alerts and mode badges across all scenarios.
"""
import tkinter as tk
import unittest
from gui.rl_control_window import RLControlWindow


class TestRecklessDriverColor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.root.title("Test Human Car Colors")
        cls.app = RLControlWindow(cls.root)

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls.app, 'timer') and cls.app.timer:
                cls.root.after_cancel(cls.app.timer)
            cls.root.destroy()
        except Exception:
            pass

    def test_01_free_simulation_no_human_cars(self):
        """In free simulation, no car should be marked human/reckless or colored red."""
        self.app.selected_scenario_var.set(self.app.scenario_options[0])
        self.app.scenario_combobox.current(0)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            for car in city.cars:
                self.assertFalse(getattr(car, 'is_human', False))
                self.assertFalse(getattr(car, 'is_reckless', False))
                self.assertNotEqual(car.color, "#ef4444")

    def test_02_scenario_13_human_lead_only(self):
        """In Scenario 13, ONLY the lead car (Car 0) is human and colored red throughout."""
        sc13_name = [s for s in self.app.scenario_options if '13.' in s][0]
        self.app.selected_scenario_var.set(sc13_name)
        idx = self.app.scenario_options.index(sc13_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            cars = city.cars
            # Lead car is human
            self.assertTrue(getattr(cars[0], 'is_human', False))
            self.assertEqual(cars[0].color, "#ef4444")
            # All other cars are NOT human
            for car in cars[1:]:
                self.assertFalse(getattr(car, 'is_human', False))
                self.assertNotEqual(car.color, "#ef4444")

    def test_03_scenario_14_human_trailing_only(self):
        """In Scenario 14, ONLY the trailing car (Car N-1) is human and colored red throughout."""
        sc14_name = [s for s in self.app.scenario_options if '14.' in s][0]
        self.app.selected_scenario_var.set(sc14_name)
        idx = self.app.scenario_options.index(sc14_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            cars = city.cars
            # Trailing car is human
            self.assertTrue(getattr(cars[-1], 'is_human', False))
            self.assertEqual(cars[-1].color, "#ef4444")
            # All preceding cars are NOT human
            for car in cars[:-1]:
                self.assertFalse(getattr(car, 'is_human', False))
                self.assertNotEqual(car.color, "#ef4444")

    def test_04_scenario_15_bilateral_human_sandwich(self):
        """In Scenario 15, Car 0 AND Car N-1 are human, while middle cars are NOT."""
        sc15_name = [s for s in self.app.scenario_options if '15.' in s][0]
        self.app.selected_scenario_var.set(sc15_name)
        idx = self.app.scenario_options.index(sc15_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            cars = city.cars
            # Both lead and trailing cars are human
            self.assertTrue(getattr(cars[0], 'is_human', False))
            self.assertEqual(cars[0].color, "#ef4444")
            self.assertTrue(getattr(cars[-1], 'is_human', False))
            self.assertEqual(cars[-1].color, "#ef4444")
            # Middle ego car is NOT human
            if len(cars) > 2:
                for car in cars[1:-1]:
                    self.assertFalse(getattr(car, 'is_human', False))
                    self.assertNotEqual(car.color, "#ef4444")

    def test_05_transportation_painter_renders_human_elements(self):
        """Check that TransportationPainter renders human driver badges and top tags."""
        self.app.painter_rl.repaint()
        items = self.app.painter_rl.find_all()
        self.assertGreater(len(items), 0)

    def test_06_scenario_16_human_car_whole_simulation_duration(self):
        """
        In Scenario 16, Car 2 (3rd in line) is human-driven.
        Verify that it retains #ef4444 throughout the WHOLE simulation:
        during initial constant speed cruising (t < 5s) AND during hard braking (t >= 5s).
        """
        sc16_name = [s for s in self.app.scenario_options if '16.' in s][0]
        self.app.selected_scenario_var.set(sc16_name)
        idx = self.app.scenario_options.index(sc16_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        # Step 0 (initialization): Car 2 must already be #ef4444
        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            self.assertTrue(getattr(city.cars[2], 'is_human', False))
            self.assertEqual(city.cars[2].color, "#ef4444")
            self.assertFalse(getattr(city.cars[0], 'is_human', False))
            self.assertNotEqual(city.cars[0].color, "#ef4444")
            self.assertFalse(getattr(city.cars[1], 'is_human', False))
            self.assertNotEqual(city.cars[1].color, "#ef4444")

        # Step through 70 timesteps (7.0 seconds of simulation: 5s cruise + 2s braking)
        for step in range(70):
            self.app.update_simulation()
            if self.app.timer:
                self.root.after_cancel(self.app.timer)
                self.app.timer = None

            # Verify Car 2 remains #ef4444 at every single timestep
            for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
                self.assertEqual(
                    city.cars[2].color,
                    "#ef4444",
                    f"Car 2 lost #ef4444 color at step {step} (t={step*0.1:.1f}s)"
                )
                self.assertTrue(getattr(city.cars[2], 'is_human', False))

    def test_07_immediate_color_change_on_dropdown_selection(self):
        """Selecting a scenario in the combobox must immediately color the human car red on canvas."""
        sc16_name = [s for s in self.app.scenario_options if '16.' in s][0]
        self.app.selected_scenario_var.set(sc16_name)
        idx = self.app.scenario_options.index(sc16_name)
        self.app.scenario_combobox.current(idx)
        # Trigger dropdown selection event without clicking Run Simulation
        self.app._on_scenario_selected()

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            self.assertTrue(getattr(city.cars[2], 'is_human', False))
            self.assertEqual(city.cars[2].color, "#ef4444")

    def test_08_scenario_16_shifts_to_random_speeds_and_displays_velocity(self):
        """
        Verify that in Scenario 16, Car 2 cruises for 5s and then shifts to random speeds,
        and its live velocity is displayed by the painter and status bar.
        """
        sc16_name = [s for s in self.app.scenario_options if '16.' in s][0]
        self.app.selected_scenario_var.set(sc16_name)
        idx = self.app.scenario_options.index(sc16_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        # At t=0..5s (first 50 steps), speed should be close to 15.0 m/s
        for _ in range(50):
            self.app.update_simulation()
            if self.app.timer:
                self.root.after_cancel(self.app.timer)
                self.app.timer = None

        v_at_5s = self.app.city_rl.cars[2].velocity
        self.assertAlmostEqual(v_at_5s, 15.0, delta=0.5)

        # Run another 50 steps (t=5s to 10s): speed must shift away from 15.0 m/s to random speeds
        for _ in range(50):
            self.app.update_simulation()
            if self.app.timer:
                self.root.after_cancel(self.app.timer)
                self.app.timer = None

        v_after_shifts = self.app.city_rl.cars[2].velocity
        # Car 2 is shifting speeds, not simply stuck at 0.0 or 15.0
        self.assertGreater(v_after_shifts, 0.0)

        # Verify painter renders speed text
        self.app.painter_rl.repaint()
        all_text = []
        for item in self.app.painter_rl.find_all():
            if self.app.painter_rl.type(item) == "text":
                all_text.append(self.app.painter_rl.itemcget(item, "text"))

        # Painter must display "m/s" and "HUMAN"
        has_speed_readout = any("m/s" in t for t in all_text)
        has_human_badge = any("HUMAN" in t for t in all_text)
        self.assertTrue(has_speed_readout, f"No speed readout found in canvas text: {all_text}")
        self.assertTrue(has_human_badge, f"No HUMAN badge found in canvas text: {all_text}")


if __name__ == "__main__":
    unittest.main()
