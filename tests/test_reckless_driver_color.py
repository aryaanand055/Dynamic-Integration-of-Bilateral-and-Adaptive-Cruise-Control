"""
Unit tests to verify vehicle roles, driver models, and color styling across
the 10 operational benchmark scenarios:
- Free simulation: standard colors.
- Scenario 6 & 7: 3-car platoon dynamically initialized.
- Scenario 8: Trailing car styled red (#ef4444).
- Scenario 9: Car 2 (3rd car) emergency brakes to halt and is styled red.
- Scenario 10: 50% mixed heterogeneous traffic with distinct color-coded driver models.
"""
import tkinter as tk
import unittest
from gui.rl_control_window import RLControlWindow


class TestScenarioSuiteAndColors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.root.title("Test Scenario Suite")
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

    def test_02_scenario_count(self):
        """Verify that exactly 11 curated scenarios are available."""
        self.assertEqual(len(self.app.benchmark_scenarios), 11)
        self.assertEqual(len(self.app.scenario_options), 12)  # 0. Free sim + 11 scenarios

    def test_03_scenario_6_three_car_trailing_halt(self):
        """In Scenario 6, exactly 3 cars exist, and Car 2 (trailing) is marked human / red."""
        sc6_name = [s for s in self.app.scenario_options if '6.' in s][0]
        self.app.selected_scenario_var.set(sc6_name)
        idx = self.app.scenario_options.index(sc6_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            self.assertEqual(len(city.cars), 3)
            # Lead and Ego are not human
            self.assertFalse(getattr(city.cars[0], 'is_human', False))
            self.assertFalse(getattr(city.cars[1], 'is_human', False))
            # Trailing car is human and red
            self.assertTrue(getattr(city.cars[2], 'is_human', False))
            self.assertEqual(city.cars[2].color, "#ef4444")

    def test_04_scenario_7_three_car_lead_stop(self):
        """In Scenario 7, exactly 3 cars exist."""
        sc7_name = [s for s in self.app.scenario_options if '7.' in s][0]
        self.app.selected_scenario_var.set(sc7_name)
        idx = self.app.scenario_options.index(sc7_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()

        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            self.assertEqual(len(city.cars), 3)

    def test_05_scenario_9_car3_emergency_brake_persistence(self):
        """
        In Scenario 9, Car 2 (3rd car) is human-driven and halts at t=5s.
        Verify color persistence and velocity drop.
        """
        sc9_name = [s for s in self.app.scenario_options if '9.' in s][0]
        self.app.selected_scenario_var.set(sc9_name)
        idx = self.app.scenario_options.index(sc9_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()
        self.app.run_simulation()
        if self.app.timer:
            self.root.after_cancel(self.app.timer)
            self.app.timer = None

        # Verify 15 cars
        for city in [self.app.city_acc, self.app.city_bcc, self.app.city_accbcc, self.app.city_rl]:
            self.assertEqual(len(city.cars), 15)
            self.assertTrue(getattr(city.cars[2], 'is_human', False))
            self.assertEqual(city.cars[2].color, "#ef4444")

        # Step 60 times (6 seconds, passing t=5s braking initiation)
        for _ in range(60):
            self.app.update_simulation()
            if self.app.timer:
                self.root.after_cancel(self.app.timer)
                self.app.timer = None

        # Car 2 has begun braking
        v_brake = self.app.city_rl.cars[2].velocity
        self.assertLess(v_brake, 15.0)

    def test_06_scenario_10_mixed_heterogeneous_traffic(self):
        """
        In Scenario 10, ~50% of vehicles are human drivers with distinct profiles
        (Aggressive, Cautious, Normal) and appropriate color badges.
        """
        sc10_name = [s for s in self.app.scenario_options if '10.' in s][0]
        self.app.selected_scenario_var.set(sc10_name)
        idx = self.app.scenario_options.index(sc10_name)
        self.app.scenario_combobox.current(idx)
        self.app._on_scenario_selected()

        city = self.app.city_rl
        self.assertEqual(len(city.cars), 15)
        
        # Verify specific assignments
        self.assertEqual(city.cars[2].human_profile_name, 'AGGRESSIVE')
        self.assertEqual(city.cars[2].color, '#ef4444')
        self.assertEqual(city.cars[4].human_profile_name, 'CAUTIOUS')
        self.assertEqual(city.cars[4].color, '#eab308')
        self.assertEqual(city.cars[6].human_profile_name, 'NORMAL')
        self.assertEqual(city.cars[6].color, '#94a3b8')

        # Autonomous followers are not human
        self.assertFalse(getattr(city.cars[1], 'is_human', False))
        self.assertFalse(getattr(city.cars[3], 'is_human', False))
        self.assertFalse(getattr(city.cars[5], 'is_human', False))

    def test_07_painter_elements(self):
        """Verify painter renders properly."""
        self.app.painter_rl.repaint()
        items = self.app.painter_rl.find_all()
        self.assertGreater(len(items), 0)


if __name__ == "__main__":
    unittest.main()
