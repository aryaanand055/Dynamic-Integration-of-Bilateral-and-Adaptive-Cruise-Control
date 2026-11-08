"""
Unit and integration tests for GUI intuitiveness features, resilience safeguards,
and absence of hardcoded metrics.
"""

import os
import sys
import unittest
import tkinter as tk
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.rl_control_window import RLControlWindow, INTUITIVENESS_CRITERIA_CONFIG


class TestGUIIntuitiveness(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Headless mode

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_gui_step_loop_60_simulated_seconds(self):
        """
        Runs the GUI step loop for 60 simulated seconds (600 steps at dt=0.1)
        and verifies that it never stops early. Also verifies resilience against
        injected intuitiveness exceptions.
        """
        app = RLControlWindow(self.root)
        app.run_simulation()
        if app.timer:
            self.root.after_cancel(app.timer)
            app.timer = None

        dt = float(app.dt)
        total_seconds = 60.0
        expected_steps = int(total_seconds / dt)

        # Step until exactly expected_steps (600 steps)
        while app.current_step < expected_steps:
            app.update_simulation()
            if app.timer:
                self.root.after_cancel(app.timer)
                app.timer = None

        self.assertEqual(
            app.current_step,
            expected_steps,
            f"GUI simulation stopped early! Reached step {app.current_step}, expected {expected_steps}."
        )

        # Verify resilience safeguard: simulation must never stop even if intuitiveness raises
        def faulty_intuitiveness():
            raise RuntimeError("Injected Intuitiveness Fault")

        app._update_live_intuitiveness_tiles = faulty_intuitiveness
        resilience_steps = 20
        target_step = expected_steps + resilience_steps
        while app.current_step < target_step:
            app.update_simulation()
            if app.timer:
                self.root.after_cancel(app.timer)
                app.timer = None

        self.assertEqual(
            app.current_step,
            target_step,
            "Simulation stopped unexpectedly after intuitiveness exception was raised!"
        )

    def test_no_hardcoded_metric_strings(self):
        """
        Verifies that no GUI label contains hardcoded metric numbers upon initialization.
        Live tiles must show '--' before the run and while data is insufficient.
        """
        app = RLControlWindow(self.root)

        # 1. Verify all 4 controllers exist in live_tiles with initial '--'
        expected_controllers = ["ACC", "BCC", "ACC+BCC", "RL"]
        for ctrl in expected_controllers:
            self.assertIn(ctrl, app.live_tiles, f"Missing live tiles for controller: {ctrl}")
            tiles = app.live_tiles[ctrl]
            expected_keys = ["jerk", "flips", "wave", "gap", "col"]
            for k in expected_keys:
                self.assertIn(k, tiles, f"Missing live tile '{k}' for controller {ctrl}")
                lbl_text = tiles[k].cget("text")
                self.assertEqual(
                    lbl_text,
                    "--",
                    f"Controller {ctrl} tile '{k}' contains hardcoded value: {lbl_text}"
                )

        # 2. Run simulation: step 1 (current_step == 1, insufficient data < 3), tiles must still show '--'
        app.run_simulation()
        if app.timer:
            self.root.after_cancel(app.timer)
            app.timer = None

        self.assertEqual(app.current_step, 1)
        for ctrl in expected_controllers:
            for k, lbl in app.live_tiles[ctrl].items():
                self.assertEqual(
                    lbl.cget("text"),
                    "--",
                    f"Controller {ctrl} tile '{k}' prematurely populated on step 1: {lbl.cget('text')}"
                )

        # 3. Verify that after sufficient data (step >= 5), tiles update dynamically
        for _ in range(5):
            app.update_simulation()
            if app.timer:
                self.root.after_cancel(app.timer)
                app.timer = None

        for ctrl in expected_controllers:
            tiles = app.live_tiles[ctrl]
            # Tiles should now contain dynamic values with their respective unit strings
            self.assertIn("m/s³", tiles["jerk"].cget("text"))
            self.assertIn("/min", tiles["flips"].cget("text"))
            self.assertIn("x", tiles["wave"].cget("text"))
            self.assertIn("m", tiles["gap"].cget("text"))
            self.assertTrue(tiles["col"].cget("text").isdigit())

    def test_criteria_verdict_computation(self):
        """
        Verifies that verdict is computed dynamically in code from the config dict
        and never hardcoded or typed in as a static string.
        """
        app = RLControlWindow(self.root)
        self.assertIn(app.verdict_label.cget("text"), ["INTUITIVE", "NOT INTUITIVE", "--"])

        # Mock an evaluation payload where zero_collisions fails
        mock_fail_data = {
            "RL": {
                "DII": 90.0,
                "raw": {
                    "collisions": 3,  # Fail
                    "min_gap": 2.5,
                    "rms_jerk": 0.50,
                    "wave_ratio": 1.10,
                }
            },
            "ACC": {"DII": 88.0},
            "IDM_gap_matched": {"DII": 78.0}
        }
        app._compute_and_update_verdict(mock_fail_data)
        self.assertEqual(app.verdict_label.cget("text"), "NOT INTUITIVE")
        self.assertEqual(app.crit_badge_zero_collisions.cget("text"), "FAIL")

        # Mock an evaluation payload where all criteria pass
        mock_pass_data = {
            "RL": {
                "DII": 92.0,
                "raw": {
                    "collisions": 0,
                    "min_gap": 2.2,
                    "rms_jerk": 0.45,
                    "wave_ratio": 1.05,
                }
            },
            "ACC": {"DII": 88.0},
            "IDM_gap_matched": {"DII": 78.0}
        }
        app._compute_and_update_verdict(mock_pass_data)
        self.assertEqual(app.verdict_label.cget("text"), "INTUITIVE")
        self.assertEqual(app.crit_badge_zero_collisions.cget("text"), "PASS")
        self.assertEqual(app.crit_badge_minimum_gap.cget("text"), "PASS")
        self.assertEqual(app.crit_badge_rms_jerk.cget("text"), "PASS")
        self.assertEqual(app.crit_badge_dii_vs_classical.cget("text"), "PASS")
        self.assertEqual(app.crit_badge_wave_ratio.cget("text"), "PASS")


if __name__ == "__main__":
    unittest.main()
