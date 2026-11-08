"""
Automated Verification Tests for Driver Intuitiveness Metric Suite & IDM Human Reference.

Tests:
1. Constant acceleration trace gives zero jerk and C2 close to 1.
2. Hard braking trace with clear road ahead is counted as phantom braking (C1 < 1).
3. Static platoon at standstill gives C3 = 1.
4. IDMCity: equilibrium gap & speed gives acceleration ~ 0; stopped leader with small gap
   triggers braking; follower never exceeds sim's acceleration and jerk limits.
5. Smoke test: full pipeline runs on a shortened scenario and produces all 4 artifact files.

Runnable via:
- python main.py --test
- python -m unittest tests/test_driver_intuitiveness.py
- pytest tests/test_driver_intuitiveness.py
"""

import os
import sys
import math
import json
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.config import DEFAULT_SIM_PARAMS, resolve_model_path, LOGS_DIR, PLOTS_DIR
from evaluation.driver_intuitiveness import (
    IDMCity,
    IDM_PARAMS,
    get_gap_matched_idm_params,
    compute_dii_metrics,
    run_driver_intuitiveness_suite,
)


class TestDriverIntuitiveness(unittest.TestCase):
    """Unit and integration test cases for Driver Intuitiveness & IDM reference."""

    def test_constant_acceleration_c2(self):
        """A constant-acceleration trace gives zero jerk and C2 close to 1."""
        dur = 30.0
        dt = 0.1
        steps = int(dur / dt)

        # 2 follower cars with constant positive acceleration
        follower_traces = []
        for _ in range(2):
            follower_traces.append({
                'a': [1.2] * steps,
                'v': [15.0 + 1.2 * i * dt for i in range(steps)],
                'gap': [25.0] * steps,
                'vf': [15.0 + 1.2 * i * dt for i in range(steps)],
            })
        leader_trace = {
            'a': [1.2] * steps,
            'v': [15.0 + 1.2 * i * dt for i in range(steps)],
        }

        res = compute_dii_metrics(
            follower_traces,
            leader_trace,
            dur=dur,
            dt=dt,
            min_dis=6.0,
            reaction_time=0.8,
        )

        self.assertAlmostEqual(res['raw']['rms_jerk'], 0.0, places=5,
                               msg="Constant acceleration trace must yield zero RMS jerk")
        self.assertAlmostEqual(res['raw']['sign_changes_per_min'], 0.0, places=5,
                               msg="Constant acceleration trace must yield zero sign flips")
        self.assertGreaterEqual(res['C2'], 0.99,
                                msg="C2 smoothness must be ~1.0 for constant acceleration")

    def test_phantom_braking_c1(self):
        """A trace that brakes hard with a clear road ahead is counted as phantom braking (C1 < 1)."""
        dur = 10.0
        dt = 0.1
        steps = int(dur / dt)

        # follower speed = 15 m/s, desired gap = 6.0 + 0.8 * 15 = 18.0m
        # Clear road gap = 50.0m (> 1.2 * desired_gap = 21.6m)
        # vrel = 0.0 (v = vf = 15.0)
        # Brakes hard at -2.5 m/s^2 (< -1.0 m/s^2)
        follower_traces = [{
            'a': [-2.5] * steps,
            'v': [15.0] * steps,
            'gap': [50.0] * steps,
            'vf': [15.0] * steps,
        }]
        leader_trace = {'a': [0.0] * steps, 'v': [15.0] * steps}

        res = compute_dii_metrics(
            follower_traces,
            leader_trace,
            dur=dur,
            dt=dt,
            min_dis=6.0,
            reaction_time=0.8,
        )

        self.assertGreater(res['raw']['phantom_rate'], 0.0,
                           msg="Hard braking with clear road ahead must be flagged as phantom braking")
        self.assertLess(res['C1'], 1.0,
                        msg="C1 Appropriateness must drop below 1.0 under phantom braking")

    def test_standstill_c3(self):
        """A static platoon at standstill gives C3 = 1."""
        dur = 20.0
        dt = 0.1
        steps = int(dur / dt)

        # All vehicles completely stationary: v=0, vf=0, a=0
        follower_traces = [{
            'a': [0.0] * steps,
            'v': [0.0] * steps,
            'gap': [6.0] * steps,
            'vf': [0.0] * steps,
        } for _ in range(3)]
        leader_trace = {'a': [0.0] * steps, 'v': [0.0] * steps}

        res = compute_dii_metrics(
            follower_traces,
            leader_trace,
            dur=dur,
            dt=dt,
            min_dis=6.0,
            reaction_time=0.8,
        )

        self.assertAlmostEqual(res['C3'], 1.0, places=5,
                               msg="Static platoon at standstill must yield C3 Calmness = 1.0")

    def test_idm_city_equilibrium_and_braking(self):
        """
        IDMCity: at equilibrium gap and speed, IDM acceleration is about 0;
        with a stopped leader and a small gap it brakes;
        the follower never exceeds the sim's accel and jerk limits.
        """
        params = dict(DEFAULT_SIM_PARAMS)
        params['car_number'] = 2
        params['v_des'] = 15.0
        dt = 0.1
        init_args = [
            params[k] for k in [
                'car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v',
                'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap'
            ]
        ]

        # 1. Equilibrium Verification
        # At v = 10 m/s, v0 = 15 m/s, dv = 0:
        # s* = s0 + v * T = 6.0 + 10.0 * 0.8 = 14.0 m
        # s_eq = s* / sqrt(1 - (v/v0)^4) = 14.0 / sqrt(1 - (10/15)^4)
        v_test = 10.0
        v0 = 15.0
        idm_p = get_gap_matched_idm_params(params)
        s_star = idm_p['s0'] + v_test * idm_p['T']
        s_eq = s_star / math.sqrt(1.0 - (v_test / v0) ** 4)

        city = IDMCity(idm_params=idm_p)
        city.init(*init_args, dt=dt, model='ACC')

        leader = city.cars[0]
        follower = city.cars[1]

        leader.velocity = v_test
        follower.velocity = v_test
        follower.acceleration = 0.0

        # Position follower exactly at s_eq behind leader:
        # On this circular road coordinate system, follower trailing leader has:
        # (follower.pos - leader.pos - leader.length) % road_len = s_eq
        road_len = city.roads[0].length
        follower.pos = (leader.pos + leader.length + s_eq) % road_len

        city.driver_decision()
        self.assertAlmostEqual(follower.acceleration, 0.0, delta=0.08,
                               msg="Follower acceleration at equilibrium gap and speed must be ~0 m/s^2")

        # 2. Stopped Leader with Small Gap (Should brake hard)
        leader.velocity = 0.0
        follower.velocity = 10.0
        follower.acceleration = 0.0
        small_gap = 4.0  # smaller than s0
        follower.pos = (leader.pos + leader.length + small_gap) % road_len

        # Step 1: Follower immediately begins braking (jerk-limited to -0.5 in 1 step from 0)
        city.driver_decision()
        self.assertLess(follower.acceleration, 0.0,
                        msg="Follower must immediately begin braking when leader is stopped with small gap")

        # After a few steps of continued braking, acceleration ramps past -1.0 m/s^2
        for _ in range(4):
            city.driver_decision()
            city.move_forward(dt)
        self.assertLess(follower.acceleration, -1.0,
                        msg="Follower must brake hard past -1.0 m/s^2 as jerk limit ramps")

        # 3. Acceleration and Jerk Limits Verification
        # Apply a sequence of extreme situations and confirm limits hold on every step
        prev_a = follower.acceleration
        for _ in range(50):
            city.driver_decision()
            cur_a = follower.acceleration

            self.assertGreaterEqual(cur_a, city.min_a - 1e-6,
                                    msg="Acceleration must not exceed min_a")
            self.assertLessEqual(cur_a, city.max_a + 1e-6,
                                 msg="Acceleration must not exceed max_a")

            jerk = abs(cur_a - prev_a) / dt
            self.assertLessEqual(jerk, 5.0 + 1e-5,
                                 msg="Jerk magnitude must never exceed 5.0 m/s^3")
            prev_a = cur_a
            city.move_forward(dt)

    def test_rear_vehicle_brake_phantom_deceleration(self):
        """
        Verify that when the rear vehicle drops/brakes on a clear road:
        - ACC ignores the rear vehicle and exhibits 0 phantom braking.
        - BCC reacts to the rear vehicle, causing unwanted phantom braking.
        """
        params = dict(DEFAULT_SIM_PARAMS)
        params['car_number'] = 3
        params['v_des'] = 15.0
        dt = 0.1
        init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]

        # 1. ACC simulation
        city_acc = City()
        city_acc.init(*init_args, dt=dt, model='ACC')
        city_acc.follower_velocity_profile = [(0.0, 15.0), (2.0, 15.0), (4.0, 5.0), (10.0, 5.0)]

        acc_decel_steps = 0
        for _ in range(100):
            city_acc.run(dt)
            if city_acc.cars[1].acceleration < -0.5:
                acc_decel_steps += 1

        # 2. BCC simulation
        city_bcc = City()
        city_bcc.init(*init_args, dt=dt, model='BCC')
        city_bcc.follower_velocity_profile = [(0.0, 15.0), (2.0, 15.0), (4.0, 5.0), (10.0, 5.0)]

        bcc_decel_steps = 0
        for _ in range(100):
            city_bcc.run(dt)
            if city_bcc.cars[1].acceleration < -0.5:
                bcc_decel_steps += 1

        self.assertEqual(acc_decel_steps, 0,
                         msg="ACC must completely ignore rear car braking (zero phantom braking)")
        self.assertGreater(bcc_decel_steps, 0,
                           msg="BCC must exhibit rear-induced phantom braking when rear car drops")

    def test_smoke_pipeline(self):
        """Smoke test: the full pipeline runs on a shortened scenario and produces all output files."""
        short_scenario = [{
            "id": "smoke_steady",
            "name": "Smoke Test Scenario",
            "v_des": 15.0,
            "type": "none",
            "dur": 2.0,  # 2 seconds (20 steps)
        }]

        model_path = resolve_model_path("td3_accel_agent.zip")
        self.assertTrue(os.path.exists(model_path), f"Base RL model must exist at {model_path}")

        results = run_driver_intuitiveness_suite(
            model_path=model_path,
            scenarios=short_scenario,
            save_outputs=False,
        )

        self.assertIn('overall_dii', results)
        self.assertIn('overall_idm_gap', results)

        dii_json = LOGS_DIR / "driver_intuitiveness.json"
        idm_json = LOGS_DIR / "idm_comparison.json"
        dii_plot = PLOTS_DIR / "dii_comparison.png"
        idm_plot = PLOTS_DIR / "idm_comparison.png"

        self.assertTrue(dii_json.exists(), f"Log file missing: {dii_json}")
        self.assertTrue(idm_json.exists(), f"Log file missing: {idm_json}")
        self.assertTrue(dii_plot.exists(), f"Plot file missing: {dii_plot}")
        self.assertTrue(idm_plot.exists(), f"Plot file missing: {idm_plot}")

        # Validate JSON content parses properly
        with open(dii_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('overall', data)
            self.assertIn('RL', data['overall'])

        with open(idm_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('gap_matched_reference', data)


def run_intuitiveness_tests(model_path="td3_accel_agent.zip"):
    """Entrypoint function called by main.py --test."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDriverIntuitiveness)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Driver Intuitiveness verification test suite failed!")
    print("\n>>> PASS: All Driver Intuitiveness & IDM reference tests passed successfully!")


if __name__ == '__main__':
    unittest.main()
