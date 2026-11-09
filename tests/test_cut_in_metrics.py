"""
Unit Tests for Cut-In Vulnerability and X-Factor Kinematic Metric Suite.
"""

import unittest
import numpy as np
from stable_baselines3 import TD3

from evaluation.cut_in_metrics import (
    compute_kinematic_x,
    compute_cut_in_metrics,
    DEFAULT_CAR_LENGTH,
    DEFAULT_MIN_GAP,
    DEFAULT_REACTION_TIME,
)
from simulation.city import City
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv
from evaluation.evaluate_all_scenarios import get_scenarios, DEFAULT_SIM_PARAMS


class TestCutInMetrics(unittest.TestCase):

    def test_compute_kinematic_x_scalar(self):
        """Test X-factor calculation with scalar speed inputs."""
        # ve = 15, vl = 15, vf = 15, Tr = 0.8, ae = 5.0, af = 3.5, Le = 5.0, Lb = 2.0
        # Gfront_min = 15 * 0.8 + 0 + 2.0 = 14.0
        # Grear_min = 15 * 0.8 + 0 + 2.0 = 14.0
        # X = 14.0 + 5.0 + 14.0 = 33.0
        x_val = compute_kinematic_x(
            v_ego=15.0,
            v_lead=15.0,
            v_rear=15.0,
            min_a=-5.0,
            reaction_time=0.8,
            min_gap=2.0,
            car_length=5.0,
        )
        self.assertAlmostEqual(x_val, 33.0, places=2)

    def test_compute_kinematic_x_relative_speed(self):
        """Test that relative speed increases braking buffer quadratically."""
        # Leader is moving slower (closing gap fast): vl = 10, ve = 15
        # (vl - ve)^2 / (2 * ae) = 25 / 10 = 2.5m extra front buffer
        x_val = compute_kinematic_x(
            v_ego=15.0,
            v_lead=10.0,
            v_rear=15.0,
            min_a=-5.0,
            reaction_time=0.8,
            min_gap=2.0,
            car_length=5.0,
        )
        self.assertAlmostEqual(x_val, 33.0 + 2.5, places=2)

    def test_compute_kinematic_x_vectorized(self):
        """Test vectorized computation over numpy arrays."""
        speeds = np.array([0.0, 10.0, 15.0, 20.0])
        xs = compute_kinematic_x(speeds)
        self.assertEqual(len(xs), 4)
        # At standstill v=0: Gfront = 2, Grear = 2, Le = 5 => X = 9.0
        self.assertAlmostEqual(xs[0], 9.0, places=2)
        # X should increase monotonically with speed
        self.assertTrue(np.all(np.diff(xs) > 0))

    def test_steady_cruise_cut_in_metrics(self):
        """A steady 18m gap at 15 m/s should have zero X-breach and low cut-in risk."""
        dt = 0.1
        steps = 200  # 20s
        gaps = [18.0] * steps
        speeds = [15.0] * steps

        m = compute_cut_in_metrics(
            gaps=gaps,
            speeds=speeds,
            dt=dt,
            v_des=15.0,
            reaction_time=0.8,
            min_gap=2.0,
        )
        self.assertEqual(m['t_x_breach'], 0.0)
        self.assertEqual(m['pct_x_breach'], 0.0)
        self.assertEqual(m['civi_x'], 0.0)
        self.assertEqual(m['civi_x_score'], 0.0)
        self.assertLess(m['max_x_ratio'], 1.0)
        self.assertEqual(m['t_headway_2s'], 0.0)
        self.assertEqual(m['p_cut_in'], 0.0)

    def test_severe_gap_ballooning_metrics(self):
        """A gap expanding to 150m should flag severe X-breach and cut-in risk."""
        dt = 0.1
        steps = 250
        # Gap starts at 18m and expands linearly to 150m
        gaps = np.linspace(18.0, 150.0, steps)
        speeds = [15.0] * steps

        m = compute_cut_in_metrics(
            gaps=gaps,
            speeds=speeds,
            dt=dt,
            v_des=15.0,
            reaction_time=0.8,
            min_gap=2.0,
        )
        self.assertGreater(m['max_x_ratio'], 3.0)
        self.assertGreater(m['t_x_breach'], 15.0)
        self.assertGreater(m['civi_x'], 1.0)
        self.assertGreater(m['p_cut_in'], 80.0)
        self.assertGreater(m['t_headway_2s'], 15.0)

    def test_scenario6_bcc_vs_rl_cut_in_vulnerability(self):
        """
        Verify on Scenario 6 (three_car_trailing_halt):
        - BCC severely inflates gap (Xi > 3.0, t_breach > 15s)
        - RL caps gap dilation below X (Xi <= 1.0, t_breach == 0s)
        """
        sc_list = [s for s in get_scenarios() if s['id'] == 'three_car_trailing_halt']
        self.assertTrue(len(sc_list) > 0, "Scenario three_car_trailing_halt not found")
        sc = sc_list[0]

        params = dict(DEFAULT_SIM_PARAMS)
        params['v_des'] = sc['v_des']
        params['car_number'] = 3
        init_args = [
            params[k] for k in [
                'car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v',
                'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap'
            ]
        ]

        # 1. Run BCC
        c_bcc = City()
        c_bcc.init(*init_args, dt=0.1, model='BCC')
        c_bcc.follower_velocity_profile = list(sc['follower_profile'])
        c_bcc.cars[-1].is_human = True

        gaps_bcc, speeds_bcc, vl_bcc, vb_bcc = [], [], [], []
        for _ in range(int(sc['dur'] / 0.1)):
            c_bcc.run(0.1)
            ego = c_bcc.cars[1]
            front = c_bcc.cars[0]
            back = c_bcc.cars[2]
            fg = (ego.pos - front.pos - front.length) % 1000
            gaps_bcc.append(fg)
            speeds_bcc.append(ego.velocity)
            vl_bcc.append(front.velocity)
            vb_bcc.append(back.velocity)

        m_bcc = compute_cut_in_metrics(
            gaps=gaps_bcc,
            speeds=speeds_bcc,
            lead_speeds=vl_bcc,
            rear_speeds=vb_bcc,
            dt=0.1,
            v_des=sc['v_des'],
        )

        # 2. Run RL
        c_rl = RLCity()
        c_rl.init(*init_args, dt=0.1, model='ACC')
        c_rl.follower_velocity_profile = list(sc['follower_profile'])
        c_rl.cars[0].is_rl = False
        c_rl.cars[2].is_rl = False
        c_rl.cars[2].is_human = True
        c_rl.cars[1].is_rl = True
        c_rl.cars[1].rl_action = 0.0
        rl_model = TD3.load("models/td3_accel_agent.zip")

        gaps_rl, speeds_rl, vl_rl, vb_rl = [], [], [], []
        for _ in range(int(sc['dur'] / 0.1)):
            obs_arr = AccelTrafficEnv.get_city_obs(c_rl, params['v_des'])
            act, _ = rl_model.predict(obs_arr, deterministic=True)
            c_rl.cars[1].rl_action = float(act[0][0] if np.ndim(act) > 1 else act[0])
            c_rl.run(0.1)
            ego = c_rl.cars[1]
            front = c_rl.cars[0]
            back = c_rl.cars[2]
            fg = (ego.pos - front.pos - front.length) % 1000
            gaps_rl.append(fg)
            speeds_rl.append(ego.velocity)
            vl_rl.append(front.velocity)
            vb_rl.append(back.velocity)

        m_rl = compute_cut_in_metrics(
            gaps=gaps_rl,
            speeds=speeds_rl,
            lead_speeds=vl_rl,
            rear_speeds=vb_rl,
            dt=0.1,
            v_des=sc['v_des'],
        )

        # Assertions
        # BCC must have severe X-breach
        self.assertGreater(m_bcc['max_x_ratio'], 3.0)
        self.assertGreater(m_bcc['t_x_breach'], 15.0)
        self.assertGreater(m_bcc['civi_x'], 1.0)
        self.assertGreater(m_bcc['p_cut_in'], 90.0)

        # RL must NOT breach X-envelope
        self.assertLessEqual(m_rl['max_x_ratio'], 1.0)
        self.assertEqual(m_rl['t_x_breach'], 0.0)
        self.assertEqual(m_rl['civi_x'], 0.0)


if __name__ == '__main__':
    unittest.main()

