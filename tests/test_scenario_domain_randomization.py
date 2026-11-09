"""
Unit tests for Scenario-Conditioned Domain Randomization (SCDR) in AccelTrafficEnv.
Verifies all 11 scenario archetypes, dynamic cut-in / cut-out triggers, tailgater surge,
mixed IDM traffic injection, and sensor delay/noise handling.
"""

import unittest
import numpy as np
from env.accel_traffic_env import AccelTrafficEnv


class TestScenarioDomainRandomization(unittest.TestCase):
    def setUp(self):
        self.env = AccelTrafficEnv(num_cars=6, max_steps=100)

    def test_01_archetype_registry(self):
        """Verify that all 11 scenario archetypes are registered."""
        self.assertEqual(len(AccelTrafficEnv.SCENARIO_ARCHETYPES), 11)
        expected = [
            'steady_cruise',
            'urban_stop_and_go',
            'full_stop_queue',
            'lead_emergency_braking',
            'sinusoidal_wave',
            'lead_cut_in',
            'lead_cut_out',
            'aggressive_tailgater',
            'speed_drop_recovery',
            'mixed_idm_traffic',
            'highway_empirical_highd'
        ]
        for name in expected:
            self.assertIn(name, AccelTrafficEnv.SCENARIO_ARCHETYPES)

    def test_02_all_archetypes_initialization(self):
        """Verify that every archetype initializes and generates valid 6D canonical observations."""
        for arch in AccelTrafficEnv.SCENARIO_ARCHETYPES:
            obs = self.env.reset(scenario_archetype=arch)
            self.assertEqual(self.env.scenario_archetype, arch)
            self.assertEqual(obs.shape, (5, 6))
            self.assertFalse(np.isnan(obs).any())
            self.assertFalse(np.isinf(obs).any())

    def test_03_lead_cut_in_execution(self):
        """Verify dynamic cut-in trigger shrinks Car 1's front gap to insertion distance."""
        self.env.reset(scenario_archetype='lead_cut_in')
        self.assertTrue(self.env.cut_in_active)
        self.assertFalse(self.env.cut_in_triggered)
        
        # Override cut-in step to step 5 for fast testing
        self.env.cut_in_step = 5
        self.env.cut_in_gap = 5.2
        self.env.cut_in_vel_delta = -1.5
        
        # Step until step 4 (pre-cut-in)
        for _ in range(4):
            self.env.step_wait()
        self.assertFalse(self.env.cut_in_triggered)
        
        # Step 5: Cut-in fires
        obs, rewards, dones, infos = self.env.step_wait()
        self.assertTrue(self.env.cut_in_triggered)
        
        # Check that car0 is positioned ahead of car1 by approximately cut_in_gap
        road_len = self.env.city.roads[0].length
        car1 = self.env.city.cars[1]
        car0 = self.env.city.cars[0]
        actual_gap = (car1.pos - car0.pos - car0.length) % road_len
        # Accounting for single-step motion update (0.1s * v)
        self.assertAlmostEqual(actual_gap, 5.2, delta=2.5)

    def test_04_lead_cut_out_execution(self):
        """Verify dynamic cut-out trigger expands Car 1's front gap."""
        self.env.reset(scenario_archetype='lead_cut_out')
        self.assertTrue(self.env.cut_out_active)
        self.assertFalse(self.env.cut_out_triggered)
        
        self.env.cut_out_step = 5
        self.env.cut_out_gap_shift = 40.0
        
        for _ in range(4):
            self.env.step_wait()
        gap_before = (self.env.city.cars[1].pos - self.env.city.cars[0].pos - self.env.city.cars[0].length) % 1000
        
        self.env.step_wait()
        self.assertTrue(self.env.cut_out_triggered)
        gap_after = (self.env.city.cars[1].pos - self.env.city.cars[0].pos - self.env.city.cars[0].length) % 1000
        self.assertGreater(gap_after, gap_before + 20.0)

    def test_05_aggressive_tailgater_execution(self):
        """Verify aggressive tailgater activates reckless trailing driver on last car."""
        self.env.reset(scenario_archetype='aggressive_tailgater')
        self.assertTrue(self.env.tailgater_active)
        self.env.tailgater_step = 3
        
        for _ in range(2):
            self.env.step_wait()
        last_car = self.env.city.cars[-1]
        self.assertFalse(last_car.is_reckless)
        
        self.env.step_wait()
        self.assertTrue(last_car.is_reckless)
        self.assertFalse(last_car.is_rl)
        self.assertEqual(last_car.color, '#ef4444')

    def test_06_mixed_idm_traffic_execution(self):
        """Verify mixed IDM traffic seeds human driver profiles into follower vehicles."""
        self.env.reset(scenario_archetype='mixed_idm_traffic')
        self.assertTrue(self.env.mixed_traffic_active)
        
        human_cars = [c for c in self.env.city.cars if getattr(c, 'is_human', False)]
        self.assertGreaterEqual(len(human_cars), 1)
        for hc in human_cars:
            self.assertEqual(hc.driver_type, 'HUMAN')
            self.assertFalse(hc.is_rl)
            self.assertIsNotNone(hc.human_profile)
            self.assertIn('a_max', hc.human_profile)
            self.assertIn('s0', hc.human_profile)

    def test_07_perception_delay_and_noise_integration(self):
        """Verify that training step executes cleanly with perception delay and noise."""
        self.env.reset(scenario_archetype='steady_cruise')
        self.env.sim_params['sensor_delay'] = 0.2
        self.env.sim_params['sensor_noise'] = 0.15
        self.env.city.sensor_delay = 0.2
        self.env.city.sensor_noise = 0.15
        
        actions = np.zeros((5, 1), dtype=np.float32)
        for _ in range(10):
            self.env.step_async(actions)
            obs, rewards, dones, infos = self.env.step_wait()
            self.assertEqual(obs.shape, (5, 6))
            self.assertEqual(rewards.shape, (5,))
            self.assertFalse(np.isnan(obs).any())


if __name__ == '__main__':
    unittest.main()

