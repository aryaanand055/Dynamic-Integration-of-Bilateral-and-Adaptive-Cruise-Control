import os
import unittest
import numpy as np
from pathlib import Path

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import DEFAULT_SIM_PARAMS, resolve_model_path, DATA_DIR
from env.accel_traffic_env import AccelTrafficEnv
from evaluation.evaluate_all_scenarios import get_scenarios, load_csv_profile
from stable_baselines3 import TD3


class TestSensorDelayAndHighwayScenario(unittest.TestCase):
    def setUp(self):
        self.params = dict(DEFAULT_SIM_PARAMS)
        self.init_args = [
            self.params[k] for k in [
                'car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v',
                'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap'
            ]
        ]

    def test_sensor_delay_zero(self):
        """Zero sensor delay and noise must return bit-for-bit exact current car states."""
        city = City()
        city.init(*self.init_args, dt=0.1, model='ACC', sensor_delay=0.0, sensor_noise=0.0)
        current = [(c.pos, c.velocity) for c in city.cars]
        perceived = city.get_perceived_car_states()
        self.assertEqual(current, perceived)

    def test_sensor_delay_propagation(self):
        """Sensor delay of 0.2s (2 steps at dt=0.1s) must accurately lag behind ground truth."""
        dt = 0.1
        city = City()
        city.init(*self.init_args, dt=dt, model='ACC', sensor_delay=0.2, sensor_noise=0.0)

        # Run 5 steps to populate the delay buffer
        history = []
        for _ in range(5):
            history.append([(c.pos, c.velocity) for c in city.cars])
            city.run(dt)

        # At this point, delay buffer should serve state from ~2 steps ago
        perceived = city.get_perceived_car_states()
        # Perceived should match the 2-step delayed state from history
        expected_delayed = history[-3]  # 2 steps back from current
        for (p_act, v_act), (p_exp, v_exp) in zip(perceived, expected_delayed):
            self.assertAlmostEqual(p_act, p_exp, places=4)
            self.assertAlmostEqual(v_act, v_exp, places=4)

    def test_sensor_noise_variation(self):
        """Sensor noise > 0 should perturb perceived readings while remaining finite."""
        city = City()
        city.init(*self.init_args, dt=0.1, model='ACC', sensor_delay=0.0, sensor_noise=0.5)
        city.run(0.1)

        perceived = city.get_perceived_car_states()
        truth = [(c.pos, c.velocity) for c in city.cars]

        differences = [abs(p[0] - t[0]) for p, t in zip(perceived, truth)]
        # At least one car should have non-zero noise perturbation
        self.assertTrue(any(d > 1e-4 for d in differences))
        # Velocities must never be negative
        for p in perceived:
            self.assertGreaterEqual(p[1], 0.0)

    def test_env_obs_with_delay(self):
        """AccelTrafficEnv.get_city_obs must operate without error with delay active."""
        city = RLCity()
        city.init(*self.init_args, dt=0.1, model='ACC', sensor_delay=0.2, sensor_noise=0.1)
        city.run(0.1)

        obs = AccelTrafficEnv.get_city_obs(city, v_des=15.0, obs_dim=6)
        self.assertEqual(obs.shape, (len(city.cars) - 1, 6))
        self.assertFalse(np.isnan(obs).any())
        self.assertFalse(np.isinf(obs).any())
        obs_7d = AccelTrafficEnv.get_city_obs(city, v_des=15.0, obs_dim=7)
        self.assertEqual(obs_7d.shape, (len(city.cars) - 1, 7))

    def test_scenario_11_registration(self):
        """Ensure official benchmark suite includes exactly 11 curated scenarios."""
        scenarios = get_scenarios()
        self.assertEqual(len(scenarios), 11)

        sc11 = scenarios[10]
        self.assertEqual(sc11['id'], 'highway_empirical_highd')
        self.assertEqual(sc11['type'], 'csv')
        self.assertEqual(sc11['csv_file'], 'highway_traffic.csv')
        self.assertEqual(sc11['car_number'], 15)
        self.assertAlmostEqual(sc11['dur'], 60.0)

    def test_scenario_11_csv_loadable(self):
        """Verify highway_traffic.csv exists and has authentic highway speeds."""
        profile = load_csv_profile("highway_traffic.csv")
        self.assertGreater(len(profile), 30)
        self.assertAlmostEqual(profile[-1][0], 60.0)

        # Highway speeds should fluctuate between 15.0 and 30.0 m/s
        speeds = [v for t, v in profile]
        self.assertGreaterEqual(min(speeds), 15.0)
        self.assertLessEqual(max(speeds), 30.0)

    def test_scenario_11_short_simulation(self):
        """Test that Scenario 11 runs seamlessly across all 4 controllers with sensor delay."""
        scenarios = get_scenarios()
        sc11 = scenarios[10]
        lead_profile = load_csv_profile(sc11['csv_file'])

        dt = 0.1
        steps = 20  # 2.0 seconds smoke test

        for model_type in ['ACC', 'BCC', 'ACC+BCC']:
            city = City()
            city.init(*self.init_args, dt=dt, model=model_type, sensor_delay=0.2, sensor_noise=0.05)
            city.lead_velocity_profile = lead_profile
            for _ in range(steps):
                city.run(dt)
            self.assertGreater(len(city.cars), 1)

        # Test RL controller on Scenario 11
        model_file = resolve_model_path('td3_accel_agent.zip')
        if os.path.exists(model_file):
            rl_model = TD3.load(model_file)
            city_rl = RLCity()
            city_rl.init(*self.init_args, dt=dt, model='ACC', sensor_delay=0.2, sensor_noise=0.05)
            city_rl.lead_velocity_profile = lead_profile
            for c in city_rl.cars[1:]:
                c.is_rl = True
            obs_dim = rl_model.observation_space.shape[0]
            for _ in range(steps):
                obs = AccelTrafficEnv.get_city_obs(city_rl, v_des=24.0, obs_dim=obs_dim)
                act, _ = rl_model.predict(obs, deterministic=True)
                for i, car in enumerate(city_rl.cars[1:]):
                    act_val = act[i][0] if (np.ndim(act) > 1 and i < len(act)) else (act[i] if i < len(act) else 0.0)
                    car.rl_action = float(act_val)
                city_rl.run(dt)
            self.assertFalse(city_rl.collided_this_step)


if __name__ == '__main__':
    unittest.main()

