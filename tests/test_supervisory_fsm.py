"""
Unit tests for the Modular Finite-State Supervisory Machine (FSM) and 6D Car-Following Architecture.
Verifies:
- State transitions between FREE_CRUISE and CAR_FOLLOWING
- 5-meter hysteresis boundary (65m acquire, 70m release)
- Open-road free-cruise velocity stabilization (eliminating runaway acceleration)
- 6D canonical observation space and 7D legacy compatibility
"""

import unittest
import numpy as np
from simulation.supervisor import VehicleSupervisor, FSMState
from simulation.car import Car
from simulation.road import Road
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv


class TestSupervisoryFSM(unittest.TestCase):
    def setUp(self):
        self.supervisor = VehicleSupervisor()
        self.road = Road(1000, 0, 0, 1, 0)

    def test_01_fsm_initial_state(self):
        """Verify default supervisor starts in FREE_CRUISE."""
        self.assertEqual(self.supervisor.state, FSMState.FREE_CRUISE)
        self.assertEqual(self.supervisor.mode_switches, 0)

    def test_02_target_acquisition_at_65m(self):
        """Verify transition to CAR_FOLLOWING occurs when front_gap <= 65m."""
        # 68m (within radar, but above 65m acquire): remains FREE_CRUISE
        dummy_car = Car(4, '#38bdf8', 900, 6.0, 15.0, 0, self.road)
        state = self.supervisor.update(dummy_car, front_gap=68.0, dt=0.1)
        self.assertEqual(state, FSMState.FREE_CRUISE)

        # 65m (exact acquire boundary): transitions to CAR_FOLLOWING
        state = self.supervisor.update(dummy_car, front_gap=65.0, dt=0.1)
        self.assertEqual(state, FSMState.CAR_FOLLOWING)
        self.assertEqual(self.supervisor.mode_switches, 1)

    def test_03_hysteresis_band_stability(self):
        """Verify that within 65m < front_gap < 70m, existing state is preserved."""
        dummy_car = Car(4, '#38bdf8', 900, 6.0, 15.0, 0, self.road)
        
        # 1. Acquire at 60m
        self.supervisor.update(dummy_car, front_gap=60.0, dt=0.1)
        self.assertEqual(self.supervisor.state, FSMState.CAR_FOLLOWING)

        # 2. Open up to 67m (between 65m and 70m): should STAY in CAR_FOLLOWING
        self.supervisor.update(dummy_car, front_gap=67.0, dt=0.1)
        self.assertEqual(self.supervisor.state, FSMState.CAR_FOLLOWING)
        self.assertEqual(self.supervisor.mode_switches, 1)

        # 3. Breach release boundary at 71m: transitions to FREE_CRUISE
        self.supervisor.update(dummy_car, front_gap=71.0, dt=0.1)
        self.assertEqual(self.supervisor.state, FSMState.FREE_CRUISE)
        self.assertEqual(self.supervisor.mode_switches, 2)

        # 4. Move to 67m from FREE_CRUISE: should STAY in FREE_CRUISE
        self.supervisor.update(dummy_car, front_gap=67.0, dt=0.1)
        self.assertEqual(self.supervisor.state, FSMState.FREE_CRUISE)
        self.assertEqual(self.supervisor.mode_switches, 2)

    def test_04_open_road_free_cruise_stabilization(self):
        """Verify that when left alone on a highway, RLCity regulates speed to v_des without runaway acceleration."""
        city = RLCity()
        # Initialize 1 car alone on the road (car 0) or 2 cars separated by 200m
        city.init(
            car_number=2, kd=0.9, kv=0.5, kc=0.4,
            v_des=20.0, max_v=35.0, min_v=0.0,
            min_dis=6.0, reaction_time=0.8, headway_time=1.0,
            max_a=3.0, min_a=-5.0, min_gap=2.0, dt=0.1
        )
        # Position car 1 far behind car 0 (200m gap >> 70m radar limit)
        city.cars[0].pos = 500.0
        city.cars[1].pos = 700.0
        city.cars[1].velocity = 10.0  # Starting below v_des (20.0 m/s)
        city.cars[1].is_rl = True
        city.cars[1].rl_action = 3.0  # Even if RL outputs full gas, FSM should override to FREE_CRUISE

        # Run 10 seconds of simulation (100 steps)
        for _ in range(100):
            city.run(0.1)

        car1 = city.cars[1]
        # Car 1 must be in FREE_CRUISE
        self.assertEqual(car1.fsm_state, FSMState.FREE_CRUISE.value)
        # Car 1 velocity must smoothly level off near v_des (20.0 m/s), NOT runaway to max_v (35.0 m/s)
        self.assertAlmostEqual(car1.velocity, 20.0, delta=1.5)
        self.assertLess(car1.velocity, 25.0)

    def test_05_observation_space_6d_and_7d(self):
        """Verify get_city_obs computes 6D canonical and 7D legacy observations."""
        city = RLCity()
        city.init(
            car_number=4, kd=0.9, kv=0.5, kc=0.4,
            v_des=15.0, max_v=30.0, min_v=0.0,
            min_dis=6.0, reaction_time=0.8, headway_time=1.0,
            max_a=3.0, min_a=-5.0, min_gap=2.0, dt=0.1
        )
        # 6D canonical
        obs_6d = AccelTrafficEnv.get_city_obs(city, v_des=15.0, obs_dim=6)
        self.assertEqual(obs_6d.shape, (3, 6))
        self.assertFalse(np.isnan(obs_6d).any())

        # 7D legacy
        obs_7d = AccelTrafficEnv.get_city_obs(city, v_des=15.0, obs_dim=7)
        self.assertEqual(obs_7d.shape, (3, 7))
        self.assertFalse(np.isnan(obs_7d).any())

    def test_06_env_canonical_6d_step(self):
        """Verify AccelTrafficEnv steps cleanly with canonical 6D observation space."""
        env = AccelTrafficEnv(num_cars=4, max_steps=50, obs_dim=6)
        self.assertEqual(env.observation_space.shape, (6,))
        obs = env.reset()
        self.assertEqual(obs.shape, (3, 6))

        actions = np.zeros((3, 1), dtype=np.float32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        self.assertEqual(obs.shape, (3, 6))
        self.assertEqual(rewards.shape, (3,))


if __name__ == '__main__':
    unittest.main()

