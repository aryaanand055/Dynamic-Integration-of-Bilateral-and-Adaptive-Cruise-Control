"""
Zhang et al. (2026) ACC-CPEIM scenarios P1-P6 for the repo's ring-road simulator.

Machines 14(7), 827, doi:10.3390/machines14070827.
The paper fixes the scenario types, case settings (P1 initial spacing 5-9 m, P2-P6 speeds
30-120 km/h), indicators, weights and Table 7 thresholds. It does not give lead-vehicle
manoeuvre magnitudes, so those are our choices and are listed in SETTINGS below.

Each rollout returns 0.1 s traces of the evaluated host (ego) car and its rear neighbour,
for ACC, BCC, ACC+BCC, TD3 (committed td3_accel_agent.zip = TD3_56, with its 0.35
actuator filter) and IDM (gap-matched, s0=6 m, T=0.8 s).
"""
import os, sys, math, copy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from simulation.city import City
from simulation.rl_city import RLCity
from simulation.car import Car
from simulation.config import DEFAULT_SIM_PARAMS
from env.accel_traffic_env import AccelTrafficEnv
from evaluation.driver_intuitiveness import IDMCity, get_gap_matched_idm_params

KMH = [30, 60, 80, 100]          # 120 km/h dropped: sim max_v is 30 m/s and TD3 trained on 12-28 m/s
P1_GAPS = [5, 6, 7, 8, 9]
TE = 10.0                        # event time for P2-P6 (s)
SETTINGS = {
    'P1': 'all cars stopped; lead holds 2 s then accelerates 0->10 m/s at 2 m/s^2; ego gap 5-9 m',
    'P2': 'lead at constant V; platoon starts at the controller equilibrium gap (6 m + 0.8 s * V)',
    'P3': 'lead accelerates V -> V+5 m/s at 1.5 m/s^2 at t=10 s',
    'P4': 'lead decelerates V -> 0.5 V at -3 m/s^2 at t=10 s',
    'P5': 'car directly ahead of ego leaves the lane at t=10 s; next car (same speed V) becomes the target',
    'P6': 'a car at 0.9 x ego speed enters the lane at t=10 s, half-way into the ego gap',
}
WINDOW = {'P1': 20.0, 'P2': None, 'P3': 10.0, 'P4': 15.0, 'P5': 15.0, 'P6': 15.0}
CONTROLLERS = ['ACC', 'BCC', 'ACC+BCC', 'TD3_56', 'IDM']


class FilteredRLCity(RLCity):
    """HEAD RLCity + the TD3_56-era first-order actuator filter (alpha 0.35, b392b75 rl_city.py)."""
    def driver_decision(self):
        for car in self.cars:
            if getattr(car, 'is_rl', False) and hasattr(car, 'rl_action'):
                raw = float(car.rl_action)
                prev = getattr(car, '_filtered_cmd', raw)
                if car.velocity <= 1e-4 and raw > 0.0:
                    prev = max(0.0, prev)
                car.rl_action = 0.35 * raw + 0.65 * prev
                car._filtered_cmd = car.rl_action
        super().driver_decision()


class FixedIDMCity(IDMCity):
    """IDMCity overwrites every follower, including scripted cut-in/cut-out cars.
    Re-apply the scripted profile to those cars, as City.driver_decision does."""
    def driver_decision(self):
        prof = getattr(self, 'car_velocity_profiles', {}) or {}
        prev = {i: self.cars[i].acceleration for i in prof if i < len(self.cars)}
        super().driver_decision()
        t = round(self.step_count * self.dt, 3)
        for i, pr in prof.items():
            if i >= len(self.cars) or getattr(self.cars[i], 'has_exited', False):
                continue
            car = self.cars[i]
            vt = float(np.interp(t, [a for a, _ in pr], [b for _, b in pr]))
            acc = max(self.min_a, min(self.max_a, (vt - car.velocity) / self.dt))
            last = prev[i]
            acc = max(last - 0.5, min(last + 0.5, acc))   # 5 m/s^3 jerk limit at dt=0.1
            car.acceleration = acc


def _make_city(ctrl, params):
    args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis',
                                'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]
    if ctrl == 'TD3_56':
        c = FilteredRLCity(); model = 'ACC'
    elif ctrl == 'IDM':
        c = FixedIDMCity(idm_params=get_gap_matched_idm_params(params)); model = 'ACC'
    else:
        c = City(); model = ctrl
    c.init(*args, dt=params['dt'], model=model)
    return c


def _new_car(city, pos, v):
    car = Car(length=4, color='#ef4444', pos=pos, min_dis=city.min_dis, velocity=v, acceleration=0.0,
              current_road=city.roads[0])
    car.is_human = True; car.driver_type = 'HUMAN'; car.collision_timer = 0
    car.gap_history = []; car._prev_acc = 0.0; car._filtered_cmd = 0.0
    car.just_collided = car.just_collided_front = car.just_collided_rear = False
    return car


def _gaps(city, car):
    L = city.roads[0].length
    others = [c for c in city.cars if c is not car and not getattr(c, 'has_exited', False)]
    f = min(others, key=lambda o: ((car.pos - o.pos) % L) or float('inf'))
    b = min(others, key=lambda o: ((o.pos - car.pos) % L) or float('inf'))
    return ((car.pos - f.pos - f.length) % L, f.velocity, (b.pos - car.pos - car.length) % L)


def run_case(ctrl, scen, x, model=None):
    """x = speed in km/h (P2-P6) or initial spacing in m (P1). Returns trace dict for the ego car."""
    p = dict(DEFAULT_SIM_PARAMS); p['max_v'] = 35.0; p['dt'] = 0.1
    if scen == 'P1':
        V = 0.0; lead = [(0.0, 0.0), (2.0, 0.0), (7.0, 10.0), (60.0, 10.0)]; te = 2.0; vmax = 10.0
    else:
        V = x / 3.6; te = TE
        lead = {'P2': [(0.0, V), (60.0, V)],
                'P3': [(0.0, V), (te, V), (te + 5.0 / 1.5, V + 5.0), (60.0, V + 5.0)],
                'P4': [(0.0, V), (te, V), (te + 0.5 * V / 3.0, 0.5 * V), (60.0, 0.5 * V)],
                'P5': [(0.0, V), (60.0, V)],
                'P6': [(0.0, V), (60.0, V)]}[scen]
        vmax = max(v for _, v in lead)
    n_cars = 4 if scen == 'P5' else 3
    p['car_number'] = n_cars; p['v_des'] = max(V, 1.0)
    city = _make_city(ctrl, p)
    # set speed above the lead's top speed so the controller is in following mode
    city.v_des = min(p['max_v'], max(1.1 * vmax, 5.0))
    city.lead_velocity_profile = lead
    ego_idx = 2 if scen == 'P5' else 1
    if scen == 'P1':
        for c in city.cars: c.velocity = 0.0
        for i in range(1, n_cars):
            city.cars[i].pos = city.cars[i - 1].pos + city.cars[i - 1].length + x
    if scen == 'P5':
        city.car_velocity_profiles = {1: [(0.0, V), (60.0, V)]}
        city.cars[1].is_human = True
    if ctrl == 'TD3_56':
        for i, c in enumerate(city.cars):
            c.is_rl = (i > 0 and not getattr(c, 'is_human', False)); c.rl_action = 0.0
    ego = city.cars[ego_idx]
    dur = te + (WINDOW[scen] or 40.0) + 5.0
    T = {k: [] for k in ['t', 'a', 'v', 'gap', 'vf', 'bg']}
    obs_dim = model.observation_space.shape[0] if model is not None else 7
    for step in range(int(round(dur / p['dt']))):
        t = step * p['dt']
        if scen == 'P5' and abs(t - te) < 1e-6:
            city.cars[1].has_exited = True
        if scen == 'P6' and abs(t - te) < 1e-6:
            g, _, _ = _gaps(city, ego)
            nc = _new_car(city, ego.pos - ego.length - 0.5 * g, 0.9 * ego.velocity)
            city.cars.insert(1, nc); city.roads[0].enter_road(nc)
            city.car_velocity_profiles = {1: [(0.0, 0.9 * ego.velocity), (100.0, 0.9 * ego.velocity)]}
        if ctrl == 'TD3_56':
            obs = AccelTrafficEnv.get_city_obs(city, city.v_des, obs_dim=obs_dim)
            act, _ = model.predict(obs, deterministic=True)
            for i, c in enumerate(city.cars[1:]):
                if getattr(c, 'is_rl', False):
                    c.rl_action = float(np.ravel(act[i])[0])
        city.run(p['dt'])
        g, vf, bg = _gaps(city, ego)
        T['t'].append(t + p['dt']); T['a'].append(ego.acceleration); T['v'].append(ego.velocity)
        T['gap'].append(g); T['vf'].append(vf); T['bg'].append(bg)
    T = {k: np.array(v) for k, v in T.items()}
    T['te'] = te; T['scen'] = scen; T['x'] = x; T['ctrl'] = ctrl
    return T


def all_cases():
    for s in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        for x in (P1_GAPS if s == 'P1' else KMH):
            yield s, x
