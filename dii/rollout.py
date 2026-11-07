"""Replay a controller on the 7 official benchmark scenarios and dump full trajectories.

Usage: python dii/rollout.py <version_dir> <controller> <out.npz> [bench|random]
version_dir: a checkout of the repo at some commit (use "." for the current tree)
controller: ACC | BCC | ACC+BCC | IDM | RL
Each version is run with its OWN simulation/env code so old checkpoints see the
observation layout they were trained with.
"""
import sys, os, csv, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
vdir = os.path.abspath(sys.argv[1]); controller = sys.argv[2]; out = os.path.abspath(sys.argv[3])
SET = sys.argv[4] if len(sys.argv) > 4 else 'bench'
sys.path.insert(0, vdir)
os.chdir(vdir)
flat = os.path.exists(os.path.join(vdir, 'city.py'))
if flat:
    from city import City
    from rl_city import RLCity
    import accel_traffic_env as envmod
else:
    from simulation.city import City
    from simulation.rl_city import RLCity
    import env.accel_traffic_env as envmod
Env = envmod.AccelTrafficEnv

# One fixed copy of the lead profiles (the repo's data/ folder), so every checkpoint sees the same inputs.
DATA = os.environ.get('DII_DATA', os.path.join(HERE, '..', 'data'))
P = dict(car_number=15, kd=0.9, kv=0.5, kc=0.4, v_des=15.0, max_v=30.0, min_v=0.0, min_dis=6.0,
         reaction_time=0.8, headway_time=1.0, max_a=3.0, min_a=-5.0, min_gap=2.0, dt=0.1)

def load_csv(name):
    with open(os.path.join(DATA, name)) as f:
        return [(float(r['time']), float(r['velocity'])) for r in csv.DictReader(f)]

SCEN = [
    ("steady_15", 15.0, None, 60.0),
    ("steady_25", 25.0, None, 50.0),
    ("stop_and_go", 15.0, load_csv('data.csv'), 60.0),
    ("speed_drop", 15.0, load_csv('data1.csv'), 60.0),
    ("full_stop_hold", 15.0, [(0.0, 15.0), (12.0, 15.0), (16.0, 0.0), (45.0, 0.0), (50.0, 12.0), (60.0, 15.0)], 60.0),
    ("emergency_brake", 20.0, [(0.0, 20.0), (10.0, 20.0), (13.5, 0.0), (40.0, 0.0)], 40.0),
    ("sine_wave", 15.0, [(t, 15.0 + 4.0 * np.sin(2 * np.pi * t / 15.0)) for t in np.arange(0, 61.0, 0.5)], 60.0),
]

def random_profile(seed, dur=60.0):
    """Unseen lead-vehicle profiles: holds, brakes (incl. full stops), accelerations with 2-6 s ramps."""
    rng = np.random.default_rng(seed)
    v = float(rng.choice([10.0, 15.0, 20.0])); t = 0.0; prof = [(0.0, v)]
    while t < dur:
        t += rng.uniform(5.0, 12.0); prof.append((t, v))
        ev = rng.choice(['brake', 'stop', 'accel'], p=[0.45, 0.2, 0.35])
        v = 0.0 if ev == 'stop' else (max(0.0, v - rng.uniform(3, 10)) if ev == 'brake' else min(25.0, v + rng.uniform(3, 8)))
        t += rng.uniform(2.0, 6.0); prof.append((t, v))
    return float(prof[0][1]), prof

if SET == 'random':
    SCEN = []
    for k in range(20):
        v0, prof = random_profile(1000 + k)
        SCEN.append((f"rand{k:02d}", max(15.0, v0), prof, 60.0))

# Human reference: Intelligent Driver Model (Treiber et al. 2000) with human-calibrated comfort
# parameters; spacing policy (s0, T) matched to the project so only the response style differs.
IDM = dict(a=1.4, b=2.0, delta=4.0, T=0.8, s0=6.0)

def idm_acc(v, dv_closing, gap, v0):
    # Improved IDM (Treiber & Kesting 2013): exact zero acceleration at the equilibrium gap
    s_star = IDM['s0'] + max(0.0, v * IDM['T'] + v * dv_closing / (2 * np.sqrt(IDM['a'] * IDM['b'])))
    z = s_star / max(gap, 0.1)
    a_free = IDM['a'] * (1 - (v / max(v0, 0.1)) ** IDM['delta'])
    if z >= 1:
        return IDM['a'] * (1 - z ** 2)
    return a_free * (1 - z ** (2 * IDM['a'] / max(a_free, 1e-6)))

class IDMCity(City):
    def driver_decision(self):
        prev = [c.acceleration for c in self.cars]
        super().driver_decision()     # leader follows its profile; followers overwritten below
        L = self.roads[0].length
        for i, car in enumerate(self.cars):
            if i == 0: continue
            front = self.cars[i - 1]
            gap = (car.pos - front.pos - front.length) % L
            acc = idm_acc(car.velocity, car.velocity - front.velocity, gap, self.max_v)
            acc = max(self.min_a, min(self.max_a, acc))
            acc = max(prev[i] - 0.5, min(prev[i] + 0.5, acc))  # same 5 m/s^3 jerk limit
            car.acceleration = acc

model = None
if controller == 'RL':
    from stable_baselines3 import TD3
    model = TD3.load(os.path.join(vdir, 'td3_accel_agent.zip'), device='cpu')

def rl_obs(city, v_des):
    if hasattr(Env, 'get_city_obs'):
        return Env.get_city_obs(city, v_des)
    e = object.__new__(Env)
    e.city = city; e.num_rl_cars = len(city.cars) - 1
    e.sim_params = dict(P, v_des=v_des)
    return e._get_obs()

res = {}
for sid, v_des, prof, dur in SCEN:
    p = dict(P, v_des=v_des)
    args = [p[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]
    if controller == 'RL':
        city = RLCity(); city.init(*args, dt=p['dt'], model='ACC')
        for i, c in enumerate(city.cars):
            c.is_rl = i > 0; c.rl_action = 0.0
    elif controller == 'IDM':
        city = IDMCity(); city.init(*args, dt=p['dt'], model='ACC')
    else:
        city = City(); city.init(*args, dt=p['dt'], model=controller)
    if prof:
        city.lead_velocity_profile = prof

    # count physical overlaps BEFORE the bounce-back restitution hides them
    L = city.roads[0].length
    ncoll = [0]
    orig_hc = city.handle_collisions
    def hc(orig=orig_hc, city=city):
        s = sorted(city.cars, key=lambda c: c.pos)
        for i, c in enumerate(s):
            n = s[(i + 1) % len(s)]
            if n is not c and ((n.pos - c.pos) % L) <= c.length:
                ncoll[0] += 1
        orig()
    city.handle_collisions = hc

    steps = int(round(dur / p['dt']))
    N = len(city.cars)
    V = np.zeros((steps + 1, N)); A = np.zeros((steps + 1, N)); G = np.zeros((steps + 1, N)); CMD = np.zeros((steps + 1, N))
    def snap(k):
        for i, c in enumerate(city.cars):
            V[k, i] = c.velocity; A[k, i] = c.acceleration
            G[k, i] = np.nan if i == 0 else (c.pos - city.cars[i - 1].pos - city.cars[i - 1].length) % L
    snap(0)
    for k in range(1, steps + 1):
        if controller == 'RL':
            obs = rl_obs(city, v_des)
            act, _ = model.predict(obs, deterministic=True)
            act = np.asarray(act).reshape(len(city.cars) - 1, -1)
            for i, c in enumerate(city.cars[1:]):
                c.rl_action = float(act[i][0])
                CMD[k, i + 1] = float(act[i][0])
        city.run(p['dt'])
        snap(k)
    res[sid + '__v'] = V; res[sid + '__a'] = A; res[sid + '__g'] = G; res[sid + '__cmd'] = CMD
    res[sid + '__coll'] = np.array(ncoll[0]); res[sid + '__vdes'] = np.array(v_des)
    res[sid + '__e'] = np.array(sum(c.energy_used for c in city.cars))
np.savez_compressed(out, **res)
print('ok', controller, vdir)
