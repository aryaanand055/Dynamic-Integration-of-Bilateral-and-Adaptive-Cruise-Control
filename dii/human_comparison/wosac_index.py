"""
WOSAC-style realism score (Montali et al. 2023, "The Waymo Open Sim Agents Challenge",
NeurIPS 2023 Datasets & Benchmarks, arXiv:2305.12032), adapted to single-lane car following.

As printed in the paper:
  - each logged human trajectory is scored by the likelihood of its values under histograms fitted to
    the simulated trajectory (Sec. 4.2), values pooled over time (time-independent, Sec. 4.2.1);
  - per component m = exp(mean_t log p_t) over valid steps (Sec. 4.2.1);
  - Laplace smoothing, pseudocount 0.1 (App. A.6);
  - composite = weighted mean over components and scenarios, weights sum to 1, collision (and
    road departure) weighted 2x the others (Eq. 3).
Adapted (our choices, the paper does not cover them):
  - scenario = one NGSIM I-80 car-following segment; the lead car replays the human lead's speed
    (reconstructed as v_follower + d gap/dt), and the controller starts at the human's speed and gap;
  - components that exist in 1-D following: linear speed, linear acceleration, distance to the car
    ahead, TTC, collision (weights 1,1,1,1,2 -> /6). Angular speed/accel, road edge and off-road
    do not apply;
  - controllers are deterministic, so K = 1 sample instead of 32;
  - bins (20 per feature) and ranges below; TTC capped at 5 s. Waymo's own bin settings live in its
    config files, which we could not read.
"""
import os, sys, json
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rock_index as RI
import zhang_scenarios as Z
from simulation.config import DEFAULT_SIM_PARAMS

DT, PSEUDO, NB = 0.1, 0.1, 20
RANGES = {'speed': (0, 30), 'accel': (-5, 5), 'gap': (0, 80), 'ttc': (0, 5)}
W = {'speed': 1, 'accel': 1, 'gap': 1, 'ttc': 1, 'collision': 2}
W = {k: v / sum(W.values()) for k, v in W.items()}
N_SEG, MAX_LEN = 10**6, 600          # all segments, up to 60 s each


def segments():
    d = pd.read_csv(RI.NGSIM).drop_duplicates(['vehicle_id', 'frame_id'])
    d = d[(d.v_class == 2) & d.lane_id.between(1, 6) & (d.preceding > 0) & (d.space_headway > 0)]
    out = []
    for vid, g in d.sort_values('frame_id').groupby('vehicle_id'):
        f = g.frame_id.values
        cut = np.where((np.diff(f) > 10) | (np.diff(g.preceding.values) != 0))[0] + 1   # same leader throughout
        for idx in np.split(np.arange(len(f)), cut):
            if len(idx) < 2 or f[idx[-1]] - f[idx[0]] < 150:
                continue
            fr = np.arange(f[idx[0]], f[idx[-1]] + 1)[:MAX_LEN]
            v = np.interp(fr, f[idx], g.v_vel.values[idx]) * 0.3048
            gap = np.interp(fr, f[idx], g.space_headway.values[idx]) * 0.3048 - 4.5
            vs = RI.smooth(v); gs = RI.smooth(gap)
            vl = np.clip(RI.smooth(vs + np.gradient(gs, DT)), 0, None)
            out.append({'v': vs, 'gap': gs, 'vl': vl})
    return out


def feats(v, gap, vl):
    vs, a, _ = RI.kin(v)
    close = vs - vl
    ttc = np.where(close > 0, gap / np.maximum(close, 1e-6), np.inf)
    return {'speed': vs, 'accel': a, 'gap': gap, 'ttc': np.minimum(ttc, RANGES['ttc'][1])}


def lik(sim, real, rng):
    edges = np.linspace(*rng, NB + 1)
    cnt, _ = np.histogram(np.clip(sim, rng[0], rng[1] - 1e-9), edges)
    p = (cnt + PSEUDO) / (cnt.sum() + PSEUDO * NB)
    b = np.clip(np.digitize(np.clip(real, rng[0], rng[1] - 1e-9), edges) - 1, 0, NB - 1)
    return float(np.exp(np.mean(np.log(p[b]))))


def score(sim, real, sim_coll, real_coll=False):
    m = {k: lik(sim[k], real[k], RANGES[k]) for k in RANGES}
    c = np.array([float(not sim_coll), float(sim_coll)])               # 2-bin histogram of the sample
    m['collision'] = float((c[int(real_coll)] + PSEUDO) / (c.sum() + 2 * PSEUDO))
    m['composite'] = sum(W[k] * m[k] for k in W)
    return m


def rollout(ctrl, s, model):
    n = len(s['v']); p = dict(DEFAULT_SIM_PARAMS); p['max_v'] = 35.0; p['car_number'] = 3; p['dt'] = DT
    p['v_des'] = max(float(s['v'][0]), 1.0)
    city = Z._make_city(ctrl, p)
    city.v_des = min(p['max_v'], max(1.1 * float(s['vl'].max()), 5.0))
    city.lead_velocity_profile = [(i * DT, float(x)) for i, x in enumerate(s['vl'])] + [(999.0, float(s['vl'][-1]))]
    c0, c1, c2 = city.cars
    c0.velocity, c1.velocity, c2.velocity = float(s['vl'][0]), float(s['v'][0]), float(s['v'][0])
    c1.pos = (c0.pos + c0.length + max(float(s['gap'][0]), 1.0)) % 1000
    c2.pos = (c1.pos + c1.length + 6 + 0.8 * float(s['v'][0])) % 1000
    if ctrl == 'TD3_56':
        for i, c in enumerate(city.cars): c.is_rl = i > 0; c.rl_action = 0.0
    v, g, vf, bg = [], [], [], []
    for _ in range(n):
        if ctrl == 'TD3_56':
            obs = Z.AccelTrafficEnv.get_city_obs(city, city.v_des, obs_dim=model.observation_space.shape[0])
            act, _ = model.predict(obs, deterministic=True)
            for i, c in enumerate(city.cars[1:]):
                if getattr(c, 'is_rl', False): c.rl_action = float(np.ravel(act[i])[0])
        city.run(DT)
        a_, b_, c_ = Z._gaps(city, c1); v.append(c1.velocity); g.append(a_); vf.append(b_); bg.append(c_)
    v, g, vf, bg = map(np.array, (v, g, vf, bg))
    coll = bool(g[5:].min() < 0.5 or bg[5:].min() < 0.5)
    return feats(v, g, vf), coll


if __name__ == '__main__':
    segs = segments()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(segs), min(N_SEG, len(segs)), replace=False)
    segs = [segs[i] for i in pick]
    real = [feats(s['v'], s['gap'], s['vl']) for s in segs]
    model = RI.np_td3.NumpyTD3(os.path.join(Z.REPO, 'td3_accel_agent.zip'))
    keys = list(W) + ['composite']
    out = {'n_segments': len(segs), 'weights': W, 'ranges': RANGES, 'bins': NB, 'controllers': {}}
    # Oracle: the human trajectory scored against itself (the paper's "logged oracle" ceiling)
    orc = [score(r, r, False) for r in real]
    out['controllers']['Human (oracle)'] = {k: float(np.mean([m[k] for m in orc])) for k in keys}
    out['controllers']['Human (oracle)']['collisions'] = 0
    for ctrl in Z.CONTROLLERS:
        ms, nc = [], 0
        for s, r in zip(segs, real):
            sim, coll = rollout(ctrl, s, model); nc += coll
            ms.append(score(sim, r, coll))
        out['controllers'][ctrl] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        out['controllers'][ctrl]['collisions'] = int(nc)
    for c, d in out['controllers'].items():
        print(f"{c:15s} " + '  '.join(f"{k}={d[k]:.3f}" for k in keys) + f"  coll={d['collisions']}/{len(segs)}")
    json.dump(out, open(os.path.join(HERE, 'wosac_results.json'), 'w'), indent=1)
