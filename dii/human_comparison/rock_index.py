"""
Rock et al. (2023) human-likeness score, adapted to longitudinal car-following, with NGSIM I-80 as the
real-driver reference. Appl. Sci. 13(18), 10218, doi:10.3390/app131810218.

From the paper (used as printed):
  * 13 checks and their weights (Table 4, Spearman |rho| converted to weights, sum = 1)
  * fine-tuned thresholds (Table 3, right column)
  * score = sum(w_i * pass_i) / sum(w_i) * 100 (Eq. 1; the printed '+' is read as a product)
  * checks that cannot be computed in a situation count as pass (Sec. 3.4)
  * KS two-sample test against real drivers in the same context (Sec. 3.4); context = vehicle state
    (Acc / Dcc / Steady / Stop, Table 2 row 3), plus a speed bin, dropped when no reference exists (Sec. 3.3)

Our reading where the paper is ambiguous (stated in the write-up):
  * a Table 3 entry is the FAIL condition: KS statistic > thr fails; ratio < thr fails; TET > thr fails
  * extreme ratio = controller extreme / matched-human extreme, in %
  * "max critical time gap" (gap acceptance at intersections) -> max time headway while following (v > 3 m/s)
  * TET uses TTC < 3 s (as in the project DII)
  * lateral checks (KS lat vel, KS lat acc, max lat vel) and PET do not exist in single-lane following ->
    pass, per the paper's rule. A second score drops them and renormalises the weights.
Run: python rock_index.py  -> rock_results.json (also recomputes the project DII on the same rollouts)
"""
import os, sys, json, inspect
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import np_td3
import zhang_scenarios as Z           # sets repo path; FilteredRLCity, FixedIDMCity
REPO = Z.REPO; os.chdir(REPO)
import evaluation.driver_intuitiveness as di
from evaluation.evaluate_all_scenarios import get_scenarios
from simulation.city import City
from simulation.rl_city import RLCity
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); import dii as D

NGSIM = os.environ.get('NGSIM_CSV', '')
DT = 0.1
W = {'ks_lon_v': 0.016269, 'ks_lat_v': 0.03951, 'ks_lon_a': 0.068174, 'ks_lat_a': 0.004648, 'ks_jerk': 0.114656,
     'max_lon_a': 0.094514, 'min_lon_a': 0.112346, 'max_lon_v': 0.026876, 'max_lat_v': 0.068396,
     'min_dist': 0.080333, 'crit_gap': 0.208539, 'tet': 0.115409, 'pet': 0.050329}
THR = {'ks_lon_v': 0.668406, 'ks_lon_a': 0.559198, 'ks_jerk': 0.511960, 'max_lon_v': 66.67, 'max_lon_a': 71.80,
       'min_lon_a': 72.00, 'min_dist': 93.60, 'crit_gap': 5.40, 'tet': 3.90}
NOT_APPLICABLE = ['ks_lat_v', 'ks_lat_a', 'max_lat_v', 'pet']
SPEED_BINS = [0, 5, 10, 15, 20, 99]


def smooth(x, n=10):
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - n // 2 - 1), mode='edge')
    return np.convolve(pad, k, mode='valid')


def kin(v):
    """Same pipeline for humans and controllers: 1 s moving average on speed, then differentiate."""
    vs = smooth(v)
    a = smooth(np.gradient(vs, DT))
    j = np.gradient(a, DT)
    return vs, a, j


def context(v, a):
    st = np.where(v < 0.5, 3, np.where(a > 0.3, 0, np.where(a < -0.3, 1, 2)))   # Acc, Dcc, Steady, Stop
    sb = np.digitize(v, SPEED_BINS) - 1
    return st, sb


def build_reference():
    d = pd.read_csv(NGSIM).drop_duplicates(['vehicle_id', 'frame_id'])
    d = d[(d.v_class == 2) & d.lane_id.between(1, 6) & (d.preceding > 0) & (d.space_headway > 0)]
    R = {k: [] for k in ['v', 'a', 'j', 'gap']}
    nseg = 0
    for vid, g in d.sort_values('frame_id').groupby('vehicle_id'):
        f = g.frame_id.values
        cut = np.where(np.diff(f) > 10)[0] + 1        # the upload is a row sample: bridge gaps up to 1 s
        for idx in np.split(np.arange(len(f)), cut):
            if len(idx) < 2 or f[idx[-1]] - f[idx[0]] < 150:          # >= 15 s of following
                continue
            fr = np.arange(f[idx[0]], f[idx[-1]] + 1)
            v = np.interp(fr, f[idx], g.v_vel.values[idx]) * 0.3048
            gap = np.interp(fr, f[idx], g.space_headway.values[idx]) * 0.3048 - 4.5
            vs, a, j = kin(v)
            s = slice(10, -10)                                           # drop filter edges
            R['v'].append(vs[s]); R['a'].append(a[s]); R['j'].append(j[s]); R['gap'].append(gap[s]); nseg += 1
    R = {k: np.concatenate(v) for k, v in R.items()}
    R['st'], R['sb'] = context(R['v'], R['a'])
    R['n_segments'] = nseg
    return R


def matched_ref(R, st, sb, rng, n_max=4000):
    """Real samples drawn in the same (state, speed bin) mix as the controller trace (Sec. 3.3 abstraction)."""
    picks = []
    cells, counts = np.unique(np.stack([st, sb]), axis=1, return_counts=True)
    for (s, b), c in zip(cells.T, counts):
        pool = np.where((R['st'] == s) & (R['sb'] == b))[0]
        if len(pool) < 50:
            pool = np.where(R['st'] == s)[0]                             # drop speed context
        if len(pool) == 0:
            continue
        k = max(1, int(round(n_max * c / len(st))))
        picks.append(rng.choice(pool, k, replace=True))
    return np.concatenate(picks)


def rock_checks(R, v, gap, vf, rng):
    vs, a, j = kin(v)
    s = slice(10, len(v) - 10)
    vs, a, j, g, vf_ = vs[s], a[s], j[s], gap[s], vf[s]
    st, sb = context(vs, a)
    ref = matched_ref(R, st, sb, rng)
    rv, ra, rj, rg = R['v'][ref], R['a'][ref], R['j'][ref], R['gap'][ref]
    m = {}
    m['ks_lon_v'] = ks_2samp(vs, rv).statistic
    m['ks_lon_a'] = ks_2samp(a, ra).statistic
    m['ks_jerk'] = ks_2samp(j, rj).statistic
    # NGSIM extremes are noisy (Montanino & Punzo 2015), so human extremes are the 1st / 99th percentiles
    m['max_lon_v'] = 100 * vs.max() / max(np.percentile(rv, 99), 1e-6)
    m['max_lon_a'] = 100 * max(a.max(), 0) / max(np.percentile(ra, 99), 1e-6)
    m['min_lon_a'] = 100 * min(a.min(), 0) / min(np.percentile(ra, 1), -1e-6)
    m['min_dist'] = 100 * g.min() / max(np.percentile(rg[rg > 0], 1), 0.5)
    mov = vs > 3.0
    m['crit_gap'] = float((g[mov] / vs[mov]).max()) if mov.any() else None
    closing = vs - vf_
    ttc = np.where(closing > 0.1, g / np.maximum(closing, 1e-6), np.inf)
    m['tet'] = float((ttc < 3.0).sum() * DT)
    fail = {'ks_lon_v': m['ks_lon_v'] > THR['ks_lon_v'], 'ks_lon_a': m['ks_lon_a'] > THR['ks_lon_a'],
            'ks_jerk': m['ks_jerk'] > THR['ks_jerk'], 'max_lon_v': m['max_lon_v'] < THR['max_lon_v'],
            'max_lon_a': m['max_lon_a'] < THR['max_lon_a'], 'min_lon_a': m['min_lon_a'] < THR['min_lon_a'],
            'min_dist': m['min_dist'] < THR['min_dist'],
            'crit_gap': (m['crit_gap'] is not None) and m['crit_gap'] > THR['crit_gap'], 'tet': m['tet'] > THR['tet']}
    passed = {k: (not fail[k]) for k in fail}
    passed.update({k: True for k in NOT_APPLICABLE})
    score = 100 * sum(W[k] * passed[k] for k in W) / sum(W.values())
    app = [k for k in W if k not in NOT_APPLICABLE]
    score_app = 100 * sum(W[k] * passed[k] for k in app) / sum(W[k] for k in app)
    return score, score_app, passed, {k: (None if v is None else float(v)) for k, v in m.items()}


# ---------------- rollouts of the 11 repo scenarios (same fixes as dii/new_scenarios/run_new_scenarios.py)
orig_dd = City.driver_decision
def dd(self):
    orig_dd(self)
    self._human_acc = {i: self.cars[i].acceleration for i in getattr(self, 'car_velocity_profiles', {}) or {}}
City.driver_decision = dd
orig_idm = di.IDMCity.driver_decision
def idm_dd(self):
    orig_idm(self)
    for i, a in getattr(self, '_human_acc', {}).items(): self.cars[i].acceleration = a
di.IDMCity.driver_decision = idm_dd
orig_rl = Z.FilteredRLCity.driver_decision
def rl_dd(self):
    for i in (getattr(self, 'car_velocity_profiles', {}) or {}): self.cars[i].is_rl = False
    orig_rl(self)
Z.FilteredRLCity.driver_decision = rl_dd
di.RLCity = Z.FilteredRLCity                                         # TD3_56 with its 0.35 actuator filter

CAP = []
orig_cdm = di.compute_dii_metrics
def cdm(followers, leader, **kw):
    CAP.append(dict(followers=followers, leader=leader, collisions=kw['collisions']))
    return orig_cdm(followers, leader, **kw)
di.compute_dii_metrics = cdm

src = inspect.getsource(D.scenario).replace("vf = V[:, :-1]", "vf = d[s + '__vf']")
ns = dict(D.__dict__); exec(src, ns); dii_score = ns['scenario']

NAMES = ['ACC', 'BCC', 'ACC+BCC', 'TD3_56', 'IDM', 'IDM_default']

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    R = build_reference()
    print('NGSIM reference:', R['n_segments'], 'following segments,', len(R['v']), 'samples',
          f"v {R['v'].min():.1f}-{R['v'].max():.1f} m/s, a {R['a'].min():.2f}..{R['a'].max():.2f}", flush=True)
    model = np_td3.NumpyTD3(os.path.join(REPO, 'td3_accel_agent.zip'))
    out = {'reference': {'segments': int(R['n_segments']), 'samples': int(len(R['v'])),
                         'v_range': [float(R['v'].min()), float(R['v'].max())],
                         'a_p1_p99': [float(np.percentile(R['a'], 1)), float(np.percentile(R['a'], 99))],
                         'rms_jerk': float(np.sqrt(np.mean(R['j'] ** 2)))},
           'scenarios': {}}
    for sc in get_scenarios():
        CAP.clear(); di.simulate_benchmark_scenario(sc, model)
        res = {}
        for nm, rec in zip(NAMES, CAP):
            fl = rec['followers']
            if sc.get('car_velocity_profiles'):
                fl = [f for k, f in enumerate(fl) if (k + 1) not in sc['car_velocity_profiles']]
            rows = [rock_checks(R, np.array(f['v']), np.array(f['gap']), np.array(f['vf']), rng) for f in fl]
            A = np.column_stack([rec['leader']['a']] + [f['a'] for f in fl])
            V = np.column_stack([rec['leader']['v']] + [f['v'] for f in fl])
            G = np.column_stack([np.zeros(len(V))] + [f['gap'] for f in fl])
            VF = np.column_stack([f['vf'] for f in fl]); BG = np.column_stack([f['bg'] for f in fl])
            coll = int(G[:, 1:].min() < 0.5 or BG.min() < 0.5)
            dres = dii_score({'x__a': A, 'x__v': V, 'x__g': G, 'x__vf': VF, 'x__coll': coll}, 'x')
            res[nm] = {'rock': float(np.mean([r[0] for r in rows])), 'rock_applicable': float(np.mean([r[1] for r in rows])),
                       'pass_rate': {k: float(np.mean([r[2][k] for r in rows])) for k in W},
                       'metrics_mean': {k: (None if any(r[3][k] is None for r in rows) else float(np.mean([r[3][k] for r in rows]))) for k in rows[0][3]},
                       'dii_weighted': float(dres['DII']), 'collision': coll}
        out['scenarios'][sc['id']] = res
        print(sc['id'], ' | '.join(f"{n}: R{res[n]['rock']:.1f}/{res[n]['rock_applicable']:.1f} DII{res[n]['dii_weighted']:.1f}{' C' if res[n]['collision'] else ''}" for n in NAMES), flush=True)
    out['overall'] = {n: {k: float(np.mean([out['scenarios'][s][n][k] for s in out['scenarios']])) for k in ['rock', 'rock_applicable', 'dii_weighted']} for n in NAMES}
    for n in NAMES: print(n, out['overall'][n])
    json.dump(out, open(os.path.join(HERE, 'rock_results.json'), 'w'), indent=1)
