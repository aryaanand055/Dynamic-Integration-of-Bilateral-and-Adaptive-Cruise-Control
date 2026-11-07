"""Driver Intuitiveness Index (DII) from replayed trajectories (out_*.npz)."""
import numpy as np, json, sys

DT = 0.1
SCEN = ['steady_15', 'steady_25', 'stop_and_go', 'speed_drop', 'full_stop_hold', 'emergency_brake', 'sine_wave']
DYN = SCEN[2:]
W = dict(S=0.30, J=0.40 * 0.20 / 0.60, A=0.10, H=0.10, O=0.40 * 0.10 / 0.60, Hm=0.30)
IDM = dict(a=1.4, b=2.0, delta=4.0, T=0.8, s0=6.0, v0=30.0)

def lin(x, good, bad):
    """100 at `good`, 0 at `bad`, linear in between (works for either direction)."""
    return float(100 * np.clip((x - bad) / (good - bad), 0, 1))

def idm_acc(v, dv, s):
    """Improved IDM (Treiber & Kesting 2013) acceleration of a human driver in state (v, closing speed, gap)."""
    s_star = IDM['s0'] + np.maximum(0, v * IDM['T'] + v * dv / (2 * np.sqrt(IDM['a'] * IDM['b'])))
    z = s_star / np.maximum(s, 0.1)
    a_free = IDM['a'] * (1 - (v / IDM['v0']) ** IDM['delta'])
    with np.errstate(over='ignore', invalid='ignore'):
        free = a_free * (1 - z ** (2 * IDM['a'] / np.maximum(a_free, 1e-6)))
    return np.where(z >= 1, IDM['a'] * (1 - z ** 2), free)

def scenario(d, s):
    V, A, G = d[s + '__v'], d[s + '__a'], d[s + '__g']
    coll = int(d[s + '__coll'])
    vL, aL = V[:, 0], A[:, 0]
    v, a, g = V[:, 1:], A[:, 1:], G[:, 1:]
    vf = V[:, :-1]                      # speed of each follower's predecessor
    m = {}
    # ---- S: safety margin -------------------------------------------------
    closing = v - vf
    ttc = np.where(closing > 0.1, g / np.maximum(closing, 1e-6), np.inf)
    tet = (ttc < 3.0).sum(0).mean() * DT                     # time-exposed TTC, s per vehicle
    dur_min = len(V) * DT / 60
    m['TET_s_per_min'] = tet / dur_min
    m['min_TTC_s'] = float(np.min(ttc))
    m['min_gap_m'] = float(np.min(g))
    m['collisions'] = coll
    s_tet = lin(m['TET_s_per_min'], 0.0, 6.0)               # 0 s/min -> 100, >=6 s/min (10% of time) -> 0
    s_gap = lin(m['min_gap_m'], 6.0, 1.0)                    # >= min_dis (6 m) -> 100, <=1 m -> 0
    S = 0 if coll else 0.5 * s_tet + 0.5 * s_gap
    # ---- J: jerk ----------------------------------------------------------
    jerk = np.diff(a, axis=0) / DT
    m['rms_jerk'] = float(np.sqrt(np.mean(jerk ** 2)))
    m['p95_abs_jerk'] = float(np.percentile(np.abs(jerk), 95))
    J = 0.5 * lin(m['rms_jerk'], 0.3, 2.0) + 0.5 * lin(m['p95_abs_jerk'], 0.9, 4.0)
    # ---- A: acceleration comfort -------------------------------------------
    m['rms_acc'] = float(np.sqrt(np.mean(a ** 2)))
    m['frac_outside_band'] = float(np.mean((a > 2.0) | (a < -3.0)))
    A_ = 0.5 * lin(m['rms_acc'], 0.3, 1.5) + 0.5 * lin(m['frac_outside_band'], 0.0, 0.05)
    # ---- H: time-gap consistency -------------------------------------------
    mov = v > 3.0
    thw = np.where(mov, g / np.maximum(v, 1e-6), np.nan)
    cvs = []
    for i in range(thw.shape[1]):
        x = thw[:, i][~np.isnan(thw[:, i])]
        if len(x) > 10: cvs.append(np.std(x) / np.mean(x))
    m['thw_cv'] = float(np.mean(cvs)) if cvs else 0.0
    vals = thw[~np.isnan(thw)]
    m['thw_in_band'] = float(np.mean((vals >= 0.8) & (vals <= 3.0))) if len(vals) else 1.0
    m['thw_median'] = float(np.median(vals)) if len(vals) else float('nan')
    H = 0.5 * lin(m['thw_cv'], 0.1, 0.6) + 0.5 * lin(m['thw_in_band'], 1.0, 0.5)
    # ---- O: oscillation & control effort -------------------------------------
    sgn = np.sign(np.where(np.abs(a) > 0.1, a, 0))
    rev = 0
    for i in range(sgn.shape[1]):
        x = sgn[:, i][sgn[:, i] != 0]
        rev += np.sum(x[1:] != x[:-1])
    m['reversals_per_min'] = rev / sgn.shape[1] / dur_min
    lmax = max(0.0, -aL.min())
    m['shockwave_ratio'] = float(max(0.0, -a.min()) / lmax) if lmax > 0.1 else float('nan')
    effL = np.mean(np.abs(aL))
    m['effort_ratio'] = float(np.mean(np.abs(a)) / effL) if effL > 0.05 else float('nan')
    o_rev = lin(m['reversals_per_min'], 2.0, 20.0)
    o_str = 100.0 if np.isnan(m['shockwave_ratio']) else lin(m['shockwave_ratio'], 1.0, 2.0)
    o_eff = lin(np.mean(np.abs(a)), 0.05, 0.5) if np.isnan(m['effort_ratio']) else lin(m['effort_ratio'], 1.0, 2.5)
    O = (o_rev + o_str + o_eff) / 3
    # ---- Hm: similarity to a human (IDM) driver in the same state ------------
    a_h = np.clip(idm_acc(v, closing, g), -5.0, 3.0)
    a_h = np.where((v < 0.05) & (a_h < 0), 0.0, a_h)       # a stopped human just holds the brake
    m['rmse_vs_human'] = float(np.sqrt(np.mean((a - a_h) ** 2)))
    Hm = lin(m['rmse_vs_human'], 0.2, 1.5)
    comp = dict(S=S, J=J, A=A_, H=H, O=O, Hm=Hm)
    dii = sum(W[k] * comp[k] for k in W)
    if coll: dii *= 0.5
    return dict(DII=dii, comp=comp, metrics=m)

def evaluate_random(path):
    d = np.load(path)
    ids = sorted({k.split('__')[0] for k in d.files})
    per = {s: scenario(d, s) for s in ids}
    return dict(per=per, DII=float(np.mean([per[s]['DII'] for s in ids])),
                comp={k: float(np.mean([per[s]['comp'][k] for s in ids])) for k in W},
                collisions_total=int(sum(per[s]['metrics']['collisions'] for s in ids)),
                n_coll_profiles=int(sum(per[s]['metrics']['collisions'] > 0 for s in ids)))

def evaluate(path):
    d = np.load(path)
    per = {s: scenario(d, s) for s in SCEN}
    out = dict(per=per)
    out['DII_dynamic'] = float(np.mean([per[s]['DII'] for s in DYN]))
    out['DII_all'] = float(np.mean([per[s]['DII'] for s in SCEN]))
    out['comp_dynamic'] = {k: float(np.mean([per[s]['comp'][k] for s in DYN])) for k in W}
    out['collisions_total'] = int(sum(per[s]['metrics']['collisions'] for s in SCEN))
    return out

if __name__ == '__main__':
    import os
    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])   # folder holding out_*.npz / rnd_*.npz
    labels = dict(ACC='ACC', BCC='BCC only', ACCBCC='ACC+BCC', TD3_56='TD3 (TD3_56, current)', IDM='Human ref (IDM)',
                  TD3_P1='TD3 early (Jun 2026)', TD3_25='TD3_25', TD3_39='TD3_39', TD3_41='TD3_41', TD3_49='TD3_49')
    R = {}
    for k in labels:
        R[k] = evaluate(f'out_{k}.npz')
        R[k]['random'] = evaluate_random(f'rnd_{k}.npz')
    # weight sensitivity: 2000 random weight vectors (Dirichlet(1)) -> how often each controller ranks first
    rng = np.random.default_rng(0)
    main = ['ACC', 'BCC', 'ACCBCC', 'TD3_56']
    keys = list(W)
    def dii_with(k, w, which):
        per = R[k]['per'] if which == 'bench' else R[k]['random']['per']
        ss = DYN if which == 'bench' else list(per)
        vals = []
        for s_ in ss:
            v = sum(w[i] * per[s_]['comp'][c] for i, c in enumerate(keys))
            vals.append(v * (0.5 if per[s_]['metrics']['collisions'] else 1))
        return np.mean(vals)
    sens = {}
    for which in ['bench', 'random']:
        wins = {k: 0 for k in main}; ranks = {k: [] for k in main}
        for _ in range(2000):
            w = rng.dirichlet(np.ones(len(keys)))
            sc = {k: dii_with(k, w, which) for k in main}
            order = sorted(main, key=lambda k: -sc[k])
            wins[order[0]] += 1
            for r_, k in enumerate(order): ranks[k].append(r_ + 1)
        eq = {k: dii_with(k, np.ones(len(keys)) / len(keys), which) for k in main}
        sens[which] = dict(first_place_share={k: wins[k] / 2000 for k in main},
                           mean_rank={k: float(np.mean(ranks[k])) for k in main}, equal_weights=eq)
    # bootstrap over the 20 random profiles: TD3_56 minus each classical controller
    boot = {}
    ids = list(R['TD3_56']['random']['per'])
    for k in ['ACC', 'BCC', 'ACCBCC', 'TD3_49']:
        diff = np.array([R['TD3_56']['random']['per'][i]['DII'] - R[k]['random']['per'][i]['DII'] for i in ids])
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        boot[k] = dict(mean=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                       td3_better_in=int((diff > 0).sum()), n=len(diff))
    json.dump(dict(R=R, sens=sens, boot=boot, W=W, labels=labels), open('dii_results.json', 'w'), indent=1, default=float)
    print(f"{'controller':24s} DIIdyn DIIall DIIrnd  " + '  '.join(f'{k:>4s}' for k in W) + '  coll  rnd-coll(profiles)')
    for k, lab in labels.items():
        r = R[k]
        print(f"{lab:24s} {r['DII_dynamic']:6.1f} {r['DII_all']:6.1f} {r['random']['DII']:6.1f}  " + '  '.join(f"{r['comp_dynamic'][c]:4.0f}" for c in W)
              + f"  {r['collisions_total']:4d}  {r['random']['collisions_total']}({r['random']['n_coll_profiles']})")
        print(' ' * 47 + '  '.join(f"{r['random']['comp'][c]:4.0f}" for c in W) + '   <- random-profile components')
    print(json.dumps(sens, indent=1)); print(json.dumps(boot, indent=1))
