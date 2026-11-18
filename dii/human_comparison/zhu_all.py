"""
Zhu et al. (2020) comparison on ALL 1,341 NGSIM I-80 events (authors' trainSet.mat 938 + testSet.mat 403).
Our controllers never trained on NGSIM, so every event is test data. Same metrics as zhu_exact.py
(better-than-human, Zhu 2020) and zhu_humanlike.py (closeness to the real driver, Zhu 2018 RMSPE + KS).
Outputs: zhu_all_results.json, zhu_all_cdfs.png.
"""
import os, sys, json
import numpy as np, scipy.io as sio
from scipy.stats import ks_2samp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import zhu_exact as ZE
from zhu_humanlike import rmspe
import rock_index as RI
import zhang_scenarios as Z
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.environ.get('ZHU_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zhu2020_data')) + '/'
M = ('ttc', 'headway', 'jerk')

if __name__ == '__main__':
    ev = list(sio.loadmat(D + 'testSet.mat')['validationData'][:, 0]) + list(sio.loadmat(D + 'trainSet.mat')['calibrationData'][:, 0])
    model = RI.np_td3.NumpyTD3(os.path.join(Z.REPO, 'td3_accel_agent.zip'))
    per = {'Human (NGSIM)': [ZE.metrics(e[:, 0], e[:, 1], e[:, 3], np.diff(e[:, 1]) / ZE.DT) for e in ev]}
    extra = {'Human (NGSIM)': {'collisions': 0, 'near_crash_0.5m': int(sum(e[:, 0].min() < 0.5 for e in ev))}}
    for c in ZE.CTRLS:
        out, rs, rv, nc, nn = [], [], [], 0, 0
        for e in ev:
            g, v, vl, a, col = ZE.rollout(c, e, model)
            out.append(ZE.metrics(g, v, vl, a))
            n = min(len(g), len(e) - 1)
            rs.append(rmspe(g[:n], e[1:n + 1, 0])); rv.append(rmspe(v[:n], e[1:n + 1, 1]))
            nc += int(g.min() < 0); nn += col
        per[c] = out
        extra[c] = {'collisions': nc, 'near_crash_0.5m': nn, 'rmspe_spacing_mean': float(np.mean(rs)), 'rmspe_speed_mean': float(np.mean(rv))}
    P = {k: {m: np.concatenate([x[m] for x in L]) for m in M} for k, L in per.items()}
    H = P['Human (NGSIM)']
    res = {'n_events': len(ev), 'controllers': {}}
    for k, d in P.items():
        r = {'events_min_ttc_below_5s': float(np.mean([x['min_ttc'] < 5 for x in per[k]])),
             'headway_mean': float(np.mean(d['headway'])), 'headway_p5': float(np.percentile(d['headway'], 5)),
             'headway_p95': float(np.percentile(d['headway'], 95)), 'headway_share_below_1s': float(np.mean(d['headway'] < 1)),
             'jerk_mean_abs': float(np.mean(d['jerk'])), **extra[k]}
        if k != 'Human (NGSIM)':
            for m in M: r[f'ks_{m}'] = float(ks_2samp(d[m], H[m]).statistic)
            r['ks_mean'] = float(np.mean([r[f'ks_{m}'] for m in M]))
        res['controllers'][k] = r
        print(f"{k:14s}", {a: round(b, 3) for a, b in r.items()})
    json.dump(res, open(os.path.join(HERE, 'zhu_all_results.json'), 'w'), indent=1)

    col = {'Human (NGSIM)': '#111111', 'ACC': '#2563eb', 'BCC': '#16a34a', 'ACC+BCC': '#9333ea', 'TD3_56': '#dc2626'}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, d in P.items():
        for i, (m, lim) in enumerate([('ttc', (0, 50)), ('headway', (0, 6)), ('jerk', (0, 6))]):
            x = np.sort(d[m]); y = np.arange(1, len(x) + 1) / len(x)
            ax[i].plot(x, y, color=col[k], lw=2.8 if k.startswith('Human') else 1.6,
                       ls='--' if k == 'ACC+BCC' else '-', label=k); ax[i].set_xlim(*lim)
    ax[0].set_xlabel('TTC (s)'); ax[1].set_xlabel('Time headway (s)'); ax[2].set_xlabel('|Jerk| (m/s$^3$)')
    ax[0].set_ylabel('Cumulative probability')
    for a_, t in zip(ax, ['(a) Safety: TTC (0-50 s)', '(b) Efficiency: time headway', '(c) Comfort: jerk']):
        a_.set_title(t, fontsize=11); a_.grid(alpha=.3)
    ax[2].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'Zhu et al. (2020) evaluation on all {len(ev):,} NGSIM I-80 car-following events', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'zhu_all_cdfs.png'), dpi=160)
