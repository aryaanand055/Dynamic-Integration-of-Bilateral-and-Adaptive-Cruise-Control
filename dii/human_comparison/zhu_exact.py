"""
Zhu et al. (2020), Transportation Research Part C 117, 102662 - Section 6 evaluation reproduced on the
authors' own NGSIM test set (testSet.mat from github.com/MeixinZhu/Velocity_control: 403 events,
columns = [spacing gap, follower speed, relative speed (lead - follower), lead speed], 0.1 s).

Metric definitions copied from the authors' simulation_env.py:
  TTC = -gap / relSpd (shown for 0 < TTC <= 50 s, Fig. 4), headway = gap / follower speed (Fig. 5),
  jerk = (a_t - a_{t-1}) / 0.1 (Fig. 6, mean of |jerk|). Human values use the same formulas on the data,
  with a = diff(speed)/0.1. No extra smoothing on either side.
Each repo controller replaces the human follower: the lead replays the recorded lead speed and the
follower starts at the recorded gap and speed (Sec. 5.2). Our ring-road sim adds a car behind the
follower (needed for BCC). The paper's DDPG and MPC-ACC numbers are quoted from Sec. 6 (their model
weights were not supplied).
"""
import os, sys, json
import numpy as np, scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import wosac_index as WX
import rock_index as RI
import zhang_scenarios as Z
CTRLS = [c for c in Z.CONTROLLERS if c != 'IDM']   # IDM dropped: compared against real drivers only
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEST = os.path.join(os.environ.get('ZHU_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zhu2020_data')), 'testSet.mat')
DT = 0.1


def metrics(gap, v, vl, a):
    rel = vl - v
    with np.errstate(divide='ignore', invalid='ignore'):
        ttc = -gap / rel
        hw = gap / v
    j = np.abs(np.diff(a) / DT)
    pos = ttc[(ttc > 0) & np.isfinite(ttc)]
    return {'ttc': ttc[(ttc > 0) & (ttc <= 50)], 'headway': hw[np.isfinite(hw) & (v > 0.1)], 'jerk': j,
            'min_ttc': float(pos.min()) if len(pos) else np.inf}


def rollout(ctrl, e, model):
    s = {'gap': e[:, 0], 'v': e[:, 1], 'vl': e[:, 3]}
    orig = Z._gaps; v, g, vl, a = [], [], [], []
    def spy(city, car):
        r = orig(city, car)
        if car is city.cars[1]:
            v.append(car.velocity); g.append(r[0]); vl.append(r[1]); a.append(car.acceleration)
        return r
    Z._gaps = spy
    try:
        _, coll = WX.rollout(ctrl, s, model)
    finally:
        Z._gaps = orig
    return np.array(g), np.array(v), np.array(vl), np.array(a), coll


if __name__ == '__main__':
    ev = [x for x in sio.loadmat(TEST)['validationData'][:, 0]]
    model = RI.np_td3.NumpyTD3(os.path.join(Z.REPO, 'td3_accel_agent.zip'))
    per = {'Human (NGSIM)': [metrics(e[:, 0], e[:, 1], e[:, 3], np.r_[0.0, np.diff(e[:, 1]) / DT][1:]) for e in ev]}
    coll = {'Human (NGSIM)': 0}
    for ctrl in CTRLS:
        out, nc = [], 0
        for e in ev:
            g, v, vl, a, c = rollout(ctrl, e, model); nc += c
            out.append(metrics(g, v, vl, a))
        per[ctrl] = out; coll[ctrl] = nc
    P = {k: {m: np.concatenate([x[m] for x in L]) for m in ('ttc', 'headway', 'jerk')} for k, L in per.items()}
    res = {'n_events': len(ev), 'paper_reported': {'DDPG': {'headway_mean': 1.24, 'jerk_mean_abs': 0.63},
                                                  'MPC-ACC': {'headway_mean': 1.23, 'jerk_mean_abs': 0.70},
                                                  'Human': {'headway_mean': 1.61, 'jerk_mean_abs': 1.73}},
           'controllers': {}}
    for k, d in P.items():
        res['controllers'][k] = {
            'ttc_share_below_5s_of_0to50': float(np.mean(d['ttc'] < 5)),
            'ttc_share_below_10s_of_0to50': float(np.mean(d['ttc'] < 10)),
            'events_min_ttc_below_5s': float(np.mean([x['min_ttc'] < 5 for x in per[k]])),
            'headway_mean': float(np.mean(d['headway'])), 'headway_median': float(np.median(d['headway'])),
            'headway_p5': float(np.percentile(d['headway'], 5)), 'headway_p95': float(np.percentile(d['headway'], 95)),
            'headway_share_below_1s': float(np.mean(d['headway'] < 1)), 'headway_share_above_3s': float(np.mean(d['headway'] > 3)),
            'jerk_mean_abs': float(np.mean(d['jerk'])), 'jerk_p95_abs': float(np.percentile(d['jerk'], 95)),
            'collisions': coll[k]}
        print(f"{k:14s}", {a: round(b, 3) for a, b in res['controllers'][k].items()})
    json.dump(res, open(os.path.join(HERE, 'zhu_exact_results.json'), 'w'), indent=1)

    col = {'Human (NGSIM)': '#111111', 'ACC': '#2563eb', 'BCC': '#16a34a', 'ACC+BCC': '#9333ea', 'TD3_56': '#dc2626'}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, d in P.items():
        lw = 2.8 if k.startswith('Human') else 1.6
        ls = '--' if k in ('IDM', 'ACC+BCC') else '-'
        for i, (m, lim) in enumerate([('ttc', (0, 50)), ('headway', (0, 6)), ('jerk', (0, 6))]):
            x = np.sort(d[m]); y = np.arange(1, len(x) + 1) / len(x)
            ax[i].plot(x, y, color=col[k], lw=lw, ls=ls, label=k); ax[i].set_xlim(*lim)
    ax[0].set_xlabel('TTC (s)'); ax[1].set_xlabel('Time headway (s)'); ax[2].set_xlabel('|Jerk| (m/s$^3$)')
    ax[0].set_ylabel('Cumulative probability')
    for a_, t in zip(ax, ['(a) Safety: TTC (0-50 s)', '(b) Efficiency: time headway', '(c) Comfort: jerk']):
        a_.set_title(t, fontsize=11); a_.grid(alpha=.3)
    ax[2].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'Zhu et al. (2020) evaluation, their NGSIM I-80 test set ({len(ev)} events)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'zhu_exact_cdfs.png'), dpi=160)
