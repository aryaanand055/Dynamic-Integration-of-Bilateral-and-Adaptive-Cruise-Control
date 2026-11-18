"""
Human-likeness on the Zhu et al. (2020) NGSIM test set (403 events): how CLOSE each controller is to
the human driver who actually drove the same event.

1. Trajectory closeness, per event, as in Zhu, Wang & Wang (2018), "Human-like autonomous car-following
   model with deep reinforcement learning", Transportation Research Part C 97, 348-368,
   doi:10.1016/j.trc.2018.10.024 (Eqs. 5-6): RMSPE of spacing and of speed,
   RMSPE = sqrt( sum (sim - obs)^2 / sum obs^2 ).
2. Behaviour-distribution closeness: two-sample Kolmogorov-Smirnov distance (0 = identical, 1 = no
   overlap) between controller and human distributions of headway, TTC (0-50 s) and |jerk|, using the
   Zhu 2020 metric definitions (zhu_exact.metrics).
"""
import os, sys, json
import numpy as np, scipy.io as sio
from scipy.stats import ks_2samp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import zhu_exact as ZE
import rock_index as RI
import zhang_scenarios as Z
CTRLS = [c for c in Z.CONTROLLERS if c != 'IDM']   # IDM dropped: compared against real drivers only


def rmspe(sim, obs):
    return float(np.sqrt(np.sum((sim - obs) ** 2) / np.sum(obs ** 2)))


if __name__ == '__main__':
    ev = list(sio.loadmat(ZE.TEST)['validationData'][:, 0])
    model = RI.np_td3.NumpyTD3(os.path.join(Z.REPO, 'td3_accel_agent.zip'))
    hum = [ZE.metrics(e[:, 0], e[:, 1], e[:, 3], np.diff(e[:, 1]) / ZE.DT) for e in ev]
    H = {m: np.concatenate([x[m] for x in hum]) for m in ('ttc', 'headway', 'jerk')}
    res = {}
    for c in CTRLS:
        rs, rv, ms = [], [], []
        for e in ev:
            g, v, vl, a, _ = ZE.rollout(c, e, model)
            n = min(len(g), len(e) - 1)
            rs.append(rmspe(g[:n], e[1:n + 1, 0])); rv.append(rmspe(v[:n], e[1:n + 1, 1]))
            ms.append(ZE.metrics(g, v, vl, a))
        S = {m: np.concatenate([x[m] for x in ms]) for m in ('ttc', 'headway', 'jerk')}
        res[c] = {'rmspe_spacing_mean': float(np.mean(rs)), 'rmspe_speed_mean': float(np.mean(rv)),
                  **{f'ks_{m}': float(ks_2samp(S[m], H[m]).statistic) for m in S}}
        res[c]['ks_mean'] = float(np.mean([res[c][f'ks_{m}'] for m in S]))
        print(f"{c:8s}", {k: round(x, 3) for k, x in res[c].items()})
    json.dump(res, open(os.path.join(HERE, 'zhu_humanlike_results.json'), 'w'), indent=1)
