"""
Driver Intuitiveness Metric Suite & IDM Human Reference Evaluation Module.

Quantitatively evaluates platoon follower controllers (ACC, BCC, ACC+BCC, RL/TD3)
and compares them against an Intelligent Driver Model (IDM) human reference across
the official 7 benchmark operational scenarios.

Metrics:
- Driver Intuitiveness Index (DII in [0, 100], based strictly on human car-following behavior):
    * C1: Appropriateness (penalizes phantom braking and missed deceleration)
    * C2: Smoothness (penalizes excessive RMS jerk and high-frequency sign flips)
    * C3: Calmness (rewards acceleration tranquility during quiet following/standstill)
    * C4: Comfort Margin (fraction of moving steps with headway >= 0.8s and TTC >= 4.0s)
  (Note: Platoon string stability / wave ratio is tracked as a raw diagnostic, but excluded from DII)
- IDM Comparison Metrics:
    * Acceleration RMSE and Gap RMSE
    * Wasserstein-1 distance of acceleration distributions
    * Wasserstein-1 distance of time-headway distributions (moving steps)
    * Human Similarity Score: exp(-accel_RMSE / 0.5)
"""

import os
import sys
import math
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from stable_baselines3 import TD3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import (
    DEFAULT_SIM_PARAMS,
    resolve_model_path,
    LOGS_DIR,
    PLOTS_DIR,
)
from env.accel_traffic_env import AccelTrafficEnv
from evaluation.evaluate_all_scenarios import get_scenarios, load_csv_profile

# ==============================================================================
# DESIGN THRESHOLDS FOR DRIVER INTUITIVENESS INDEX (DII)
# NOTE: These thresholds represent design choices for human-like intuitiveness
# and comfort proxies; they are not rigid statutory standards.
# ==============================================================================
DII_THRESHOLDS = {
    # C1: Appropriateness
    'vrel_req_thresh': 0.5,        # Minimum closing speed (m/s) before required deceleration a_req is computed
    'min_gap_headroom': 0.5,       # Buffer floor (m) in denominator max(gap - s0, 0.5) to avoid division by zero
    'phantom_decel_thresh': -1.0,  # Deceleration threshold (a < -1.0 m/s^2) indicating noticeable braking
    'phantom_gap_factor': 1.2,     # Gap headroom multiplier (> 1.2 * desired_gap) indicating clear road ahead
    'phantom_vrel_thresh': 0.0,    # Relative speed condition (vrel <= 0) indicating not closing on the front car
    'rear_decel_thresh': -0.5,     # Threshold for rear-induced deceleration when front is clear
    'missed_areq_thresh': 2.0,     # Urgency threshold (a_req > 2.0 m/s^2) requiring active follower deceleration
    'missed_decel_ratio': 0.5,     # Fraction of a_req (decel < 0.5 * a_req) below which braking is deemed missed
    'rate_penalty_scale': 0.10,    # Combined rate ceiling (10%) at which C1 appropriateness drops to 0.0

    # C2: Smoothness
    'jerk_deadband': 0.1,          # Acceleration deadband (|a| <= 0.1 m/s^2) ignored during sign flip counting
    'rms_jerk_scale': 2.0,         # Exponential decay scale for RMS jerk (m/s^3)
    'sign_changes_scale': 30.0,    # Exponential decay scale for acceleration sign flips per car per minute

    # C3: Calmness
    'quiet_vrel_thresh': 0.5,      # Speed difference tolerance (|vrel| < 0.5 m/s) for steady following
    'quiet_gap_ratio': 0.2,        # Fractional gap tolerance (|gap - desired_gap| < 0.2 * max(desired_gap, 1))
    'quiet_standstill_v': 0.2,     # Standstill speed threshold (v < 0.2 and vf < 0.2 m/s)
    'calm_accel_scale': 0.3,       # Mean absolute acceleration scale (m/s^2) over quiet steps

    # C4: Comfort Margin
    'moving_v_thresh': 1.0,        # Minimum vehicle speed (v > 1.0 m/s) defining moving steps
    'min_headway': 0.8,            # Minimum acceptable time headway (s) under comfortable driving
    'min_ttc': 4.0,                # Minimum acceptable Time-To-Collision (s)
    'ttc_vrel_thresh': 0.1,        # Closing speed floor (vrel > 0.1 m/s) for finite TTC computation

    # C5: Wave Behaviour
    'min_leader_decel': 0.1,       # Floor on leader deceleration (m/s^2) to prevent singularity on steady cruise
}

# ==============================================================================
# INTELLIGENT DRIVER MODEL (IDM) PARAMETERS
# ==============================================================================
IDM_PARAMS = {
    'a_max': 1.5,   # Maximum desired acceleration (m/s^2)
    'b': 2.0,       # Comfortable braking deceleration (m/s^2)
    'T': 1.5,       # Safe time headway (s)
    's0': 2.0,      # Minimum standstill distance (m)
    'delta': 4.0,   # Acceleration exponent
}

def get_gap_matched_idm_params(sim_params=None):
    """
    Returns IDM parameters calibrated to match the simulator's initial equilibrium
    spacing: s0 = min_dis, T = reaction_time.
    Equilibrium gap at speed v equals the ACC desired gap: min_dis + reaction_time * v.
    """
    params = dict(IDM_PARAMS)
    base = DEFAULT_SIM_PARAMS if sim_params is None else sim_params
    params['s0'] = float(base.get('min_dis', 6.0))
    params['T'] = float(base.get('reaction_time', 0.8))
    return params


class IDMCity(City):
    """
    Subclass of City implementing the Intelligent Driver Model (IDM) for follower cars.
    Preserves existing simulation physics and leader profile handling while overwriting
    follower car accelerations with IDM commands under identical kinematic limits.
    """
    def __init__(self, idm_params=None):
        super().__init__()
        self.idm_params = dict(IDM_PARAMS if idm_params is None else idm_params)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        for car in self.cars:
            car._prev_acc = 0.0

    def driver_decision(self):
        # 1. Save previous step accelerations for followers to compute exact jerk
        for car in self.cars[1:]:
            car._prev_acc = getattr(car, 'acceleration', 0.0)

        # 2. Run standard superclass decision (handles leader profile and physics state)
        super().driver_decision()

        # 3. Overwrite follower accelerations with IDM
        dt = self.dt
        road_len = self.roads[0].length if self.roads else 1000.0
        v0 = max(0.1, float(self.v_des))

        a_max = self.idm_params['a_max']
        b = self.idm_params['b']
        T = self.idm_params['T']
        s0 = self.idm_params['s0']
        delta = self.idm_params.get('delta', 4.0)
        denom_ab = 2.0 * math.sqrt(max(1e-4, a_max * b))

        for idx, car in enumerate(self.cars[1:], start=1):
            if getattr(car, 'has_exited', False):
                continue
            if idx == len(self.cars) - 1 and getattr(self, 'follower_velocity_profile', None):
                continue
            cars_same = [c for c in self.cars if c.current_road == car.current_road and c != car and not getattr(c, 'has_exited', False)]

            def gap_to(other):
                g = (car.pos - other.pos) % road_len
                return g if g > 0 else float('inf')

            front_car = min(cars_same, key=gap_to, default=None)
            if front_car is not None:
                gap = (car.pos - front_car.pos - front_car.length) % road_len
                vf = front_car.velocity
            else:
                gap = road_len
                vf = v0

            v = max(0.0, car.velocity)
            vf = max(0.0, vf)
            dv = v - vf
            s = max(gap, 0.1)

            # s* = s0 + max(0, v*T + v*dv / (2 * sqrt(a_max * b)))
            interaction = v * T + (v * dv) / denom_ab
            s_star = s0 + max(0.0, interaction)

            # a = a_max * [1 - (v / v0)^delta - (s* / s)^2]
            free_ratio = (v / v0) ** delta if v0 > 0 else 0.0
            gap_ratio = (s_star / s) ** 2
            target_acc = a_max * (1.0 - free_ratio - gap_ratio)

            # Enforce physical bounds [min_a, max_a]
            target_acc = max(self.min_a, min(self.max_a, target_acc))

            # Enforce physical jerk limit: +/- 5 m/s^3 relative to previous step acceleration
            last_acc = getattr(car, '_prev_acc', car.acceleration)
            jerk = (target_acc - last_acc) / dt
            max_jerk = 5.0

            if jerk > max_jerk:
                acc = last_acc + max_jerk * dt
            elif jerk < -max_jerk:
                acc = last_acc - max_jerk * dt
            else:
                acc = target_acc

            acc = max(self.min_a, min(self.max_a, acc))
            car.acceleration = acc


def compute_dii_metrics(follower_traces, leader_trace, dur, dt, min_dis, reaction_time, collisions=0, min_gap=0.0):
    """
    Computes Driver Intuitiveness Index (DII) and its 5 constituent components
    (C1..C5) plus raw metrics across recorded traces of all follower cars.
    """
    total_steps = 0
    phantom_count = 0
    missed_count = 0
    rear_phantom_count = 0

    all_jerks = []
    sign_flips_per_car = []

    quiet_abs_accels = []
    moving_steps_total = 0
    comfortable_steps = 0

    all_follower_accels = []
    all_follower_gaps = []

    num_cars = len(follower_traces)
    s0 = float(min_dis)
    rt = float(reaction_time)

    for car_data in follower_traces:
        accels = np.array(car_data['a'], dtype=np.float64)
        speeds = np.array(car_data['v'], dtype=np.float64)
        gaps = np.array(car_data['gap'], dtype=np.float64)
        vfs = np.array(car_data['vf'], dtype=np.float64)
        bgs = np.array(car_data.get('bg', []), dtype=np.float64)
        vbs = np.array(car_data.get('vb', []), dtype=np.float64)

        n = len(accels)
        total_steps += n

        all_follower_accels.extend(accels.tolist())
        all_follower_gaps.extend(gaps.tolist())

        # Jerks (t >= 1)
        if n > 1:
            car_jerks = np.diff(accels) / dt
            all_jerks.extend(car_jerks.tolist())

        # Acceleration sign flips (ignoring |a| <= 0.1)
        active_signs = [np.sign(a) for a in accels if abs(a) > DII_THRESHOLDS['jerk_deadband']]
        flips = 0
        for k in range(1, len(active_signs)):
            if active_signs[k] != active_signs[k - 1]:
                flips += 1
        sign_flips_per_car.append(flips)

        # Step-by-step evaluations
        for t in range(n):
            a = accels[t]
            v = speeds[t]
            gap = gaps[t]
            vf = vfs[t]

            vrel = v - vf
            desired_gap = s0 + rt * v

            # --- C1: Appropriateness ---
            if vrel > DII_THRESHOLDS['vrel_req_thresh']:
                gap_headroom = max(gap - s0, DII_THRESHOLDS['min_gap_headroom'])
                a_req = (vrel ** 2) / (2.0 * gap_headroom)
            else:
                a_req = 0.0

            decel = max(0.0, -a)

            # Phantom braking (Front road clear, not closing, yet ego brakes)
            if (a < DII_THRESHOLDS['phantom_decel_thresh'] and
                gap > DII_THRESHOLDS['phantom_gap_factor'] * desired_gap and
                vrel <= DII_THRESHOLDS['phantom_vrel_thresh']):
                phantom_count += 1

            # Rear-induced phantom braking (Front is safe, rear drops/brakes, ego phantom-decelerates)
            if len(bgs) > t and len(vbs) > t:
                bg = bgs[t]
                vb = vbs[t]
                desired_bg = s0 + rt * vb
                is_front_clear = (gap >= desired_gap) and (vrel <= 0.0)
                is_rear_disturbed = (v - vb > 0.5) or (bg > desired_bg + 4.0)
                if is_front_clear and is_rear_disturbed and (a < DII_THRESHOLDS['rear_decel_thresh']):
                    rear_phantom_count += 1

            # Missed braking
            if (a_req > DII_THRESHOLDS['missed_areq_thresh'] and
                decel < DII_THRESHOLDS['missed_decel_ratio'] * a_req):
                missed_count += 1

            # --- C3: Calmness ---
            is_quiet = (
                (abs(vrel) < DII_THRESHOLDS['quiet_vrel_thresh'] and
                 abs(gap - desired_gap) < DII_THRESHOLDS['quiet_gap_ratio'] * max(desired_gap, 1.0))
                or
                (v < DII_THRESHOLDS['quiet_standstill_v'] and vf < DII_THRESHOLDS['quiet_standstill_v'])
            )
            if is_quiet:
                quiet_abs_accels.append(abs(a))

            # --- C4: Comfort Margin ---
            if v > DII_THRESHOLDS['moving_v_thresh']:
                moving_steps_total += 1
                headway = gap / max(v, 0.1)
                ttc = gap / vrel if vrel > DII_THRESHOLDS['ttc_vrel_thresh'] else float('inf')
                if headway >= DII_THRESHOLDS['min_headway'] and ttc >= DII_THRESHOLDS['min_ttc']:
                    comfortable_steps += 1

    # C1 Calculation
    # Total phantom braking includes both forward unprovoked braking AND rear-induced phantom braking
    phantom_rate = phantom_count / total_steps if total_steps > 0 else 0.0
    rear_phantom_rate = rear_phantom_count / total_steps if total_steps > 0 else 0.0
    total_phantom_rate = phantom_rate + rear_phantom_rate
    missed_rate = missed_count / total_steps if total_steps > 0 else 0.0
    combined_rate = total_phantom_rate + missed_rate
    C1 = float(max(0.0, 1.0 - min(1.0, combined_rate / DII_THRESHOLDS['rate_penalty_scale'])))

    # C2 Calculation
    if all_jerks:
        rms_jerk = float(np.sqrt(np.mean(np.array(all_jerks) ** 2)))
    else:
        rms_jerk = 0.0

    duration_min = max(0.01, dur / 60.0)
    avg_flips_per_car = float(np.mean(sign_flips_per_car)) if sign_flips_per_car else 0.0
    sign_changes_per_min = avg_flips_per_car / duration_min

    term_jerk = math.exp(-rms_jerk / DII_THRESHOLDS['rms_jerk_scale'])
    term_flips = math.exp(-sign_changes_per_min / DII_THRESHOLDS['sign_changes_scale'])
    C2 = float(0.5 * term_jerk + 0.5 * term_flips)

    # C3 Calculation
    if quiet_abs_accels:
        mean_abs_a_quiet = float(np.mean(quiet_abs_accels))
        C3 = float(math.exp(-mean_abs_a_quiet / DII_THRESHOLDS['calm_accel_scale']))
    else:
        C3 = 1.0

    # C4 Calculation
    if moving_steps_total > 0:
        C4_base = float(comfortable_steps / moving_steps_total)
    else:
        C4_base = 1.0

    # Penalize dangerous clearance encroachment (< 2.0 m safety clearance floor)
    if min_gap < 2.0:
        clearance_penalty = (2.0 - max(0.0, min_gap)) / 2.0
        C4 = float(max(0.0, C4_base * (1.0 - 0.5 * clearance_penalty)))
    else:
        C4 = C4_base

    # C5 Calculation (Wave Behaviour - Platoon string stability ratio)
    leader_accels = np.array(leader_trace.get('a', [0.0]), dtype=np.float64)
    max_follower_decel = float(np.max(np.maximum(0.0, -np.array(all_follower_accels)))) if all_follower_accels else 0.0
    max_leader_decel = float(np.max(np.maximum(0.0, -leader_accels))) if len(leader_accels) > 0 else 0.0
    eff_leader_decel = max(max_leader_decel, DII_THRESHOLDS['min_leader_decel'])

    wave_ratio = max_follower_decel / eff_leader_decel
    if wave_ratio == 0.0:
        C5 = 1.0
    else:
        C5 = float(min(1.0, 1.0 / wave_ratio))

    # Rear Decoupling / Autonomy: Human drivers never brake because trailing vehicles decelerated
    rear_decoupling = float(max(0.0, 1.0 - (rear_phantom_rate / 0.05)))

    # Driver Intuitiveness Index (DII) represents pure in-cabin human car-following experience:
    # C1: Appropriateness (proper braking, 0 forward phantom braking, 0 rear-induced braking)
    # C2: Passenger Comfort & Smoothness (jerk and sign flips)
    # C3: Following Calmness (minimal hunting at steady state)
    # C4: Natural Safety Margin (headway >= 0.8s, TTC >= 4.0s, safe bumper clearance >= 2.0m)
    # RDI: Rear Decoupling / Autonomy (freedom from trailing vehicle disturbance)
    # (Note: Platoon string stability / wave ratio is tracked and reported as a separate platoon diagnostic)
    DII = float(100.0 * np.mean([C1, C2, C3, C4, rear_decoupling]))

    return {
        'DII': DII,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        'C4': C4,
        'C5': C5,
        'rear_decoupling': rear_decoupling,
        'raw': {
            'phantom_rate': phantom_rate,
            'missed_rate': missed_rate,
            'rear_phantom_rate': rear_phantom_rate,
            'rear_phantom_count': int(rear_phantom_count),
            'rear_decoupling': rear_decoupling,
            'rms_jerk': rms_jerk,
            'sign_changes_per_min': sign_changes_per_min,
            'wave_ratio': wave_ratio,
            'collisions': int(collisions),
            'min_gap': float(min_gap),
            'mean_gap': float(np.mean(all_follower_gaps)) if all_follower_gaps else 0.0,
        }
    }


def compute_idm_comparison_metrics(ctrl_traces, idm_traces):
    """
    Computes pairwise statistical and kinematic comparisons between a controller
    and an IDM reference model on the same scenario with the same leader profile.
    """
    car_accel_rmses = []
    car_gap_rmses = []

    all_ctrl_a = []
    all_idm_a = []

    moving_ctrl_headways = []
    moving_idm_headways = []

    num_cars = min(len(ctrl_traces), len(idm_traces))
    for i in range(num_cars):
        c_a = np.array(ctrl_traces[i]['a'], dtype=np.float64)
        m_a = np.array(idm_traces[i]['a'], dtype=np.float64)
        c_g = np.array(ctrl_traces[i]['gap'], dtype=np.float64)
        m_g = np.array(idm_traces[i]['gap'], dtype=np.float64)
        c_v = np.array(ctrl_traces[i]['v'], dtype=np.float64)
        m_v = np.array(idm_traces[i]['v'], dtype=np.float64)

        n = min(len(c_a), len(m_a))
        if n > 0:
            a_rmse = float(np.sqrt(np.mean((c_a[:n] - m_a[:n]) ** 2)))
            g_rmse = float(np.sqrt(np.mean((c_g[:n] - m_g[:n]) ** 2)))
            car_accel_rmses.append(a_rmse)
            car_gap_rmses.append(g_rmse)

            all_ctrl_a.extend(c_a[:n].tolist())
            all_idm_a.extend(m_a[:n].tolist())

            for k in range(n):
                if c_v[k] > DII_THRESHOLDS['moving_v_thresh']:
                    moving_ctrl_headways.append(float(c_g[k] / max(c_v[k], 0.1)))
                if m_v[k] > DII_THRESHOLDS['moving_v_thresh']:
                    moving_idm_headways.append(float(m_g[k] / max(m_v[k], 0.1)))

    mean_accel_rmse = float(np.mean(car_accel_rmses)) if car_accel_rmses else 0.0
    mean_gap_rmse = float(np.mean(car_gap_rmses)) if car_gap_rmses else 0.0

    # Wasserstein-1 distance on pooled accelerations
    if all_ctrl_a and all_idm_a:
        w1_accel = float(wasserstein_distance(all_ctrl_a, all_idm_a))
    else:
        w1_accel = 0.0

    # Wasserstein-1 distance on time-headways (moving steps)
    if moving_ctrl_headways and moving_idm_headways:
        w1_headway = float(wasserstein_distance(moving_ctrl_headways, moving_idm_headways))
    else:
        w1_headway = 0.0

    human_similarity = float(math.exp(-mean_accel_rmse / 0.5))

    return {
        'accel_rmse': mean_accel_rmse,
        'gap_rmse': mean_gap_rmse,
        'w1_accel': w1_accel,
        'w1_headway': w1_headway,
        'human_similarity': human_similarity,
        'per_car_accel_rmse': car_accel_rmses,
        'per_car_gap_rmse': car_gap_rmses,
    }


def simulate_benchmark_scenario(sc, rl_model, sim_params=None):
    """
    Runs an official benchmark scenario across all 4 controllers (ACC, BCC, ACC+BCC, RL)
    and both IDM variants (Gap-Matched and Default textbook).
    Collects per-follower step traces and collisions.
    """
    params = dict(DEFAULT_SIM_PARAMS if sim_params is None else sim_params)
    params['v_des'] = sc['v_des']

    init_args = [
        params[k] for k in [
            'car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v',
            'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap'
        ]
    ]

    city_acc = City()
    city_acc.init(*init_args, dt=params['dt'], model='ACC')

    city_bcc = City()
    city_bcc.init(*init_args, dt=params['dt'], model='BCC')

    city_accbcc = City()
    city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')

    follower_profile = sc.get('follower_profile', None)

    city_rl = RLCity()
    city_rl.init(*init_args, dt=params['dt'], model='ACC')
    for i, car in enumerate(city_rl.cars):
        if i == 0:
            car.is_rl = False
        elif i == len(city_rl.cars) - 1 and follower_profile:
            car.is_rl = False
        else:
            car.is_rl = True
            car.rl_action = 0.0

    # IDM Gap-Matched (s0 = min_dis, T = reaction_time)
    idm_gap_params = get_gap_matched_idm_params(params)
    city_idm_gap = IDMCity(idm_params=idm_gap_params)
    city_idm_gap.init(*init_args, dt=params['dt'], model='ACC')

    # IDM Default (s0 = 2.0, T = 1.5)
    city_idm_def = IDMCity(idm_params=IDM_PARAMS)
    city_idm_def.init(*init_args, dt=params['dt'], model='ACC')

    # Assign lead vehicle velocity profile
    lead_profile = []
    if sc['type'] == 'csv':
        lead_profile = load_csv_profile(sc['csv_file'])
    elif sc['type'] == 'synthetic':
        lead_profile = sc['profile']
    elif sc['type'] == 'wave':
        lead_profile = [
            (t, 15.0 + 4.0 * np.sin(2 * np.pi * t / 15.0))
            for t in np.arange(0, sc['dur'] + 1.0, 0.5)
        ]

    cities = {
        'ACC': city_acc,
        'BCC': city_bcc,
        'ACC+BCC': city_accbcc,
        'RL': city_rl,
        'IDM_gap_matched': city_idm_gap,
        'IDM_default': city_idm_def,
    }

    if lead_profile:
        for c in cities.values():
            c.lead_velocity_profile = lead_profile

    if follower_profile:
        for c in cities.values():
            c.follower_velocity_profile = list(follower_profile)

    rear_cutout_time = sc.get('rear_cutout_time', None)
    if rear_cutout_time is not None:
        for c in cities.values():
            c.rear_cutout_time = rear_cutout_time

    steps = int(sc['dur'] / params['dt'])
    road_len = city_acc.roads[0].length if city_acc.roads else 1000.0
    num_eval_followers = (params['car_number'] - 2) if follower_profile else (params['car_number'] - 1)

    traces = {
        k: {
            'followers': [{'a': [], 'v': [], 'gap': [], 'vf': [], 'bg': [], 'vb': []} for _ in range(num_eval_followers)],
            'leader': {'a': [], 'v': []},
            'collisions': 0,
            'min_gap': float('inf'),
        }
        for k in cities
    }

    for step in range(steps):
        # Classical controllers and IDM models
        city_acc.run(params['dt'])
        city_bcc.run(params['dt'])
        city_accbcc.run(params['dt'])
        city_idm_gap.run(params['dt'])
        city_idm_def.run(params['dt'])

        # RL model execution exactly via model.predict(deterministic=True)
        obs_arr = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'])
        act, _ = rl_model.predict(obs_arr, deterministic=True)
        for i, car in enumerate(city_rl.cars[1:]):
            if getattr(car, 'is_rl', False):
                act_val = act[i][0] if (np.ndim(act) > 1 and i < len(act)) else (act[i] if i < len(act) else 0.0)
                car.rl_action = float(act_val)
        city_rl.run(params['dt'])

        # Collect per-step metrics and follower traces
        for name, city in cities.items():
            record = traces[name]
            leader = city.cars[0]
            record['leader']['a'].append(leader.acceleration)
            record['leader']['v'].append(leader.velocity)

            step_gaps = []
            eval_cars = city.cars[1 : 1 + num_eval_followers]
            for i, car in enumerate(eval_cars):
                cars_same = [c for c in city.cars if c.current_road == car.current_road and c != car and not getattr(c, 'has_exited', False)]

                def gap_to(other):
                    g = (car.pos - other.pos) % road_len
                    return g if g > 0 else float('inf')

                def gap_from(other):
                    g = (other.pos - car.pos) % road_len
                    return g if g > 0 else float('inf')

                front = min(cars_same, key=gap_to, default=None)
                fg = (car.pos - front.pos - front.length) % road_len if front else road_len
                vf = front.velocity if front else params['v_des']

                back = min(cars_same, key=gap_from, default=None)
                bg = (back.pos - car.pos - car.length) % road_len if back else road_len
                vb = back.velocity if back else car.velocity

                record['followers'][i]['a'].append(car.acceleration)
                record['followers'][i]['v'].append(car.velocity)
                record['followers'][i]['gap'].append(fg)
                record['followers'][i]['vf'].append(vf)
                record['followers'][i]['bg'].append(bg)
                record['followers'][i]['vb'].append(vb)
                step_gaps.append(fg)

            min_g = min(step_gaps) if step_gaps else road_len
            record['min_gap'] = min(record['min_gap'], min_g)
            collided = (min_g < 0.5) or getattr(city, 'collided_this_step', False)
            if collided:
                record['collisions'] += 1

    # Compute DII results for all 6 models
    dii_results = {}
    for name, record in traces.items():
        dii_results[name] = compute_dii_metrics(
            record['followers'],
            record['leader'],
            dur=sc['dur'],
            dt=params['dt'],
            min_dis=params['min_dis'],
            reaction_time=params['reaction_time'],
            collisions=record['collisions'],
            min_gap=record['min_gap'],
        )

    # Compute pairwise IDM comparisons vs IDM (Gap-Matched) and IDM (Default)
    idm_comp_gap = {}
    idm_comp_def = {}
    for name in ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_gap_matched']:
        idm_comp_gap[name] = compute_idm_comparison_metrics(
            traces[name]['followers'],
            traces['IDM_gap_matched']['followers']
        )
    for name in ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_default']:
        idm_comp_def[name] = compute_idm_comparison_metrics(
            traces[name]['followers'],
            traces['IDM_default']['followers']
        )

    return dii_results, idm_comp_gap, idm_comp_def


def generate_dii_plots(all_dii, scenarios, output_png=None):
    """
    Generates and saves a publication-quality comparison bar chart showing:
    1. Overall DII per scenario for ACC, BCC, ACC+BCC, RL, and IDM (Gap-Matched)
    2. Five-component breakdown (C1..C5) across controllers
    """
    if output_png is None:
        output_png = PLOTS_DIR / "dii_comparison.png"

    sc_names = [sc['name'].split('.')[1].strip() if '.' in sc['name'] else sc['name'] for sc in scenarios]
    controllers = ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_gap_matched']
    colors = {
        'ACC': '#4A90E2',
        'BCC': '#50E3C2',
        'ACC+BCC': '#F5A623',
        'RL': '#E02020',
        'IDM_gap_matched': '#9013FE',
    }
    labels = {
        'ACC': 'ACC',
        'BCC': 'BCC',
        'ACC+BCC': 'ACC+BCC',
        'RL': 'RL (TD3)',
        'IDM_gap_matched': 'IDM Ref (Gap-Matched)',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 12), gridspec_kw={'height_ratios': [1.2, 1.0]})
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: DII per scenario
    x = np.arange(len(sc_names))
    width = 0.15
    for idx, ctrl in enumerate(controllers):
        scores = [all_dii[sc['id']][ctrl]['DII'] for sc in scenarios]
        offset = (idx - 2) * width
        rects = ax1.bar(x + offset, scores, width, label=labels[ctrl], color=colors[ctrl], alpha=0.9, edgecolor='black', linewidth=0.5)
        for r in rects:
            h = r.get_height()
            if h > 2:
                ax1.text(r.get_x() + r.get_width() / 2.0, h + 1.0, f"{h:.0f}", ha='center', va='bottom', fontsize=6.5, rotation=0)

    ax1.set_ylabel('Driver Intuitiveness Index (DII, 0-100)', fontsize=11, fontweight='bold')
    ax1.set_title('Driver Intuitiveness Index (DII) Across Benchmark Regimes', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=7.5, rotation=22, ha='right')
    ax1.set_ylim(0, 108)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    # Subplot 2: Component Breakdown (Mean over all scenarios)
    components = ['C1 (Approp)', 'C2 (Smooth)', 'C3 (Calm)', 'C4 (Comfort)', 'RDI (Decoup)']
    x2 = np.arange(len(components))
    for idx, ctrl in enumerate(controllers):
        comp_means = [
            np.mean([all_dii[sc['id']][ctrl]['C1'] for sc in scenarios]) * 100.0,
            np.mean([all_dii[sc['id']][ctrl]['C2'] for sc in scenarios]) * 100.0,
            np.mean([all_dii[sc['id']][ctrl]['C3'] for sc in scenarios]) * 100.0,
            np.mean([all_dii[sc['id']][ctrl]['C4'] for sc in scenarios]) * 100.0,
            np.mean([all_dii[sc['id']][ctrl]['rear_decoupling'] for sc in scenarios]) * 100.0,
        ]
        offset = (idx - 2) * width
        rects = ax2.bar(x2 + offset, comp_means, width, label=labels[ctrl], color=colors[ctrl], alpha=0.9, edgecolor='black', linewidth=0.5)
        for r in rects:
            h = r.get_height()
            if h > 2:
                ax2.text(r.get_x() + r.get_width() / 2.0, h + 1.0, f"{h:.0f}", ha='center', va='bottom', fontsize=7.5)

    ax2.set_ylabel('Component Score (0-100)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Overall Component Breakdown (Mean Across All {len(scenarios)} Scenarios)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(components, fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 108)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved DII comparison plot: {output_png}")


def generate_idm_plots(idm_comp_gap_all, idm_comp_def_all, scenarios, output_png=None):
    """
    Generates and saves a comparison bar chart showing acceleration RMSE vs IDM
    per scenario for both IDM (Gap-Matched) and IDM (Default).
    """
    if output_png is None:
        output_png = PLOTS_DIR / "idm_comparison.png"

    sc_names = [sc['name'].split('.')[1].strip() if '.' in sc['name'] else sc['name'] for sc in scenarios]
    controllers = ['ACC', 'BCC', 'ACC+BCC', 'RL']
    colors = {
        'ACC': '#4A90E2',
        'BCC': '#50E3C2',
        'ACC+BCC': '#F5A623',
        'RL': '#E02020',
    }
    labels = {
        'ACC': 'ACC',
        'BCC': 'BCC',
        'ACC+BCC': 'ACC+BCC',
        'RL': 'RL (TD3)',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 12))
    fig.patch.set_facecolor('#ffffff')

    x = np.arange(len(sc_names))
    width = 0.18

    # Top: Gap-Matched IDM
    for idx, ctrl in enumerate(controllers):
        rmses = [idm_comp_gap_all[sc['id']][ctrl]['accel_rmse'] for sc in scenarios]
        offset = (idx - 1.5) * width
        rects = ax1.bar(x + offset, rmses, width, label=labels[ctrl], color=colors[ctrl], alpha=0.9, edgecolor='black', linewidth=0.5)
        for r in rects:
            h = r.get_height()
            ax1.text(r.get_x() + r.get_width() / 2.0, h + 0.03, f"{h:.2f}", ha='center', va='bottom', fontsize=6.5)

    ax1.set_ylabel('Accel RMSE (m/s^2) vs IDM', fontsize=10.5, fontweight='bold')
    ax1.set_title('Acceleration RMSE vs Gap-Matched IDM (s0=6m, T=0.8s) Across Scenarios', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_names, fontsize=7.5, rotation=22, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    # Bottom: Default Textbook IDM
    for idx, ctrl in enumerate(controllers):
        rmses = [idm_comp_def_all[sc['id']][ctrl]['accel_rmse'] for sc in scenarios]
        offset = (idx - 1.5) * width
        rects = ax2.bar(x + offset, rmses, width, label=labels[ctrl], color=colors[ctrl], alpha=0.9, edgecolor='black', linewidth=0.5)
        for r in rects:
            h = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2.0, h + 0.03, f"{h:.2f}", ha='center', va='bottom', fontsize=6.5)

    ax2.set_ylabel('Accel RMSE (m/s^2) vs IDM', fontsize=10.5, fontweight='bold')
    ax2.set_title('Acceleration RMSE vs Default Textbook IDM (s0=2m, T=1.5s) Across Scenarios', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_names, fontsize=7.5, rotation=22, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved IDM comparison plot: {output_png}")


def run_driver_intuitiveness_suite(model_path="td3_accel_agent.zip", scenarios=None, save_outputs=True):
    """
    Main execution pipeline for Driver Intuitiveness & IDM reference benchmark.
    Runs all 7 scenarios, prints formatted comparison tables, saves JSON logs,
    and generates visualization figures.
    """
    resolved_model = resolve_model_path(model_path)
    if not os.path.exists(resolved_model):
        raise FileNotFoundError(f"Model file not found: {resolved_model}")

    if scenarios is None:
        scenarios = get_scenarios()

    print("\n" + "=" * 78)
    print(" DRIVER INTUITIVENESS & IDM HUMAN REFERENCE EVALUATION SUITE")
    print("=" * 78)
    print(f"Evaluated Policy Model: {resolved_model}")
    print(f"Primary IDM Reference: Gap-Matched IDM (s0=6.0m, T=0.8s, a_max=1.5m/s^2, b=2.0m/s^2)")
    print(f"Secondary IDM Variant: Default Textbook IDM (s0=2.0m, T=1.5s, a_max=1.5m/s^2, b=2.0m/s^2)")
    print("=" * 78)

    rl_model = TD3.load(resolved_model)

    all_dii = {}
    idm_comp_gap_all = {}
    idm_comp_def_all = {}

    ctrl_list = ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_gap_matched', 'IDM_default']

    for sc in scenarios:
        print(f"\n>>> Simulating {sc['name']}...")
        dii_res, idm_gap, idm_def = simulate_benchmark_scenario(sc, rl_model)
        all_dii[sc['id']] = dii_res
        idm_comp_gap_all[sc['id']] = idm_gap
        idm_comp_def_all[sc['id']] = idm_def

        # Print per-scenario table
        print(f"\n--- Scenario Results: {sc['name']} ---")
        print(f"{'Controller':<18} | {'DII':>6} | {'C1':>5} | {'C2':>5} | {'C3':>5} | {'C4':>5} | {'R-Decoup':>8} | {'RMS Jerk':>8} | {'Flips/m':>7} | {'Wave':>6} | {'MinGap':>6}")
        print("-" * 96)
        for c in ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_gap_matched']:
            m = dii_res[c]
            r = m['raw']
            c_label = 'RL (TD3)' if c == 'RL' else ('IDM (Gap-Ref)' if c == 'IDM_gap_matched' else c)
            print(f"{c_label:<18} | {m['DII']:6.1f} | {m['C1']:5.2f} | {m['C2']:5.2f} | {m['C3']:5.2f} | {m['C4']:5.2f} | {m['rear_decoupling']:8.2f} | {r['rms_jerk']:8.2f} | {r['sign_changes_per_min']:7.1f} | {r['wave_ratio']:6.2f} | {r['min_gap']:6.2f}m")

    # Aggregate Overall Results (Averaged over all benchmark scenarios)
    overall_dii = {}
    for c in ctrl_list:
        overall_dii[c] = {
            'DII': float(np.mean([all_dii[sc['id']][c]['DII'] for sc in scenarios])),
            'C1': float(np.mean([all_dii[sc['id']][c]['C1'] for sc in scenarios])),
            'C2': float(np.mean([all_dii[sc['id']][c]['C2'] for sc in scenarios])),
            'C3': float(np.mean([all_dii[sc['id']][c]['C3'] for sc in scenarios])),
            'C4': float(np.mean([all_dii[sc['id']][c]['C4'] for sc in scenarios])),
            'C5': float(np.mean([all_dii[sc['id']][c]['C5'] for sc in scenarios])),
            'rear_decoupling': float(np.mean([all_dii[sc['id']][c]['rear_decoupling'] for sc in scenarios])),
            'raw': {
                'phantom_rate': float(np.mean([all_dii[sc['id']][c]['raw']['phantom_rate'] for sc in scenarios])),
                'missed_rate': float(np.mean([all_dii[sc['id']][c]['raw']['missed_rate'] for sc in scenarios])),
                'rear_phantom_rate': float(np.mean([all_dii[sc['id']][c]['raw']['rear_phantom_rate'] for sc in scenarios])),
                'rear_phantom_count': int(sum(all_dii[sc['id']][c]['raw']['rear_phantom_count'] for sc in scenarios)),
                'rear_decoupling': float(np.mean([all_dii[sc['id']][c]['raw']['rear_decoupling'] for sc in scenarios])),
                'rms_jerk': float(np.mean([all_dii[sc['id']][c]['raw']['rms_jerk'] for sc in scenarios])),
                'sign_changes_per_min': float(np.mean([all_dii[sc['id']][c]['raw']['sign_changes_per_min'] for sc in scenarios])),
                'wave_ratio': float(np.mean([all_dii[sc['id']][c]['raw']['wave_ratio'] for sc in scenarios])),
                'collisions': int(sum(all_dii[sc['id']][c]['raw']['collisions'] for sc in scenarios)),
                'min_gap': float(min(all_dii[sc['id']][c]['raw']['min_gap'] for sc in scenarios)),
                'mean_gap': float(np.mean([all_dii[sc['id']][c]['raw']['mean_gap'] for sc in scenarios])),
            }
        }

    overall_idm_gap = {}
    overall_idm_def = {}
    for c in ['ACC', 'BCC', 'ACC+BCC', 'RL']:
        overall_idm_gap[c] = {
            'accel_rmse': float(np.mean([idm_comp_gap_all[sc['id']][c]['accel_rmse'] for sc in scenarios])),
            'gap_rmse': float(np.mean([idm_comp_gap_all[sc['id']][c]['gap_rmse'] for sc in scenarios])),
            'w1_accel': float(np.mean([idm_comp_gap_all[sc['id']][c]['w1_accel'] for sc in scenarios])),
            'w1_headway': float(np.mean([idm_comp_gap_all[sc['id']][c]['w1_headway'] for sc in scenarios])),
            'human_similarity': float(np.mean([idm_comp_gap_all[sc['id']][c]['human_similarity'] for sc in scenarios])),
        }
        overall_idm_def[c] = {
            'accel_rmse': float(np.mean([idm_comp_def_all[sc['id']][c]['accel_rmse'] for sc in scenarios])),
            'gap_rmse': float(np.mean([idm_comp_def_all[sc['id']][c]['gap_rmse'] for sc in scenarios])),
            'w1_accel': float(np.mean([idm_comp_def_all[sc['id']][c]['w1_accel'] for sc in scenarios])),
            'w1_headway': float(np.mean([idm_comp_def_all[sc['id']][c]['w1_headway'] for sc in scenarios])),
            'human_similarity': float(np.mean([idm_comp_def_all[sc['id']][c]['human_similarity'] for sc in scenarios])),
        }

    # Print Overall DII Benchmark Table
    print("\n" + "=" * 96)
    print(" OVERALL DRIVER INTUITIVENESS INDEX (DII) BENCHMARK SUMMARY")
    print(" (Based on In-Cabin Driver Experience: C1 Appropriateness, C2 Smoothness, C3 Calmness,")
    print("  C4 Comfort Margin, and R-Decoup Rear Decoupling / Autonomy. Wave Ratio tracked as diagnostic.)")
    print("=" * 96)
    print(f"{'Controller':<18} | {'DII':>6} | {'C1':>5} | {'C2':>5} | {'C3':>5} | {'C4':>5} | {'R-Decoup':>8} | {'RMS Jerk':>8} | {'Flips/m':>7} | {'Wave':>6} | {'MinGap':>6}")
    print("-" * 96)
    for c in ['ACC', 'BCC', 'ACC+BCC', 'RL', 'IDM_gap_matched', 'IDM_default']:
        m = overall_dii[c]
        r = m['raw']
        if c == 'RL':
            c_label = 'RL (TD3)'
        elif c == 'IDM_gap_matched':
            c_label = 'IDM (Gap-Ref)'
        elif c == 'IDM_default':
            c_label = 'IDM (Default)'
        else:
            c_label = c
        print(f"{c_label:<18} | {m['DII']:6.1f} | {m['C1']:5.2f} | {m['C2']:5.2f} | {m['C3']:5.2f} | {m['C4']:5.2f} | {m['rear_decoupling']:8.2f} | {r['rms_jerk']:8.2f} | {r['sign_changes_per_min']:7.1f} | {r['wave_ratio']:6.2f} | {r['min_gap']:6.2f}m")
    print("=" * 96)

    # Print IDM Comparison Table (Primary: Gap-Matched IDM)
    print("\n" + "=" * 90)
    print(" IDM HUMAN REFERENCE COMPARISON SUMMARY (vs Gap-Matched IDM Reference)")
    print("=" * 90)
    print(f"{'Controller':<16} | {'Accel RMSE':>10} | {'Gap RMSE':>9} | {'W1 Accel':>9} | {'W1 Headway':>10} | {'Similarity':>10}")
    print("-" * 90)
    for c in ['ACC', 'BCC', 'ACC+BCC', 'RL']:
        m = overall_idm_gap[c]
        c_label = 'RL (TD3)' if c == 'RL' else c
        print(f"{c_label:<16} | {m['accel_rmse']:10.3f} | {m['gap_rmse']:8.2f}m | {m['w1_accel']:9.3f} | {m['w1_headway']:9.3f}s | {m['human_similarity']:10.3f}")
    print("=" * 90)

    # Print IDM Comparison Table (Secondary: Default Textbook IDM)
    print("\n" + "=" * 90)
    print(" IDM HUMAN REFERENCE COMPARISON SUMMARY (vs Default Textbook IDM: s0=2m, T=1.5s)")
    print("=" * 90)
    print(f"{'Controller':<16} | {'Accel RMSE':>10} | {'Gap RMSE':>9} | {'W1 Accel':>9} | {'W1 Headway':>10} | {'Similarity':>10}")
    print("-" * 90)
    for c in ['ACC', 'BCC', 'ACC+BCC', 'RL']:
        m = overall_idm_def[c]
        c_label = 'RL (TD3)' if c == 'RL' else c
        print(f"{c_label:<16} | {m['accel_rmse']:10.3f} | {m['gap_rmse']:8.2f}m | {m['w1_accel']:9.3f} | {m['w1_headway']:9.3f}s | {m['human_similarity']:10.3f}")
    print("=" * 90)

    if save_outputs:
        # Save logs/driver_intuitiveness.json
        dii_json_path = LOGS_DIR / "driver_intuitiveness.json"
        dii_payload = {
            'metadata': {
                'evaluated_model': str(resolved_model),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'thresholds': DII_THRESHOLDS,
            },
            'overall': overall_dii,
            'scenarios': all_dii,
        }
        with open(dii_json_path, 'w', encoding='utf-8') as f:
            json.dump(dii_payload, f, indent=2)
        print(f"\nSaved DII results log: {dii_json_path}")

        # Save logs/idm_comparison.json
        idm_json_path = LOGS_DIR / "idm_comparison.json"
        idm_payload = {
            'metadata': {
                'evaluated_model': str(resolved_model),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'idm_gap_matched_params': get_gap_matched_idm_params(),
                'idm_default_params': IDM_PARAMS,
            },
            'gap_matched_reference': {
                'overall': overall_idm_gap,
                'scenarios': idm_comp_gap_all,
            },
            'default_reference': {
                'overall': overall_idm_def,
                'scenarios': idm_comp_def_all,
            }
        }
        with open(idm_json_path, 'w', encoding='utf-8') as f:
            json.dump(idm_payload, f, indent=2)
        print(f"Saved IDM comparison log: {idm_json_path}")

        # Generate plots
        generate_dii_plots(all_dii, scenarios)
        generate_idm_plots(idm_comp_gap_all, idm_comp_def_all, scenarios)

    return {
        'all_dii': all_dii,
        'overall_dii': overall_dii,
        'idm_comp_gap_all': idm_comp_gap_all,
        'idm_comp_def_all': idm_comp_def_all,
        'overall_idm_gap': overall_idm_gap,
        'overall_idm_def': overall_idm_def,
    }


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    run_driver_intuitiveness_suite(target)
