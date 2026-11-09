"""
Cut-In Vulnerability & X-Factor Kinematic Metric Suite.

Quantifies the susceptibility of platoon car-following controllers
(ACC, BCC, ACC+BCC, RL/TD3) to vehicle cut-ins from adjacent lanes.

Key Concepts:
1. Kinematic Safety Envelope (X-Factor):
   X = G_front_min + Le + G_rear_min
   where:
     G_front_min = ve * Tr + (vl - ve)^2 / (2 * ae) + Lb
     Grear_min  = vf * Tr + (vf - ve)^2 / (2 * af) + Lb
     Le = car length (5.0 m)
     Lb = standstill safety buffer (2.0 m)
   X is the exact physical space required for an adjacent vehicle to insert
   into the lane without violating emergency braking envelopes.

2. Metrics:
   - X-Factor Dilation Ratio: Xi(t) = s_front(t) / X(t)
   - X-Breach Exposure: Duration (s) and fraction (%) where s_front > X
   - CIVI_X (Cut-In Vulnerability Index): (1/T) * integral max(0, s_front/X - 1) dt
   - Time Headway Exposure: Duration (s) where headway h = s_front / v > 2.0s
   - Stochastic Cut-In Arrival Probability: Poisson hazard rate model P(Cut-In)
   - Disturbance Leakage Ratio: eta_leak = Delta s_front / Delta s_rear
"""

import math
import numpy as np


# Default physical constants matching city.py & config.py
DEFAULT_CAR_LENGTH = 5.0      # Le (m)
DEFAULT_MIN_GAP = 2.0         # Lb / standstill margin (m)
DEFAULT_REACTION_TIME = 0.8   # Tr (s)
DEFAULT_MIN_A = -5.0          # Maximum braking deceleration (m/s^2)
DEFAULT_CRIT_HEADWAY = 2.0    # Time headway threshold for cut-in opportunity (s)
DEFAULT_POISSON_LAMBDA0 = 0.04  # Baseline cut-in hazard rate (s^-1)
DEFAULT_POISSON_GAMMA = 1.2   # Hazard scaling exponent


def compute_kinematic_x(
    v_ego,
    v_lead=None,
    v_rear=None,
    min_a=DEFAULT_MIN_A,
    reaction_time=DEFAULT_REACTION_TIME,
    min_gap=DEFAULT_MIN_GAP,
    car_length=DEFAULT_CAR_LENGTH,
):
    """
    Computes the instantaneous kinematic safety envelope X for an ego vehicle.

    X = G_front_min + Le + G_rear_min
    where:
      G_front_min = ve * Tr + (vl - ve)^2 / (2 * ae) + Lb
      Grear_min  = vf * Tr + (vf - ve)^2 / (2 * af) + Lb

    Args:
        v_ego: Speed of the ego vehicle (m/s) (scalar or array)
        v_lead: Speed of the vehicle ahead (m/s). Defaults to v_ego if None.
        v_rear: Speed of the vehicle behind (m/s). Defaults to v_ego if None.
        min_a: Maximum braking deceleration limit (m/s^2, typically -5.0)
        reaction_time: Driver/sensor reaction time Tr (s, typically 0.8)
        min_gap: Standstill safety buffer Lb (m, typically 2.0)
        car_length: Vehicle footprint Le (m, typically 5.0)

    Returns:
        X: Total kinematic clearance envelope in meters (scalar or ndarray)
    """
    ve = np.maximum(0.0, np.asarray(v_ego, dtype=np.float64))
    vl = ve if v_lead is None else np.maximum(0.0, np.asarray(v_lead, dtype=np.float64))
    vf = ve if v_rear is None else np.maximum(0.0, np.asarray(v_rear, dtype=np.float64))

    ae = abs(float(min_a))
    af = ae * 0.7  # Follower braking capability assumption
    Tr = float(reaction_time)
    Lb = float(min_gap)
    Le = float(car_length)

    g_front_min = ve * Tr + ((vl - ve) ** 2) / (2.0 * ae) + Lb
    g_rear_min = vf * Tr + ((vf - ve) ** 2) / (2.0 * af) + Lb

    x_val = g_front_min + Le + g_rear_min
    return x_val


def compute_cut_in_metrics(
    gaps,
    speeds,
    lead_speeds=None,
    rear_speeds=None,
    rear_gaps=None,
    dt=0.1,
    min_a=DEFAULT_MIN_A,
    reaction_time=DEFAULT_REACTION_TIME,
    min_gap=DEFAULT_MIN_GAP,
    car_length=DEFAULT_CAR_LENGTH,
    crit_headway=DEFAULT_CRIT_HEADWAY,
    v_des=15.0,
):
    """
    Computes the complete suite of Cut-In Vulnerability and X-Factor metrics
    over a recorded car-following trajectory.

    Args:
        gaps: Array-like of forward gaps s_f(t) (m)
        speeds: Array-like of ego velocities v_e(t) (m/s)
        lead_speeds: Array-like of front vehicle velocities v_l(t) (m/s, optional)
        rear_speeds: Array-like of rear vehicle velocities v_b(t) (m/s, optional)
        rear_gaps: Array-like of rear gaps s_b(t) (m, optional)
        dt: Simulation timestep (s)
        min_a, reaction_time, min_gap, car_length: Physical parameters
        crit_headway: Headway threshold (s) defining a cut-in opportunity
        v_des: Target cruising velocity (m/s)

    Returns:
        dict containing:
            - max_gap: Maximum forward gap reached (m)
            - mean_gap: Average forward gap (m)
            - nominal_gap: Equilibrium desired gap at v_des (m)
            - r_gap: Peak gap expansion ratio max(gap) / nominal_gap
            - mean_x: Average kinematic envelope X (m)
            - max_x_ratio: Peak ratio max(s_front / X) (dimensionless Xi)
            - t_x_breach: Total duration s_front > X (s)
            - pct_x_breach: Percentage of moving time where s_front > X (%)
            - civi_x: X-Factor Cut-In Vulnerability Index (s_excess/X per second)
            - civi_x_score: Normalized CIVI score in [0, 100]
            - t_headway_2s: Total duration with time headway > crit_headway (s)
            - pct_headway_2s: % of moving time with headway > crit_headway (%)
            - max_headway: Peak time headway observed (s)
            - expected_cut_ins: Expected Poisson cut-in arrivals
            - p_cut_in: Cumulative probability of at least one cut-in event (%)
            - leakage_ratio: Disturbance leakage ratio Delta s_front / Delta s_rear
    """
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    speeds_arr = np.asarray(speeds, dtype=np.float64)
    n_steps = len(gaps_arr)

    if n_steps == 0:
        return {
            'max_gap': 0.0,
            'mean_gap': 0.0,
            'nominal_gap': min_gap + reaction_time * v_des,
            'r_gap': 1.0,
            'mean_x': 0.0,
            'max_x_ratio': 0.0,
            't_x_breach': 0.0,
            'pct_x_breach': 0.0,
            'civi_x': 0.0,
            'civi_x_score': 0.0,
            't_headway_2s': 0.0,
            'pct_headway_2s': 0.0,
            'max_headway': 0.0,
            'expected_cut_ins': 0.0,
            'p_cut_in': 0.0,
            'leakage_ratio': 0.0,
        }

    vl_arr = speeds_arr if (lead_speeds is None or len(lead_speeds) != n_steps) else np.asarray(lead_speeds, dtype=np.float64)
    vb_arr = speeds_arr if (rear_speeds is None or len(rear_speeds) != n_steps) else np.asarray(rear_speeds, dtype=np.float64)

    # 1. Kinematic X envelope across time
    x_arr = compute_kinematic_x(
        v_ego=speeds_arr,
        v_lead=vl_arr,
        v_rear=vb_arr,
        min_a=min_a,
        reaction_time=reaction_time,
        min_gap=min_gap,
        car_length=car_length,
    )
    safe_x = np.maximum(x_arr, 1.0)
    x_ratios = gaps_arr / safe_x

    # 2. Moving steps mask (speed > 1.0 m/s)
    moving_mask = speeds_arr > 1.0
    n_moving = np.sum(moving_mask)
    t_moving = float(n_moving * dt)
    t_total = float(n_steps * dt)

    # 3. X-Breach metrics
    x_breach_mask = (gaps_arr > x_arr) & moving_mask
    t_x_breach = float(np.sum(x_breach_mask) * dt)
    pct_x_breach = float((t_x_breach / t_moving * 100.0) if t_moving > 0 else 0.0)

    # Excess ratio above 1.0: max(0, gap/X - 1)
    excess_x_ratios = np.maximum(0.0, x_ratios - 1.0)
    civi_x = float(np.mean(excess_x_ratios)) if n_steps > 0 else 0.0
    civi_x_score = float(100.0 * (1.0 - math.exp(-civi_x / 0.5)))

    # 4. Time Headway metrics
    safe_speeds = np.maximum(speeds_arr, 0.5)
    headways = gaps_arr / safe_speeds
    moving_headways = headways[moving_mask] if n_moving > 0 else np.array([0.0])

    headway_breach_mask = (headways > crit_headway) & moving_mask
    t_headway_2s = float(np.sum(headway_breach_mask) * dt)
    pct_headway_2s = float((t_headway_2s / t_moving * 100.0) if t_moving > 0 else 0.0)
    max_headway = float(np.max(moving_headways)) if len(moving_headways) > 0 else 0.0

    # 5. Gap expansion ratios
    nominal_gap = float(min_gap + reaction_time * v_des)
    max_gap = float(np.max(gaps_arr))
    mean_gap = float(np.mean(gaps_arr))
    r_gap = float(max_gap / max(nominal_gap, 1.0))
    max_x_ratio = float(np.max(x_ratios))
    mean_x = float(np.mean(x_arr))

    # 6. Stochastic Poisson Cut-In Arrival Probability
    # Hazard rate lambda(t) = lambda0 * [max(0, gap(t) - X(t)) / X(t)]^gamma
    hazard_rates = DEFAULT_POISSON_LAMBDA0 * (excess_x_ratios ** DEFAULT_POISSON_GAMMA)
    expected_cut_ins = float(np.sum(hazard_rates) * dt)
    p_cut_in = float(100.0 * (1.0 - math.exp(-expected_cut_ins)))

    # 7. Rear-to-front disturbance leakage ratio eta_leak = Delta s_f / Delta s_b
    if rear_gaps is not None and len(rear_gaps) == n_steps:
        rb_arr = np.asarray(rear_gaps, dtype=np.float64)
        delta_sf = max_gap - float(gaps_arr[0])
        delta_sb = float(np.max(rb_arr)) - float(rb_arr[0])
        leakage_ratio = float(delta_sf / max(delta_sb, 0.1)) if delta_sb > 1.0 else 0.0
    else:
        leakage_ratio = 0.0

    return {
        'max_gap': round(max_gap, 2),
        'mean_gap': round(mean_gap, 2),
        'nominal_gap': round(nominal_gap, 2),
        'r_gap': round(r_gap, 2),
        'mean_x': round(mean_x, 2),
        'max_x_ratio': round(max_x_ratio, 2),
        't_x_breach': round(t_x_breach, 2),
        'pct_x_breach': round(pct_x_breach, 1),
        'civi_x': round(civi_x, 3),
        'civi_x_score': round(civi_x_score, 1),
        't_headway_2s': round(t_headway_2s, 2),
        'pct_headway_2s': round(pct_headway_2s, 1),
        'max_headway': round(max_headway, 2),
        'expected_cut_ins': round(expected_cut_ins, 2),
        'p_cut_in': round(p_cut_in, 1),
        'leakage_ratio': round(leakage_ratio, 2),
    }
