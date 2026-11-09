import os
import sys
import csv
import math
from pathlib import Path
import numpy as np
from stable_baselines3 import TD3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import DATA_DIR, resolve_model_path, DEFAULT_SIM_PARAMS
from env.accel_traffic_env import AccelTrafficEnv
from evaluation.cut_in_metrics import compute_cut_in_metrics, compute_kinematic_x

# Deterministic Human Driver Profiles (IDM parameterizations)
HUMAN_PROFILES = {
    'AGGRESSIVE': {
        'name': 'Aggressive Human',
        'a_max': 2.5,
        'b': 3.0,
        'T': 0.8,
        's0': 3.0,
        'delta': 4.0,
        'color': '#ef4444'  # Red
    },
    'CAUTIOUS': {
        'name': 'Cautious Human',
        'a_max': 1.0,
        'b': 1.5,
        'T': 1.8,
        's0': 8.0,
        'delta': 4.0,
        'color': '#eab308'  # Amber / Yellow
    },
    'NORMAL': {
        'name': 'Normal Human',
        'a_max': 1.5,
        'b': 2.0,
        'T': 1.2,
        's0': 5.0,
        'delta': 4.0,
        'color': '#94a3b8'  # Slate Silver
    }
}

def load_csv_profile(filename):
    """Loads a velocity profile from CSV file."""
    profile_path = Path(filename)
    if not profile_path.is_absolute():
        profile_path = DATA_DIR / profile_path.name
        
    profile = []
    if profile_path.exists():
        with open(profile_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile.append((float(row['time']), float(row['velocity'])))
    else:
        print(f"Warning: CSV profile not found at {profile_path}")
    return profile

def get_scenarios():
    """
    Returns the official 11 curated, non-redundant operational benchmark scenarios.
    Each scenario defines car count, primary focus, driver profiles, and durations.
    """
    scenarios = [
        {
            "id": "steady_cruise_15",
            "name": "1. Steady Highway Cruise (15 Cars, v_des=15 m/s)",
            "car_number": 15,
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "description": "Equilibrium car following and energy baseline at steady 15 m/s."
        },
        {
            "id": "urban_stop_and_go",
            "name": "2. Urban Stop-and-Go (15 Cars, data.csv)",
            "car_number": 15,
            "v_des": 15.0,
            "type": "csv",
            "csv_file": "data.csv",
            "dur": 60.0,
            "description": "Real-world stop-and-go transient micro-trip from empirical sensor data."
        },
        {
            "id": "full_stop_standstill_hold",
            "name": "3. Platoon Full Stop & Standstill Hold (15 Cars)",
            "car_number": 15,
            "v_des": 15.0,
            "type": "synthetic",
            "profile": [(0.0, 15.0), (12.0, 15.0), (16.0, 0.0), (45.0, 0.0), (50.0, 12.0), (60.0, 15.0)],
            "dur": 60.0,
            "description": "Lead vehicle full stop, 30s standstill holding, and smooth platoon restart."
        },
        {
            "id": "lead_emergency_braking",
            "name": "4. Lead Vehicle Emergency Braking (-4.5 m/s² to 0)",
            "car_number": 15,
            "v_des": 20.0,
            "type": "synthetic",
            "profile": [(0.0, 20.0), (10.0, 20.0), (14.0, 0.0), (40.0, 0.0)],
            "dur": 40.0,
            "description": "Severe front deceleration stress testing collision avoidance and minimum gap."
        },
        {
            "id": "sinusoidal_string_stability",
            "name": "5. Sinusoidal Traffic Wave & String Stability (15 Cars)",
            "car_number": 15,
            "v_des": 15.0,
            "type": "wave",
            "dur": 60.0,
            "description": "Harmonic upstream disturbance wave (15 ± 4 m/s) measuring attenuation ratio."
        },
        {
            "id": "three_car_trailing_halt",
            "name": "6. Minimalist 3-Car: Trailing Vehicle Brakes to Halt (BCC Decoupling)",
            "car_number": 3,
            "v_des": 15.0,
            "type": "none",
            "dur": 30.0,
            "human_trailing": True,
            "follower_profile": [(0.0, 15.0), (5.0, 15.0), (9.0, 0.0), (30.0, 0.0)],
            "description": "Exactly 3 cars. At t=5s Car 2 halts; Car 1 (Ego) must follow Car 0 without phantom braking."
        },
        {
            "id": "three_car_lead_stop",
            "name": "7. Minimalist 3-Car: Lead Vehicle Emergency Stop",
            "car_number": 3,
            "v_des": 15.0,
            "type": "synthetic",
            "profile": [(0.0, 15.0), (5.0, 15.0), (8.5, 0.0), (30.0, 0.0)],
            "dur": 30.0,
            "description": "Exactly 3 cars. At t=5s Car 0 stops; Car 1 safely halts while buffer with Car 2 is tracked."
        },
        {
            "id": "aggressive_rear_tailgater",
            "name": "8. Aggressive Rear Tailgater (Forward Squeeze Stress)",
            "car_number": 15,
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "human_trailing": True,
            "follower_profile": [(0.0, 15.0), (8.0, 15.0), (12.0, 22.0), (26.0, 22.0), (30.0, 15.0), (40.0, 15.0)],
            "description": "Trailing car surges to 22 m/s; verifies ego vehicles resist being pushed forward into lead cars."
        },
        {
            "id": "internal_car3_emergency_brake",
            "name": "9. Internal Platoon Rupture: Car #3 Emergency Brakes to Halt",
            "car_number": 15,
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "human_car_index": 2,
            "braking_car_index": 2,
            "car_velocity_profiles": {
                2: [(0.0, 15.0), (5.0, 15.0), (8.5, 0.0), (40.0, 0.0)]
            },
            "description": "Car 2 (3rd car) hard brakes at t=5s; Cars 3–14 must stop safely; Car 1 must not panic-brake."
        },
        {
            "id": "mixed_heterogeneous_traffic",
            "name": "10. Mixed Heterogeneous Traffic (50% Auto / 50% Diverse Human IDM)",
            "car_number": 15,
            "v_des": 16.0,
            "type": "synthetic",
            "profile": [(0.0, 16.0), (10.0, 16.0), (14.0, 11.0), (22.0, 11.0), (26.0, 18.0), (34.0, 18.0), (38.0, 16.0), (50.0, 16.0)],
            "dur": 50.0,
            "mixed_traffic": True,
            # Assign alternating human profiles across 50% of platoon
            "human_assignments": {
                2: 'AGGRESSIVE',
                4: 'CAUTIOUS',
                6: 'NORMAL',
                8: 'AGGRESSIVE',
                10: 'CAUTIOUS',
                12: 'NORMAL',
                14: 'AGGRESSIVE'
            },
            "description": "50% Autonomous controllers interleaved with non-random Aggressive, Cautious, and Normal human drivers."
        },
        {
            "id": "highway_empirical_highd",
            "name": "11. Real-World Highway Cruise (highD / NGSIM Trajectory, 15 Cars)",
            "car_number": 15,
            "v_des": 24.0,
            "type": "csv",
            "csv_file": "highway_traffic.csv",
            "dur": 60.0,
            "description": "High-speed empirical highway trajectory testing tracking and comfort under authentic human speed flutter."
        }
    ]
    return scenarios

def evaluate_model_on_scenarios(model_path="td3_accel_agent.zip", scenarios=None, sensor_delay=0.0, sensor_noise=0.0):
    """
    Evaluates policy model across benchmark scenarios and computes comparative
    metrics against classical ACC, BCC, and ACC+BCC models.
    """
    resolved_model = resolve_model_path(model_path)
    if scenarios is None:
        scenarios = get_scenarios()
        
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {resolved_model} (Sensor Delay={sensor_delay}s, Noise={sensor_noise})")
    print(f"=======================================================")
    
    if not os.path.exists(resolved_model):
        print(f"Error: Model file {resolved_model} does not exist!")
        return None
        
    model = TD3.load(resolved_model)
    results = {}
    
    for sc in scenarios:
        params = dict(DEFAULT_SIM_PARAMS)
        params['v_des'] = sc['v_des']
        car_count = sc.get('car_number', params['car_number'])
        params['car_number'] = car_count
        
        init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]
        
        city_rl = RLCity()
        city_rl.init(*init_args, dt=params['dt'], model='ACC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        city_acc = City()
        city_acc.init(*init_args, dt=params['dt'], model='ACC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        city_bcc = City()
        city_bcc.init(*init_args, dt=params['dt'], model='BCC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        city_accbcc = City()
        city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC', sensor_delay=sensor_delay, sensor_noise=sensor_noise)
        
        follower_profile = sc.get('follower_profile', None)
        rear_cutout_time = sc.get('rear_cutout_time', None)
        human_assignments = sc.get('human_assignments', {})
        human_idx = sc.get('human_car_index', sc.get('braking_car_index', None))
        
        # Configure human drivers and profiles for all cities
        for c in [city_acc, city_bcc, city_accbcc, city_rl]:
            if follower_profile:
                c.follower_velocity_profile = list(follower_profile)
                c.cars[-1].is_human = True
                c.cars[-1].driver_type = 'HUMAN'
                c.cars[-1].color = '#ef4444'

            if human_idx is not None and human_idx < len(c.cars):
                c.cars[human_idx].is_human = True
                c.cars[human_idx].driver_type = 'HUMAN'
                c.cars[human_idx].color = '#ef4444'

            if human_assignments:
                for idx, p_key in human_assignments.items():
                    if idx < len(c.cars) and p_key in HUMAN_PROFILES:
                        c.cars[idx].is_human = True
                        c.cars[idx].human_profile = dict(HUMAN_PROFILES[p_key])
                        c.cars[idx].human_profile_name = p_key
                        c.cars[idx].driver_type = 'HUMAN'
                        c.cars[idx].color = HUMAN_PROFILES[p_key]['color']

        for i, car in enumerate(city_rl.cars):
            if i == 0 or getattr(car, 'is_human', False):
                car.is_rl = False
            else:
                car.is_rl = True
                car.rl_action = 0.0
                
        # Assign lead vehicle profile
        lead_profile = []
        if sc['type'] == 'csv':
            lead_profile = load_csv_profile(sc['csv_file'])
        elif sc['type'] == 'synthetic':
            lead_profile = sc['profile']
        elif sc['type'] == 'wave':
            lead_profile = [(t, 15.0 + 4.0 * np.sin(2 * np.pi * t / 15.0)) for t in np.arange(0, sc['dur'] + 1.0, 0.5)]
            
        if lead_profile:
            for c in [city_acc, city_bcc, city_accbcc, city_rl]:
                c.lead_velocity_profile = lead_profile

        if rear_cutout_time is not None:
            for c in [city_acc, city_bcc, city_accbcc, city_rl]:
                c.rear_cutout_time = rear_cutout_time

        car_velocity_profiles = sc.get('car_velocity_profiles', None)
        if car_velocity_profiles:
            for c in [city_acc, city_bcc, city_accbcc, city_rl]:
                c.car_velocity_profiles = dict(car_velocity_profiles)
                
        steps = int(sc['dur'] / params['dt'])
        collisions = 0
        min_gaps_seen = []
        gaps_rl = []
        rl_speeds = []
        road_len = city_rl.roads[0].length
        
        # Evaluated cars: followers that are autonomous (skip leader and scripted trailing if present)
        eval_indices = [i for i, car in enumerate(city_rl.cars) if i > 0 and (not follower_profile or i < len(city_rl.cars) - 1)]
        num_eval = len(eval_indices)
        
        follower_max_decel_rl = np.zeros(num_eval)
        follower_max_decel_acc = np.zeros(num_eval)
        follower_max_decel_bcc = np.zeros(num_eval)
        follower_max_decel_accbcc = np.zeros(num_eval)
        leader_max_decel = 0.0
        
        prev_acc_rl = np.zeros(num_eval)
        prev_acc_acc = np.zeros(num_eval)
        prev_acc_bcc = np.zeros(num_eval)
        prev_acc_accbcc = np.zeros(num_eval)
        
        jerks_rl, jerks_acc, jerks_bcc, jerks_accbcc = [], [], [], []
        gaps_acc, gaps_bcc, gaps_accbcc = [], [], []
        collisions_acc = 0
        collisions_bcc = 0
        collisions_accbcc = 0
        
        traces_rl = [{'gaps': [], 'speeds': [], 'vfs': [], 'vbs': []} for _ in eval_indices]
        traces_acc = [{'gaps': [], 'speeds': [], 'vfs': [], 'vbs': []} for _ in eval_indices]
        traces_bcc = [{'gaps': [], 'speeds': [], 'vfs': [], 'vbs': []} for _ in eval_indices]
        traces_accbcc = [{'gaps': [], 'speeds': [], 'vfs': [], 'vbs': []} for _ in eval_indices]
        
        for step in range(steps):
            city_acc.run(params['dt'])
            city_bcc.run(params['dt'])
            city_accbcc.run(params['dt'])
            
            obs_dim = getattr(getattr(model, 'observation_space', None), 'shape', [6])[0]
            obs_arr = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'], obs_dim=obs_dim)
            act, _ = model.predict(obs_arr, deterministic=True)
            for i, car in enumerate(city_rl.cars[1:]):
                if getattr(car, 'is_rl', False):
                    act_val = act[i][0] if (np.ndim(act) > 1 and i < len(act)) else (act[i] if i < len(act) else 0.0)
                    car.rl_action = float(act_val)
            city_rl.run(params['dt'])
            
            leader = city_rl.cars[0]
            leader_max_decel = max(leader_max_decel, max(0.0, -leader.acceleration))
            
            def step_metrics(city, prev_acc, follower_max_decel, jerks_list, gaps_list, traces_list):
                step_gaps = []
                cur_acc = []
                for k, i in enumerate(eval_indices):
                    car = city.cars[i]
                    cars_same = [c for c in city.cars if c.current_road == car.current_road and c != car and not getattr(c, 'has_exited', False)]
                    def gap_to(other):
                        g = (car.pos - other.pos) % road_len
                        return g if g > 0 else float('inf')
                    def gap_from(other):
                        g = (other.pos - car.pos) % road_len
                        return g if g > 0 else float('inf')
                    front = min(cars_same, key=gap_to, default=None)
                    back = min(cars_same, key=gap_from, default=None)
                    fg = (car.pos - front.pos - front.length) % road_len if front else road_len
                    vf = front.velocity if front else params['v_des']
                    vb = back.velocity if back else car.velocity
                    step_gaps.append(fg)
                    gaps_list.append(fg)
                    cur_acc.append(car.acceleration)
                    follower_max_decel[k] = max(follower_max_decel[k], max(0.0, -car.acceleration))
                    traces_list[k]['gaps'].append(fg)
                    traces_list[k]['speeds'].append(car.velocity)
                    traces_list[k]['vfs'].append(vf)
                    traces_list[k]['vbs'].append(vb)
                cur_acc = np.array(cur_acc)
                if step > 0:
                    jerks_list.extend(np.abs(cur_acc - prev_acc) / params['dt'])
                min_g = min(step_gaps) if step_gaps else road_len
                collided = (min_g < 0.5) or getattr(city, 'collided_this_step', False)
                return cur_acc, min_g, collided
            
            cur_acc_rl, min_g_rl, col_rl = step_metrics(city_rl, prev_acc_rl, follower_max_decel_rl, jerks_rl, gaps_rl, traces_rl)
            cur_acc_acc, min_g_acc, col_acc = step_metrics(city_acc, prev_acc_acc, follower_max_decel_acc, jerks_acc, gaps_acc, traces_acc)
            cur_acc_bcc, min_g_bcc, col_bcc = step_metrics(city_bcc, prev_acc_bcc, follower_max_decel_bcc, jerks_bcc, gaps_bcc, traces_bcc)
            cur_acc_accbcc, min_g_accbcc, col_accbcc = step_metrics(city_accbcc, prev_acc_accbcc, follower_max_decel_accbcc, jerks_accbcc, gaps_accbcc, traces_accbcc)
            
            prev_acc_rl = cur_acc_rl
            prev_acc_acc = cur_acc_acc
            prev_acc_bcc = cur_acc_bcc
            prev_acc_accbcc = cur_acc_accbcc
            
            min_gaps_seen.append(min_g_rl)
            if col_rl: collisions += 1
            if col_acc: collisions_acc += 1
            if col_bcc: collisions_bcc += 1
            if col_accbcc: collisions_accbcc += 1
            
            for i in eval_indices:
                rl_speeds.append(city_rl.cars[i].velocity)
                
        e_acc = sum(c.energy_used for c in city_acc.cars)
        e_bcc = sum(c.energy_used for c in city_bcc.cars)
        e_accbcc = sum(c.energy_used for c in city_accbcc.cars)
        e_rl = sum(c.energy_used for c in city_rl.cars)
        
        # String stability damping ratios
        l_decel = max(leader_max_decel, 0.1)
        damping_rl = float(np.max(follower_max_decel_rl) / l_decel) if len(follower_max_decel_rl) > 0 else 0.0
        damping_acc = float(np.max(follower_max_decel_acc) / l_decel) if len(follower_max_decel_acc) > 0 else 0.0
        damping_bcc = float(np.max(follower_max_decel_bcc) / l_decel) if len(follower_max_decel_bcc) > 0 else 0.0
        damping_accbcc = float(np.max(follower_max_decel_accbcc) / l_decel) if len(follower_max_decel_accbcc) > 0 else 0.0

        # Cut-In and X-factor evaluation across controllers
        def get_ctrl_cut_in(traces_list):
            ratios, t_breaches, civis = [], [], []
            for t_k in traces_list:
                m = compute_cut_in_metrics(
                    gaps=t_k['gaps'],
                    speeds=t_k['speeds'],
                    lead_speeds=t_k['vfs'],
                    rear_speeds=t_k['vbs'],
                    dt=params['dt'],
                    min_a=params['min_a'],
                    reaction_time=params['reaction_time'],
                    min_gap=params['min_gap'],
                    v_des=params['v_des']
                )
                ratios.append(m['max_x_ratio'])
                t_breaches.append(m['t_x_breach'])
                civis.append(m['civi_x'])
            max_r = float(max(ratios)) if ratios else 0.0
            max_t = float(max(t_breaches)) if t_breaches else 0.0
            mean_c = float(np.mean(civis)) if civis else 0.0
            return max_r, max_t, mean_c

        max_x_rl, t_x_rl, civi_rl = get_ctrl_cut_in(traces_rl)
        max_x_acc, t_x_acc, civi_acc = get_ctrl_cut_in(traces_acc)
        max_x_bcc, t_x_bcc, civi_bcc = get_ctrl_cut_in(traces_bcc)
        max_x_accbcc, t_x_accbcc, civi_accbcc = get_ctrl_cut_in(traces_accbcc)
        
        sc_res = {
            "name": sc['name'],
            "collisions_rl": collisions,
            "min_gap_rl": float(min(min_gaps_seen)) if min_gaps_seen else 0.0,
            "mean_gap_rl": float(np.mean(gaps_rl)) if gaps_rl else 0.0,
            "mean_speed_rl": float(np.mean(rl_speeds)) if rl_speeds else 0.0,
            "mean_jerk_rl": float(np.mean(jerks_rl)) if jerks_rl else 0.0,
            "mean_jerk_acc": float(np.mean(jerks_acc)) if jerks_acc else 0.0,
            "mean_jerk_bcc": float(np.mean(jerks_bcc)) if jerks_bcc else 0.0,
            "mean_jerk_accbcc": float(np.mean(jerks_accbcc)) if jerks_accbcc else 0.0,
            "energy_rl": float(e_rl),
            "energy_acc": float(e_acc),
            "energy_bcc": float(e_bcc),
            "energy_accbcc": float(e_accbcc),
            "damping_rl": damping_rl,
            "damping_acc": damping_acc,
            "damping_bcc": damping_bcc,
            "damping_accbcc": damping_accbcc,
            "max_x_rl": max_x_rl,
            "max_x_acc": max_x_acc,
            "max_x_bcc": max_x_bcc,
            "max_x_accbcc": max_x_accbcc,
            "t_x_rl": t_x_rl,
            "t_x_bcc": t_x_bcc,
            "civi_rl": civi_rl,
            "civi_bcc": civi_bcc,
        }
        results[sc['id']] = sc_res
        
        print(f"\n--- {sc['name']} ---")
        print(f"  Collisions: RL={collisions} | ACC={collisions_acc} | BCC={collisions_bcc} | ACC+BCC={collisions_accbcc}")
        print(f"  Min Gap:    RL={sc_res['min_gap_rl']:.2f}m | Mean Gap: RL={sc_res['mean_gap_rl']:.2f}m")
        print(f"  Mean Jerk:  RL={sc_res['mean_jerk_rl']:.2f} m/s^3 | ACC={sc_res['mean_jerk_acc']:.2f} | BCC={sc_res['mean_jerk_bcc']:.2f} | ACC+BCC={sc_res['mean_jerk_accbcc']:.2f}")
        print(f"  Energy:     RL={e_rl:.3f} kWh | ACC={e_acc:.3f} | BCC={e_bcc:.3f} | ACC+BCC={e_accbcc:.3f}")
        print(f"  Shockwave:  RL={damping_rl:.2f}x | ACC={damping_acc:.2f}x | BCC={damping_bcc:.2f}x | ACC+BCC={damping_accbcc:.2f}x")
        print(f"  Cut-In s_f/X: RL={max_x_rl:.2f}x (T={t_x_rl:.1f}s) | ACC={max_x_acc:.2f}x | BCC={max_x_bcc:.2f}x (T={t_x_bcc:.1f}s) | ACC+BCC={max_x_accbcc:.2f}x (T={t_x_accbcc:.1f}s)")
        
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model across benchmark scenarios.")
    parser.add_argument("model", nargs="?", default="td3_accel_agent.zip", help="Path to model file")
    parser.add_argument("--delay", type=float, default=0.0, help="Perception delay in seconds")
    parser.add_argument("--noise", type=float, default=0.0, help="Sensor noise std dev in meters")
    args = parser.parse_args()
    evaluate_model_on_scenarios(args.model, sensor_delay=args.delay, sensor_noise=args.noise)
