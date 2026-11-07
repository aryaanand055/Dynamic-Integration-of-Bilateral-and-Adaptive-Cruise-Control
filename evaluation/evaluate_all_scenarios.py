import os
import sys
import csv
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

def generate_random_speed_profile(t_start=5.0, dur=40.0, v_init=15.0, min_speed=4.0, max_speed=24.0, seed=None):
    """
    Generates a realistic multi-stage velocity profile where the vehicle maintains v_init
    for t_start seconds, then dynamically shifts between varied random speeds.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    profile = [(0.0, v_init), (t_start, v_init)]
    cur_t = t_start
    cur_v = v_init
    while cur_t < dur:
        new_v = round(float(rng.uniform(min_speed, max_speed)), 1)
        while abs(new_v - cur_v) < 4.0:
            new_v = round(float(rng.uniform(min_speed, max_speed)), 1)
        dv = abs(new_v - cur_v)
        transition_dur = round(max(2.0, min(4.5, dv / 3.0)), 1)
        trans_t = round(min(dur, cur_t + transition_dur), 2)
        profile.append((trans_t, new_v))
        hold_dur = round(float(rng.uniform(2.0, 4.0)), 1)
        hold_t = round(min(dur, trans_t + hold_dur), 2)
        if hold_t > trans_t:
            profile.append((hold_t, new_v))
        cur_t = hold_t
        cur_v = new_v
    if profile[-1][0] < dur:
        profile.append((dur, profile[-1][1]))
    return profile

def get_scenarios():
    """Returns the official 9 operational evaluation scenarios."""
    scenarios = [
        {
            "id": "steady_15",
            "name": "1. Steady Cruise & Equilibrium (v_des=15 m/s)",
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0
        },
        {
            "id": "stop_and_go_data",
            "name": "2. Urban Stop-and-Go (data.csv)",
            "v_des": 15.0,
            "type": "csv",
            "csv_file": "data.csv",
            "dur": 60.0
        },
        {
            "id": "speed_drop_data1",
            "name": "3. Speed Drop & Recovery (data1.csv)",
            "v_des": 15.0,
            "type": "csv",
            "csv_file": "data1.csv",
            "dur": 60.0
        },
        {
            "id": "full_stop_hold",
            "name": "4. Platoon Full Stop & Standstill Hold",
            "v_des": 15.0,
            "type": "synthetic",
            "profile": [(0.0, 15.0), (12.0, 15.0), (16.0, 0.0), (45.0, 0.0), (50.0, 12.0), (60.0, 15.0)],
            "dur": 60.0
        },
        {
            "id": "emergency_brake",
            "name": "5. Emergency Braking (-4.5 m/s^2 to 0)",
            "v_des": 20.0,
            "type": "synthetic",
            "profile": [(0.0, 20.0), (10.0, 20.0), (13.5, 0.0), (40.0, 0.0)],
            "dur": 40.0
        },
        {
            "id": "rear_vehicle_drop",
            "name": "6. Trailing Vehicle Exit / Drop-Off (Phantom Braking Stress)",
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "follower_profile": [(0.0, 15.0), (10.0, 15.0), (15.0, 5.0), (28.0, 5.0), (33.0, 15.0), (40.0, 15.0)]
        },
        {
            "id": "rear_tailgater",
            "name": "7. Aggressive Rear Tailgater (Forward Squeeze Stress)",
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "follower_profile": [(0.0, 15.0), (8.0, 15.0), (12.0, 22.0), (26.0, 22.0), (30.0, 15.0), (40.0, 15.0)]
        },
        {
            "id": "asymmetric_squeeze",
            "name": "8. Asymmetric Squeeze (Front Decel + Rear Push)",
            "v_des": 18.0,
            "type": "synthetic",
            "profile": [(0.0, 18.0), (10.0, 18.0), (14.0, 10.0), (26.0, 10.0), (30.0, 18.0), (40.0, 18.0)],
            "dur": 40.0,
            "follower_profile": [(0.0, 18.0), (10.0, 18.0), (14.0, 23.0), (26.0, 23.0), (30.0, 18.0), (40.0, 18.0)]
        },
        {
            "id": "string_stability_wave",
            "name": "9. Sinusoidal Wave String Stability (v=15+-4m/s)",
            "v_des": 15.0,
            "type": "wave",
            "dur": 60.0
        },
        {
            "id": "rear_accordion_wave",
            "name": "10. Rear Repeated Accel & Decel (Accordion Stress)",
            "v_des": 15.0,
            "type": "none",
            "dur": 45.0,
            "human_lead": False,
            "human_trailing": True,
            "reckless_lead": False,
            "reckless_trailing": True,
            "follower_profile": [
                (0.0, 15.0),
                (5.0, 15.0),
                (8.0, 22.0),
                (12.0, 8.0),
                (16.0, 22.0),
                (20.0, 8.0),
                (24.0, 22.0),
                (28.0, 8.0),
                (32.0, 22.0),
                (36.0, 8.0),
                (40.0, 15.0),
                (45.0, 15.0)
            ]
        },
        {
            "id": "emergency_brake_tailgater",
            "name": "11. Front Emergency Brake with Rear Tailgater (Severe Squeeze)",
            "v_des": 20.0,
            "type": "synthetic",
            "dur": 40.0,
            "human_lead": False,
            "human_trailing": True,
            "reckless_lead": False,
            "reckless_trailing": True,
            "profile": [
                (0.0, 20.0),
                (10.0, 20.0),
                (13.5, 0.0),
                (40.0, 0.0)
            ],
            "follower_profile": [
                (0.0, 20.0),
                (6.0, 20.0),
                (10.0, 24.0),
                (14.0, 24.0),
                (18.0, 0.0),
                (40.0, 0.0)
            ]
        },
        {
            "id": "rear_cutout_lane_change",
            "name": "12. Rear Vehicle Sudden Lane Change / Disappearance",
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "reckless_lead": False,
            "reckless_trailing": False,
            "rear_cutout_time": 12.0,
            "follower_profile": [
                (0.0, 15.0),
                (12.0, 15.0)
            ]
        },
        {
            "id": "reckless_lead_human",
            "name": "13. Reckless Human Lead Driver (Erratic Surges & Brake Checks)",
            "v_des": 18.0,
            "type": "synthetic",
            "dur": 45.0,
            "human_lead": True,
            "human_trailing": False,
            "reckless_lead": True,
            "reckless_trailing": False,
            "profile": [
                (0.0, 18.0),
                (5.0, 18.0),
                (8.0, 25.0),
                (10.0, 25.0),
                (12.5, 7.0),
                (16.0, 7.0),
                (20.0, 26.0),
                (23.0, 26.0),
                (25.5, 5.0),
                (29.0, 16.0),
                (33.0, 24.0),
                (36.0, 12.0),
                (40.0, 20.0),
                (45.0, 18.0)
            ]
        },
        {
            "id": "reckless_trailing_human",
            "name": "14. Reckless Human Trailing Driver (Aggressive Tailgate & Dive Surges)",
            "v_des": 18.0,
            "type": "none",
            "dur": 45.0,
            "human_lead": False,
            "human_trailing": True,
            "reckless_lead": False,
            "reckless_trailing": True,
            "follower_profile": [
                (0.0, 18.0),
                (5.0, 18.0),
                (7.5, 22.5),
                (9.5, 22.5),
                (12.0, 9.0),
                (15.0, 9.0),
                (18.0, 23.0),
                (20.5, 23.0),
                (23.5, 8.0),
                (26.5, 8.0),
                (29.5, 22.5),
                (33.0, 14.0),
                (37.0, 21.0),
                (41.0, 18.0),
                (45.0, 18.0)
            ]
        },
        {
            "id": "bilateral_reckless_human",
            "name": "15. Bilateral Reckless Human Drivers (Double Reckless Sandwich)",
            "v_des": 18.0,
            "type": "synthetic",
            "dur": 45.0,
            "human_lead": True,
            "human_trailing": True,
            "reckless_lead": True,
            "reckless_trailing": True,
            "profile": [
                (0.0, 18.0),
                (5.0, 18.0),
                (8.0, 25.0),
                (10.0, 25.0),
                (12.5, 6.0),
                (16.0, 6.0),
                (19.5, 24.0),
                (22.5, 24.0),
                (25.0, 5.0),
                (29.0, 5.0),
                (34.0, 20.0),
                (45.0, 18.0)
            ],
            "follower_profile": [
                (0.0, 18.0),
                (5.0, 18.0),
                (7.5, 22.0),
                (10.0, 22.0),
                (13.0, 6.0),
                (16.5, 6.0),
                (19.0, 23.0),
                (22.0, 23.0),
                (25.5, 5.0),
                (29.5, 5.0),
                (34.0, 20.0),
                (38.0, 18.0),
                (45.0, 18.0)
            ]
        },
        {
            "id": "third_car_random_speeds",
            "name": "16. 3rd Car Shifts to Random Speeds After 5s Constant Speed (Human-in-Platoon)",
            "v_des": 15.0,
            "type": "none",
            "dur": 40.0,
            "human_car_index": 2,
            "braking_car_index": 2,
            "reckless_car_index": 2,
            "reckless_lead": False,
            "reckless_trailing": False,
            "random_speed_shifts": True,
            "car_velocity_profiles": {
                2: generate_random_speed_profile(t_start=5.0, dur=40.0, v_init=15.0, seed=42)
            },
            "follower_profile": generate_random_speed_profile(t_start=5.0, dur=40.0, v_init=15.0, seed=42)
        }
    ]
    return scenarios

def evaluate_model_on_scenarios(model_path="td3_accel_agent.zip", scenarios=None):
    """
    Evaluates policy model across benchmark scenarios and computes comparative
    metrics against classical ACC, BCC, and ACC+BCC models.
    """
    resolved_model = resolve_model_path(model_path)
    if scenarios is None:
        scenarios = get_scenarios()
        
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {resolved_model}")
    print(f"=======================================================")
    
    if not os.path.exists(resolved_model):
        print(f"Error: Model file {resolved_model} does not exist!")
        return None
        
    model = TD3.load(resolved_model)
    results = {}
    
    for sc in scenarios:
        params = dict(DEFAULT_SIM_PARAMS)
        params['v_des'] = sc['v_des']
        
        init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]
        
        city_rl = RLCity()
        city_rl.init(*init_args, dt=params['dt'], model='ACC')
        city_acc = City()
        city_acc.init(*init_args, dt=params['dt'], model='ACC')
        city_bcc = City()
        city_bcc.init(*init_args, dt=params['dt'], model='BCC')
        city_accbcc = City()
        city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')
        
        follower_profile = sc.get('follower_profile', None)
        rear_cutout_time = sc.get('rear_cutout_time', None)
        
        for i, car in enumerate(city_rl.cars):
            if i == 0:
                car.is_rl = False
            elif i == len(city_rl.cars) - 1 and follower_profile:
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
                
        if follower_profile:
            for c in [city_acc, city_bcc, city_accbcc, city_rl]:
                c.follower_velocity_profile = list(follower_profile)

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
        
        num_eval = (len(city_rl.cars) - 2) if follower_profile else (len(city_rl.cars) - 1)
        eval_slice = slice(1, -1) if follower_profile else slice(1, None)
        
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
        
        for step in range(steps):
            city_acc.run(params['dt'])
            city_bcc.run(params['dt'])
            city_accbcc.run(params['dt'])
            
            obs_arr = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'])
            act, _ = model.predict(obs_arr, deterministic=True)
            for i, car in enumerate(city_rl.cars[1:]):
                if getattr(car, 'is_rl', False):
                    act_val = act[i][0] if (np.ndim(act) > 1 and i < len(act)) else (act[i] if i < len(act) else 0.0)
                    car.rl_action = float(act_val)
            city_rl.run(params['dt'])
            
            leader = city_rl.cars[0]
            leader_max_decel = max(leader_max_decel, max(0.0, -leader.acceleration))
            
            def step_metrics(city, prev_acc, follower_max_decel, jerks_list, gaps_list):
                step_gaps = []
                cur_acc = []
                eval_cars = city.cars[eval_slice]
                for i, car in enumerate(eval_cars):
                    cars_same = [c for c in city.cars if c.current_road == car.current_road and c != car and not getattr(c, 'has_exited', False)]
                    def gap_to(other):
                        g = (car.pos - other.pos) % road_len
                        return g if g > 0 else float('inf')
                    front = min(cars_same, key=gap_to, default=None)
                    fg = (car.pos - front.pos - front.length) % road_len if front else road_len
                    step_gaps.append(fg)
                    gaps_list.append(fg)
                    cur_acc.append(car.acceleration)
                    follower_max_decel[i] = max(follower_max_decel[i], max(0.0, -car.acceleration))
                cur_acc = np.array(cur_acc)
                if step > 0:
                    jerks_list.extend(np.abs(cur_acc - prev_acc) / params['dt'])
                min_g = min(step_gaps) if step_gaps else road_len
                collided = (min_g < 0.5) or getattr(city, 'collided_this_step', False)
                return cur_acc, min_g, collided
            
            cur_acc_rl, min_g_rl, col_rl = step_metrics(city_rl, prev_acc_rl, follower_max_decel_rl, jerks_rl, gaps_rl)
            cur_acc_acc, min_g_acc, col_acc = step_metrics(city_acc, prev_acc_acc, follower_max_decel_acc, jerks_acc, gaps_acc)
            cur_acc_bcc, min_g_bcc, col_bcc = step_metrics(city_bcc, prev_acc_bcc, follower_max_decel_bcc, jerks_bcc, gaps_bcc)
            cur_acc_accbcc, min_g_accbcc, col_accbcc = step_metrics(city_accbcc, prev_acc_accbcc, follower_max_decel_accbcc, jerks_accbcc, gaps_accbcc)
            
            prev_acc_rl = cur_acc_rl
            prev_acc_acc = cur_acc_acc
            prev_acc_bcc = cur_acc_bcc
            prev_acc_accbcc = cur_acc_accbcc
            
            min_gaps_seen.append(min_g_rl)
            if col_rl: collisions += 1
            if col_acc: collisions_acc += 1
            if col_bcc: collisions_bcc += 1
            if col_accbcc: collisions_accbcc += 1
            
            for car in city_rl.cars[eval_slice]:
                rl_speeds.append(car.velocity)
                
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
        
        sc_res = {
            "name": sc['name'],
            "collisions_rl": collisions,
            "min_gap_rl": float(min(min_gaps_seen)),
            "mean_gap_rl": float(np.mean(gaps_rl)),
            "mean_speed_rl": float(np.mean(rl_speeds)),
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
        }
        results[sc['id']] = sc_res
        
        print(f"\n--- {sc['name']} ---")
        print(f"  Collisions: RL={collisions} | ACC={collisions_acc} | BCC={collisions_bcc} | ACC+BCC={collisions_accbcc}")
        print(f"  Min Gap:    RL={sc_res['min_gap_rl']:.2f}m | Mean Gap: RL={sc_res['mean_gap_rl']:.2f}m")
        print(f"  Mean Jerk:  RL={sc_res['mean_jerk_rl']:.2f} m/s^3 | ACC={sc_res['mean_jerk_acc']:.2f} | BCC={sc_res['mean_jerk_bcc']:.2f} | ACC+BCC={sc_res['mean_jerk_accbcc']:.2f}")
        print(f"  Energy:     RL={e_rl:.3f} kWh | ACC={e_acc:.3f} | BCC={e_bcc:.3f} | ACC+BCC={e_accbcc:.3f}")
        print(f"  Shockwave:  RL={damping_rl:.2f}x | ACC={damping_acc:.2f}x | BCC={damping_bcc:.2f}x | ACC+BCC={damping_accbcc:.2f}x")
        
    return results

if __name__ == '__main__':
    target_model = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    evaluate_model_on_scenarios(target_model)
