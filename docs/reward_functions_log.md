# Reinforcement Learning Reward Functions Specification & Input/Output Log (`reward_functions_log.md`)

This document serves as the comprehensive, mathematical specification and operational log for all reward function terms, their input dependencies, mathematical formulations, operational gating conditions, and numerical output bounds in the autonomous vehicle platoon control environment ([env/accel_traffic_env.py](file:///d:/Desktop/Arya/Projects/ACC%20and%20BCC%20Python%20Code/Dynamic-Integration-of-Bilateral-and-Adaptive-Cruise-Control/env/accel_traffic_env.py)).

---

## 1. Input Features & State Variables Taxonomy

Every reward component is computed per follower vehicle (index `i = 1 .. N-1`) at every simulation step (`dt = 0.1 s`). Below are the primary inputs extracted from the simulation and vehicle state:

| Input Variable | Notation | Unit | Source / Physical Meaning |
| :--- | :--- | :--- | :--- |
| `car.velocity` | v | m/s | Ego vehicle current forward velocity |
| `car.acceleration` | a | m/s² | Ego vehicle current commanded acceleration |
| `self.prev_accel[car_idx]` | a_prev | m/s² | Ego vehicle acceleration at previous time step t - dt |
| `delta_a` | Δa | m/s² | Acceleration step change: `a - a_prev` |
| `self.action_deltas[obs_idx]` | Δu | m/s² | Policy raw neural network action change: `|u_t - u_{t-1}|` |
| `front_car.velocity` | v_lead | m/s | Velocity of vehicle directly ahead (0.0 if none) |
| `front_car.acceleration` | a_lead | m/s² | Commanded acceleration of vehicle directly ahead |
| `front_gap` | s_f | m | Bumper-to-bumper distance to vehicle ahead: `(pos_ego - pos_lead - len_lead) mod L` |
| `v_rel_f` | Δv_f | m/s | Relative velocity to front car: `v_lead - v_ego` |
| `back_car.velocity` | v_rear | m/s | Velocity of vehicle directly behind (0.0 if none or lead car) |
| `back_gap_raw` | s_b | m | Bumper-to-bumper distance from vehicle behind: `(pos_rear - pos_ego - len_ego) mod L` |
| `min_dis` | s_0 | m | Standstill minimum safety headway jam buffer (default: 6.0 m) |
| `reaction_time` | T | s | Time headway constant (default: 0.8 s) |
| `v_des` | v_des | m/s | Platoon cruise target speed (GUI setting, default: 15.0 m/s) |
| `max_a` | a_max | m/s² | Maximum physical acceleration bound (+3.0 m/s²) |
| `min_a` | a_min | m/s² | Maximum physical braking deceleration bound (-5.0 m/s²) |
| `has_collided` | col | bool | Physical collision flag (`just_collided`, `front_gap < 0.5 m`, or active contact) |

---

## 2. Derived Kinematic State Variables

Before evaluating individual reward terms, several derived kinematic variables are calculated:

### 2.1 Desired Headway Distance
```text
desired_gap = min_dis + car.velocity * reaction_time
            = s_0 + v * T
```

### 2.2 Dynamic Kinematic Stopping Cushion
When rapidly closing on a vehicle (`v_rel_f < -0.5 m/s`), the stopping distance buffer expands quadratically based on the closing speed assuming comfortable braking deceleration (`a_comf = 3.5 m/s²`):
```text
d_kinematic = (v_rel_f)² / (2 × 3.5)   if v_rel_f < -0.5 m/s
            = 0.0                      otherwise

effective_desired_gap = desired_gap + d_kinematic
```

### 2.3 Normalized Headway Gap Error
```text
e_f = (front_gap - effective_desired_gap) / max(effective_desired_gap, 1.0)
```

### 2.4 Continuous Adaptive Target Velocity
To prevent step discontinuities when gaps widen or close, target velocity smoothly blends from lead vehicle speed to open-road cruise speed:
```text
if front_car and front_car.velocity < v_des:
    blend_start = effective_desired_gap
    blend_end = max(effective_desired_gap × 2.5, 25.0)
    if front_gap <= blend_start:
        target_v = front_car.velocity
    elif front_gap >= blend_end:
        target_v = v_des
    else:
        w = (front_gap - blend_start) / (blend_end - blend_start)
        target_v = (1.0 - w) × front_car.velocity + w × v_des
else:
    target_v = v_des

v_err = (car.velocity - target_v) / max(target_v, 1.0)
```

---

## 3. Detailed Specification of All 24 Reward Function Terms

### 🔴 Group 1: Collision Avoidance & Safety Boundary

#### Term 1.1: Collision Penalty (`r_collision`)
* **Inputs**: `has_collided`, `self.in_collision[obs_idx]`
* **Condition**: Active whenever bumper overlap or contact is detected (`front_gap < 0.5 m` or physical contact flag).
* **Mathematical Formula**:
  ```text
  if has_collided:
      r_collision = -40.0
      if not was_in_collision_previous_step:
          r_collision -= 40.0    # -80.0 total penalty on collision initiation
  else:
      r_collision = 0.0
  ```
* **Output Range**: `[-80.0, 0.0]`
* **Purpose**: Severe deterrent against physical vehicle contact.

#### Term 1.2: Progressive Danger Boundary Penalty (`r_danger`)
* **Inputs**: `front_gap`, `desired_gap`
* **Condition**: Active when `front_gap < 4.5 m` or `front_gap < 0.65 × desired_gap`.
* **Mathematical Formula**:
  ```text
  if front_gap < 4.5:
      r_danger = -15.0 × [ (4.5 - max(front_gap, 0.0)) / 4.5 ]² - 5.0
  elif front_gap < 0.65 × desired_gap:
      r_danger = -4.0 × [ (0.65 × desired_gap - front_gap) / (0.65 × desired_gap) ]²
  else:
      r_danger = 0.0
  ```
* **Output Range**: `[-20.0, 0.0]`
* **Purpose**: Creates an insurmountable potential wall preventing encroachment within 4.5 meters.

#### Term 1.3: Emergency Braking Reward & Under-Braking Penalty (`r_safety_brake`)
* **Inputs**: `front_gap`, `effective_desired_gap`, `v_rel_f`, `car.acceleration`, `min_dis`, `min_a`
* **Condition**: Active during collision danger: `front_gap < 1.1 × effective_desired_gap` and `v_rel_f < -1.0 m/s`.
* **Mathematical Formula**:
  ```text
  # Required kinematic stopping deceleration:
  a_req = min(-1.5, - (v_rel_f)² / (2 × max(front_gap - min_dis + 0.8, 0.5)))
  a_req = max(-5.0, a_req)

  if car.acceleration <= a_req + 0.4:
      # Reward decisive braking matching or exceeding required stopping rate
      r_safety_brake = +3.5 × exp(- [ (car.acceleration - a_req) / 1.0 ]²)
  else:
      # Heavily penalize hesitation or under-braking during danger
      r_safety_brake = -6.0 × min( |car.acceleration - a_req| / 1.5, 4.0 )
  ```
* **Output Range**: `[-6.0, +3.5]`
* **Purpose**: Overrides passivity; explicitly trains the policy to hit hard deceleration (-3.0 to -5.0 m/s²) during rapid closing.

---

### 🟢 Group 2: Headway Tracking & Gap Regulation

#### Term 2.1: Safe Headway Bell-Curve Reward (`r_headway`)
* **Inputs**: `e_f`, `is_tail_car`
* **Condition**: Evaluated continuously across all time steps.
* **Mathematical Formula**:
  ```text
  r_headway = 2.5 × exp(- (e_f / 0.25)²)
  if is_tail_car:
      r_headway *= 1.25    # Tail vehicle compensation
  ```
* **Output Range**: `[0.0, +3.125]`
* **Purpose**: Gaussian sweet spot reward maximizing at exact desired time-headway (`e_f = 0.0`).

#### Term 2.2: Anti-Lag Headway Penalty (`r_lag`)
* **Inputs**: `e_f`
* **Condition**: Active when falling significantly behind: `e_f > 0.35`.
* **Mathematical Formula**:
  ```text
  if e_f > 0.35:
      r_lag = -1.5 × min(e_f - 0.35, 3.0)
  else:
      r_lag = 0.0
  ```
* **Output Range**: `[-4.5, 0.0]`
* **Purpose**: Provides a linear restoring pull penalizing excessive gaps.

#### Term 2.3: Open-Gap Catch-Up Acceleration Reward (`r_catchup`)
* **Inputs**: `e_f`, `car.acceleration`
* **Condition**: Active when open headway is available ahead (`e_f > 0.35`).
* **Mathematical Formula**:
  ```text
  if e_f > 0.35:
      if car.acceleration < 0.0:
          r_catchup = -4.0 × min(|car.acceleration|, 3.0)    # Penalize braking into open gap
      elif car.acceleration <= 0.2:
          r_catchup = -1.5                                   # Penalize sluggish hesitation
      else:
          r_catchup = +2.0 × min(car.acceleration / 1.5, 1.5) # Reward positive forward acceleration
  else:
      r_catchup = 0.0
  ```
* **Output Range**: `[-4.0, +2.25]`
* **Purpose**: Eliminates open-road phantom braking; motivates positive acceleration to re-join the platoon.

#### Term 2.4: Open-Road Hesitation Penalty (`r_open_road_hesitation`)
* **Inputs**: `front_gap`, `effective_desired_gap`, `v_rel_f`, `car.velocity`, `v_des`, `car.acceleration`
* **Condition**: `front_gap > effective_desired_gap + 2.0`, `v_rel_f >= -0.2`, `car.velocity < v_des`.
* **Mathematical Formula**:
  ```text
  if car.acceleration < 0.0:
      r_open_road_hesitation = -4.0 × min(|car.acceleration| / 1.5, 3.0)
  else:
      r_open_road_hesitation = 0.0
  ```
* **Output Range**: `[-4.0, 0.0]`
* **Purpose**: Strictly suppresses phantom braking when driving under cruise speed on empty road.

---

### 🔵 Group 3: Velocity Tracking & Smooth Progression

#### Term 3.1: Relative Velocity Matching (`r_v_match`)
* **Inputs**: `v_rel_f`
* **Condition**: Evaluated continuously.
* **Mathematical Formula**:
  ```text
  r_v_match = 1.5 × exp(- (v_rel_f / 2.0)²)
  ```
* **Output Range**: `[0.0, +1.5]`
* **Purpose**: Rewards zero relative velocity (`v_ego = v_lead`), damping velocity differences.

#### Term 3.2: Asymmetric Closing Damping (`r_closing_damp`)
* **Inputs**: `v_rel_f`, `front_gap`, `effective_desired_gap`, `car.acceleration`, `front_car.acceleration`
* **Condition**: `v_rel_f < -1.0 m/s` and `front_gap < 2.0 × effective_desired_gap`.
* **Mathematical Formula**:
  ```text
  if front_car and car.acceleration > front_car.acceleration:
      r_closing_damp = -2.5 × min([ (-v_rel_f - 1.0) / 3.0 ]², 4.0)
  else:
      r_closing_damp = 0.0
  ```
* **Output Range**: `[-10.0, 0.0]`
* **Purpose**: Penalizes failing to decelerate when rapidly overtaking vehicle ahead.

#### Term 3.3: Target Speed Matching (`r_speed`)
* **Inputs**: `v_err`
* **Condition**: Evaluated continuously.
* **Mathematical Formula**:
  ```text
  r_speed = 2.0 × exp(- (v_err / 0.15)²)
  ```
* **Output Range**: `[0.0, +2.0]`
* **Purpose**: Tight Gaussian reward maximizing when vehicle velocity exactly tracks target speed.

#### Term 3.4: Open-Road Linear Speed Error Penalty (`r_speed_linear`)
* **Inputs**: `front_gap`, `effective_desired_gap`, `v_err`
* **Condition**: `front_gap >= effective_desired_gap`.
* **Mathematical Formula**:
  ```text
  r_speed_linear = -1.5 × |v_err|
  ```
* **Output Range**: `[-3.0, 0.0]`
* **Purpose**: Linear gradient ensuring vehicle cruises at full `v_des` on open highway.

#### Term 3.5: Anti-Stall Crawling Penalty (`r_stall`)
* **Inputs**: `car.velocity`, `target_v`, `front_gap`, `effective_desired_gap`
* **Condition**: `car.velocity < 3.0 m/s` and `target_v > 8.0 m/s` and `front_gap > effective_desired_gap`.
* **Mathematical Formula**:
  ```text
  r_stall = -4.0 × (1.0 - car.velocity / 3.0)
  ```
* **Output Range**: `[-4.0, 0.0]`
* **Purpose**: Eliminates the stationary crawling local optimum when highway is open ahead.

---

### 🟣 Group 4: String Stability & Wave Damping

#### Term 4.1: Downstream Wave Attenuation (`r_string`)
* **Inputs**: `front_car.acceleration`, `car.acceleration`, `e_f`
* **Condition**: Active strictly within platoon string equilibrium: `|e_f| <= 0.35`.
* **Mathematical Formula**:
  ```text
  if front_car.acceleration < -0.1 and car.acceleration < front_car.acceleration:
      # Over-braking relative to leader (amplifying deceleration wave)
      r_string = -6.0 × min((front_car.acceleration - car.acceleration) / 1.0, 4.0)
  elif front_car.acceleration > 0.1 and car.acceleration > front_car.acceleration:
      # Over-accelerating relative to leader (amplifying acceleration wave)
      r_string = -3.0 × min((car.acceleration - front_car.acceleration) / 1.0, 3.0)
  else:
      r_string = 0.0
  ```
* **Output Range**: `[-24.0, 0.0]`
* **Purpose**: Mathematically enforces string stability condition (`|a_follower| <= |a_leader|`), dissipating stop-and-go waves.

#### Term 4.2: Anti-Panic Braking Penalty (`r_panic_brake`)
* **Inputs**: `front_gap`, `effective_desired_gap`, `car.acceleration`
* **Condition**: `front_gap > effective_desired_gap` and `car.acceleration < -1.8 m/s²`.
* **Mathematical Formula**:
  ```text
  r_panic_brake = -3.5 × min([ (-car.acceleration - 1.8) / 1.5 ]², 4.0)
  ```
* **Output Range**: `[-14.0, 0.0]`
* **Purpose**: Prohibits unprovoked harsh braking when safe headway is maintained.

---

### 🟡 Group 5: Standstill Auto-Hold, Queue Packing & Drive-Off Launch

#### Term 5.1: Standstill Queue Headway Spacing (`r_standstill_gap`)
* **Inputs**: `front_car.velocity`, `car.velocity`, `front_gap`, `min_dis`
* **Condition**: `front_car.velocity <= 0.2 m/s` and `car.velocity <= 1.0 m/s`.
* **Mathematical Formula**:
  ```text
  if front_gap >= (min_dis - 0.5) and front_gap <= (min_dis + 1.2):
      r_standstill_gap = +2.5                                # Ideal 5.5m - 7.2m queue buffer
  elif front_gap > (min_dis + 1.2):
      gap_excess = front_gap - (min_dis + 1.2)
      r_standstill_gap = -2.0 × min((gap_excess / 5.0)², 2.5) # Lagging behind stopped queue
  elif front_gap < (min_dis - 0.5):
      gap_deficit = (min_dis - 0.5) - front_gap
      r_standstill_gap = -4.0 × min((gap_deficit / 2.0)², 4.0) # Dangerous compression (< 5.5m)
  ```
* **Output Range**: `[-4.0, +2.5]`
* **Purpose**: Prevents queue accordion compression while discouraging stopping too far behind traffic.

#### Term 5.2: Premature Standstill Stop Penalty (`r_premature_stop`)
* **Inputs**: `car.velocity`, `front_gap`, `min_dis`
* **Condition**: `car.velocity < 0.2 m/s` and `front_gap > min_dis + 2.0 m`.
* **Mathematical Formula**:
  ```text
  r_premature_stop = -2.5 × min((front_gap - (min_dis + 2.0)) / min_dis, 2.0)
  ```
* **Output Range**: `[-5.0, 0.0]`
* **Purpose**: Prohibits stopping 15–20 meters short of a stopped queue.

#### Term 5.3: Stationary Auto-Hold Reward (`r_standstill_hold`)
* **Inputs**: `front_car.velocity`, `car.velocity`, `front_gap`, `effective_desired_gap`, `car.acceleration`
* **Condition**: Both lead and ego at standstill: `front_car.velocity <= 0.25 m/s`, `car.velocity <= 0.25 m/s`, `front_gap <= 1.5 × effective_desired_gap`.
* **Mathematical Formula**:
  ```text
  r_standstill_hold = +2.5 × exp(- (car.acceleration / 0.25)²)
  if car.acceleration > 0.1:
      r_standstill_hold -= 4.0 × (car.acceleration / 1.0)  # Penalize forward creeping into queue
  elif car.acceleration < -0.2:
      r_standstill_hold -= 1.5 × |car.acceleration|        # Penalize harsh brake chattering
  ```
* **Output Range**: `[-4.0, +2.5]`
* **Purpose**: Establishes zero-acceleration auto-hold brake state (`a = 0.0 m/s²`) at zero speed.

#### Term 5.4: Drive-Off Launch Reward (`r_launch`)
* **Inputs**: `car.velocity`, `front_car.velocity`, `front_gap`, `min_dis`, `v_rel_f`, `car.acceleration`
* **Condition**: Ego at rest (`v <= 2.0 m/s`) and traffic ahead departs (`v_lead > 0.35 m/s` or `front_gap > min_dis + 1.0 m`) with safe buffer (`front_gap >= 3.5 m`, `v_rel_f >= 0.0 m/s`).
* **Mathematical Formula**:
  ```text
  if car.acceleration < 0.0:
      r_launch = -5.0 × min(|car.acceleration|, 3.0)    # Strictly penalize braking during launch
  elif car.acceleration <= 0.2:
      r_launch = -3.0                                   # Penalize dead-band hesitation
  else:
      target_launch_a = 1.6
      r_launch = +4.0 × exp(- [ (car.acceleration - target_launch_a) / 0.7 ]²)
  ```
* **Output Range**: `[-5.0, +4.0]`
* **Purpose**: Provides decisive native throttle command (`+1.0` to `+2.2 m/s²`) to drive off from standstill.

---

### 🟠 Group 6: Bilateral Control & Rear Awareness (BCC)

#### Term 6.1: Rear Tailgating Protection (`r_bcc_tailgate`)
* **Inputs**: `back_gap_raw`, `desired_back_gap`
* **Condition**: Rear vehicle present (`back_car != lead_car`) and encroaching: `e_b < -0.15`.
* **Mathematical Formula**:
  ```text
  e_b = (back_gap_raw - desired_back_gap) / max(desired_back_gap, 1.0)
  if e_b < -0.15:
      r_bcc_tailgate = -2.0 × min(|e_b + 0.15|, 2.5)
  else:
      r_bcc_tailgate = 0.0
  ```
* **Output Range**: `[-5.0, 0.0]`
* **Purpose**: Alerts ego vehicle if rear vehicle is tailgating within safety margin.

#### Term 6.2: Bilateral Velocity Damping (`r_bcc_damp`)
* **Inputs**: `car.velocity`, `back_car.velocity`
* **Condition**: Active strictly in cruising motion: `car.velocity > 3.5 m/s` and `back_car.velocity > 2.5 m/s`.
* **Mathematical Formula**:
  ```text
  r_bcc_damp = -1.0 × min([ (back_car.velocity - car.velocity) / 2.0 ]², 2.0)
  ```
* **Output Range**: `[-2.0, 0.0]`
* **Purpose**: Synchronizes ego speed with trailing vehicle speed, establishing bidirectional string damping.

#### Term 6.3: Bilateral Midpoint Velocity Consensus (`r_bcc_consensus`)
* **Inputs**: `front_car.velocity`, `back_car.velocity`, `car.velocity`
* **Condition**: Active strictly in cruising motion: `car.velocity > 3.5 m/s` and `back_car.velocity > 2.5 m/s`.
* **Mathematical Formula**:
  ```text
  v_bilateral_target = 0.5 × (front_car.velocity + back_car.velocity)
  r_bcc_consensus = +1.5 × exp(- [ (car.velocity - v_bilateral_target) / 1.5 ]²)
  ```
* **Output Range**: `[0.0, +1.5]`
* **Purpose**: Promotes smooth midpoint speed consensus across the platoon lattice.

---

### ⚪ Group 7: Passenger Comfort, Jerk & Equilibrium

#### Term 7.1: Equilibrium Cruising Bias (`r_accel`)
* **Inputs**: `car.acceleration`, gating flags (`is_launching`, `is_catching_up`, `is_emergency`)
* **Condition**: Active ONLY during steady cruising equilibrium or standstill hold.
* **Mathematical Formula**:
  ```text
  if not (is_launching or is_catching_up or is_emergency):
      r_accel = +1.0 × exp(- (car.acceleration / 0.35)²) - 0.2 × (car.acceleration / 2.0)²
  else:
      r_accel = 0.0
  ```
* **Output Range**: `[-0.2, +1.0]`
* **Purpose**: Centers acceleration on zero during steady state without penalizing necessary maneuvers.

#### Term 7.2: Passenger Jerk Minimization (`r_jerk`)
* **Inputs**: `delta_a = a_t - a_{t-1}`, `is_emergency`, `car.acceleration`
* **Condition**: Continuous, with safety scaling.
* **Mathematical Formula**:
  ```text
  r_jerk = -0.6 × (delta_a / 1.0)²
  if is_emergency and car.acceleration < -1.0:
      r_jerk *= 0.15    # 85% attenuation during emergency stops
  ```
* **Output Range**: `[-15.0, 0.0]`
* **Purpose**: Eradicates jagged high-frequency acceleration spikes for a luxury-grade smooth ride.

#### Term 7.3: Policy Action-Rate Regularization (`r_act_delta`)
* **Inputs**: `act_delta = |u_t - u_{t-1}|`, `is_emergency`, `car.acceleration`
* **Condition**: Continuous, with safety scaling.
* **Mathematical Formula**:
  ```text
  r_act_delta = -1.5 × (act_delta / 1.0)²
  if is_emergency and car.acceleration < -1.0:
      r_act_delta *= 0.15    # 85% attenuation during emergency stops
  ```
* **Output Range**: `[-37.5, 0.0]`
* **Purpose**: Directly penalizes high-frequency neural network policy chattering.

#### Term 7.4: Zero-Drift Cruising Attractor Lock (`r_cruise_lock`)
* **Inputs**: `e_f`, `v_rel_f`, `v_err`, `car.acceleration`
* **Condition**: `|e_f| < 0.15` and `|v_rel_f| < 0.8 m/s` and `|v_err| < 0.10`.
* **Mathematical Formula**:
  ```text
  r_cruise_lock = +3.5 × exp(- (car.acceleration / 0.15)²)
  ```
* **Output Range**: `[0.0, +3.5]`
* **Purpose**: Locks acceleration to `+0.00000000 m/s²` when platoon travels in perfect equilibrium.

---

### 🔋 Group 8: Energy Efficiency

#### Term 8.1: Traction Power Proxy (`r_energy`)
* **Inputs**: `car.acceleration`, `car.velocity`, `max_a`, `v_des`
* **Condition**: Evaluated continuously.
* **Mathematical Formula**:
  ```text
  p_pos = max(0.0, car.acceleration × car.velocity)
  r_energy = -0.2 × [ p_pos / max(max_a × v_des, 1.0) ]
  ```
* **Output Range**: `[-0.2, 0.0]`
* **Purpose**: Penalizes wasteful high-throttle acceleration at high speeds, minimizing kWh energy consumption.

---

## 4. Total Composite Reward Equation: The 8 Modular Reward Groups

The 24 detailed micro-terms are grouped into **8 canonical, orthogonal control objectives**:

```text
R_i = R_safety + R_headway + R_velocity + R_string + R_standstill + R_bilateral + R_comfort + R_energy
```

Where each macro-reward represents a distinct operational objective of the platoon controller:

1. **`R_safety` (Collision Avoidance & Physical Protection)**:
   $$R_{\text{safety}} = r_{\text{collision}} + r_{\text{danger}} + r_{\text{safety\_brake}}$$
2. **`R_headway` (Safe Headway & Longitudinal Gap Regulation)**:
   $$R_{\text{headway}} = r_{\text{headway}} + r_{\text{lag}} + r_{\text{catchup}} + r_{\text{open\_road\_hesitation}}$$
3. **`R_velocity` (Velocity Tracking & Smooth Progression)**:
   $$R_{\text{velocity}} = r_{\text{v\_match}} + r_{\text{closing\_damp}} + r_{\text{speed}} + r_{\text{speed\_linear}} + r_{\text{stall}}$$
4. **`R_string` (String Stability & Downstream Shockwave Damping)**:
   $$R_{\text{string}} = r_{\text{string}} + r_{\text{panic\_brake}}$$
5. **`R_standstill` (Standstill Auto-Hold, Queue Packing & Drive-Off Launch)**:
   $$R_{\text{standstill}} = r_{\text{standstill\_gap}} + r_{\text{premature\_stop}} + r_{\text{standstill\_hold}} + r_{\text{launch}}$$
6. **`R_bilateral` (Bilateral Coupling & Rear Platoon Lattice Awareness)**:
   $$R_{\text{bilateral}} = r_{\text{bcc\_tailgate}} + r_{\text{bcc\_damp}} + r_{\text{bcc\_consensus}} + r_{\text{bcc\_tension}}$$
7. **`R_comfort` (Passenger Comfort, Jerk Minimization & Zero-Acc Attractor)**:
   $$R_{\text{comfort}} = r_{\text{accel}} + r_{\text{jerk}} + r_{\text{act\_delta}} + r_{\text{cruise\_lock}}$$
8. **`R_energy` (Energy Efficiency & Eco Dynamics)**:
   $$R_{\text{energy}} = r_{\text{energy}} + r_{\text{eco\_accel}} + r_{\text{coast}}$$

In training, `R_i` passes through `VecNormalize(norm_reward=True, clip_reward=35.0)` to maintain unit variance stability in TD3 Critic updates.

---

## 5. Master Summary Table: All 24 Reward Components with Exact Numerical Weights

| Category | Component | Primary Input Variables | Exact Weight / Reward Scale | Output Range | Typical Value (Cruising) | Typical Value (Braking) | Typical Value (Standstill) | Typical Value (Launch) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Safety** | `r_collision` | `has_collided` | **W = -40.0 / -80.0** | `[-80.0, 0.0]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_danger` | `front_gap`, `desired_gap` | **W = -15.0 / -4.0** | `[-20.0, 0.0]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_safety_brake` | `front_gap`, `v_rel_f`, `a` | **W_pos = +3.50, W_neg = -6.00** | `[-6.00, +3.50]` | `0.00` | **`+3.37`** / **`-5.20`** | `0.00` | `0.00` |
| **Headway** | `r_headway` | `front_gap`, `desired_gap` | **W = +2.50 (×1.25 tail)** | `[0.00, +3.125]` | **`+2.50`** | `+2.40` | **`+2.50`** | `+2.10` |
| | `r_lag` | `e_f` | **W = -1.50** | `[-4.50, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_catchup` | `e_f`, `a` | **W_pos = +2.00, W_neg = -4.00** | `[-4.00, +2.25]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_open_road_hesitation`| `front_gap`, `v_rel_f`, `v`, `a`| **W = -4.00** | `[-4.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| **Velocity** | `r_v_match` | `v_rel_f` | **W = +1.50** | `[0.00, +1.50]` | **`+1.50`** | `+0.58` | **`+1.50`** | `+0.41` |
| | `r_closing_damp` | `v_rel_f`, `front_gap`, `a`, `a_lead`| **W = -2.50** | `[-10.00, 0.00]` | `0.00` | `-0.25` | `0.00` | `0.00` |
| | `r_speed` | `v`, `target_v` | **W = +2.00** | `[0.00, +2.00]` | **`+2.00`** | `+0.70` | **`+1.97`** | `0.00` |
| | `r_speed_linear` | `v`, `target_v`, `front_gap` | **W = -1.50** | `[-3.00, 0.00]` | `0.00` | `0.00` | `0.00` | `-1.12` |
| | `r_stall` | `v`, `target_v`, `front_gap` | **W = -4.00** | `[-4.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| **String** | `r_string` | `a`, `a_lead`, `e_f` | **W_brake = -6.00, W_acc = -3.00** | `[-24.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_panic_brake` | `front_gap`, `a` | **W = -3.50** | `[-14.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| **Standstill**| `r_standstill_gap` | `front_gap`, `min_dis`, `v_lead`, `v` | **W_pos = +2.50, W_neg = -4.00** | `[-4.00, +2.50]` | `0.00` | `0.00` | **`+2.50`** | `0.00` |
| | `r_premature_stop` | `front_gap`, `min_dis`, `v` | **W = -2.50** | `[-5.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_standstill_hold` | `v_lead`, `v`, `front_gap`, `a` | **W_pos = +2.50, W_creep = -4.00** | `[-4.00, +2.50]` | `0.00` | `0.00` | **`+2.41`** | `0.00` |
| | `r_launch` | `v_lead`, `v`, `front_gap`, `a` | **W_pos = +4.00, W_hes = -3.00** | `[-5.00, +4.00]` | `0.00` | `0.00` | `0.00` | **`+3.98`** |
| **BCC** | `r_bcc_tailgate` | `back_gap_raw`, `desired_back_gap`| **W = -2.00** | `[-5.00, 0.00]` | `0.00` | `0.00` | `0.00` | `0.00` |
| | `r_bcc_damp` | `v`, `v_rear` | **W = -1.00** | `[-2.00, 0.00]` | `0.00` | `-0.02` | `0.00` | `0.00` |
| | `r_bcc_consensus` | `v_lead`, `v_rear`, `v` | **W = +1.50** | `[0.00, +1.50]` | **`+1.50`** | `+1.10` | `0.00` | `0.00` |
| **Comfort** | `r_accel` | `a` | **W_pos = +1.00, W_neg = -0.20** | `[-0.20, +1.00]` | **`+1.00`** | `0.00` | **`+0.98`** | `0.00` |
| | `r_jerk` | `Δa = a_t - a_{t-1}` | **W = -0.60 (×0.15 emerg)** | `[-15.00, 0.00]` | `-0.00` | `-0.02` | `-0.00` | `-0.05` |
| | `r_act_delta` | `Δu = \|u_t - u_{t-1}\|` | **W = -1.50 (×0.15 emerg)** | `[-37.50, 0.00]` | `-0.00` | `-0.16` | `-0.00` | `-0.32` |
| | `r_cruise_lock` | `e_f`, `v_rel_f`, `v_err`, `a` | **W = +3.50** | `[0.00, +3.50]` | **`+3.50`** | `0.00` | **`+3.16`** | `0.00` |
| **Energy** | `r_energy` | `a`, `v` | **W = -0.20** | `[-0.20, 0.00]` | `-0.00` | `-0.00` | `-0.00` | `-0.01` |
| **TOTAL STEP** | `TOTAL_R` | All 24 Active Components | **Sum of Components** | `[-100.0, +25.0]` | **`+12.000`** | **`+7.694`** | **`+15.025`** | **`+4.998`** |

---

## 6. Generated Runtime Log Artifacts

Two runtime log files are continuously available in the project root:

1. **[reward_breakdown_log.txt](file:///d:/Desktop/Arya/Projects/ACC%20and%20BCC%20Python%20Code/Dynamic-Integration-of-Bilateral-and-Adaptive-Cruise-Control/reward_breakdown_log.txt)**:
   - Full human-readable runtime execution log across 5 operational phases.
   - Comprehensive tabular listing displaying all input features and **all 24 individual reward component values** at every 0.1s time step.
   - In-depth mathematical audit snapshots for each driving regime displaying exact formulas, inputs, and calculated reward values side-by-side.

2. **[reward_telemetry.csv](file:///d:/Desktop/Arya/Projects/ACC%20and%20BCC%20Python%20Code/Dynamic-Integration-of-Bilateral-and-Adaptive-Cruise-Control/reward_telemetry.csv)**:
   - Full structured dataset (3,000+ entries) logging all 14 follower RL vehicles.
   - 48 distinct columns capturing every raw input, derived kinematic variable, and individual reward component value for quantitative graphing and analysis.

