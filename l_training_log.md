# Reinforcement Learning Training Log & Model History (`l_training_log.md`)

This document chronicles every major training phase, hyperparameter change, reward function evolution, bug fix, and performance metric for the Twin Delayed DDPG (TD3) autonomous vehicle agent from project inception through Phase 3.

---

## 📌 Phase 1: The Initial Baseline Model
**Models:** TD3_1 to TD3_11  
**Goal:** Environment setup and initial hyperparameter tuning (learning rate, batch size, actor-critic network architecture).

**Results & Emergent Behaviors:**
* **Collision Avoidance:** The agent learned to avoid collisions and track target velocity.
* **Side Effect ("Bang-Bang" Policy):** The agent developed a chaotic "bang-bang" driving policy, constantly toggling between maximum acceleration ($+4.0\text{ m/s}^2$) and maximum braking ($-5.0\text{ m/s}^2$). This resulted in an extremely erratic velocity-time ($v-t$) graph, severe passenger discomfort (extreme jerk), and high energy consumption.

---

## 📌 Phase 2: Comfort, Energy, and String Stability
*This era focused on smoothing the agent's actions and reducing energy consumption from the chaotic Phase 1 policy.*

### 🔹 Phase 2.1: EMA Filtering & Basic Penalties
**Models:** TD3_12 to TD3_15
* Increased `r_comfort` weight (penalizing acceleration and jerk) to `0.5`.
* Introduced an `r_energy` penalty (weight `0.2`) based on the power proxy $|a \cdot v|$.
* Implemented an Exponential Moving Average (EMA) action filter ($\alpha = 0.3$) to smooth execution commands.
* **Results:** Acceleration smoothed out, but energy consumption remained high at **4.37 kWh** due to high-frequency micro-throttle adjustments.

### 🔹 Phase 2.2: The "Reward Hacking" Discovery
**Models:** TD3_16 to TD3_17
* Attempted to force energy consumption down by increasing `r_energy` weight to `0.5`.
* **Results:** The agent discovered an RL "reward hack." It learned that the best way to avoid energy penalties was to *refuse to brake*. It glided dangerously close behind leading vehicles, plummeting the minimum safety gap margin down to **0.17 meters**. Energy dropped to 3.16 kWh.

### 🔹 Phase 2.3: Action Delta Tuning (The Breakthrough)
**Models:** TD3_18 to TD3_19
* Implemented Option 2 reward function tuning to penalize step-to-step action variations directly from the neural network.
* **`r_action_delta`**: Set penalty weight to **`-1.5`** on $|a_t - a_{t-1}|$.
* **Results:** Energy consumption dropped from **6.14 kWh** down to **1.25 kWh**, outperforming classical controllers ($1.73–1.77\text{ kWh}$) in overall cruising energy efficiency.

---

## 📌 Phase 3: Physics, Geometry, and Observation Fixes
*This era dealt with fixing critical bugs in the simulation's circular road geometry formulas and stabilizing the mathematical limits of the RL agent's observations and rewards.*

### 🔹 Phase 3.1: True Geometry Alignment
**Models:** TD3_20 to TD3_21
* **Diagnosis:** Earlier codebases calculated `gap_to` backwards (`(car.pos - other.pos)`), causing the policy to learn inverted spatial observations.
* **Fix:** Corrected all bumper-to-bumper calculations across the environment and GUI to perfectly align with the physics engine.
* **Results:** `TD3_21` survived 500/500 steps, but the new geometry revealed the reward function encouraged over-speeding (45.12 m/s) and established a local minimum where holding the gas and crashing was mathematically "cheaper" than hitting the brakes.

### 🔹 Phase 3.2: Reward Function Re-balancing
**Models:** TD3_22 to TD3_23
* **Fix:** Replaced the infinite mathematical jerk penalty with a bounded quadratic (`10.0 * delta_a^2`).
* **Fix:** Scaled the collision penalty dramatically to `-100.0`.
* **Fix:** Widened domain randomization ranges (`v_des` 10–35 m/s, `max_a`, etc.) to cover the GUI defaults.
* **Results:** `TD3_23` completely stopped crashing, but the `-100` collision penalty combined with quadratic safety penalties caused the Critic (Q-function) to mathematically diverge during training (loss: 717,000).

### 🔹 Phase 3.3: VecNormalize Reward Scaling
**Models:** TD3_24
* **Fix:** Wrapped the training environment in `VecNormalize(norm_reward=True, clip_reward=10.0)`. This standard SB3 technique scales all rewards to zero mean and unit variance, preventing extreme outliers from destroying the Critic.
* **Results:** Critic loss completely stabilized (`0.005`). Agent survived all steps with 0 collisions, tracked the target speed perfectly (19.35 m/s), and energy consumption dropped from 95.5 kWh to **4.67 kWh**. Perfect run.

### 🔹 Phase 3.4: Inverse Distance Transformation (BCC Faraway Car Fix)
**Models:** TD3_25 (Active)
* **Diagnosis:** The last car in the platoon was braking randomly. Because it was on a circular track, the car "behind" it was the lead car, hundreds of meters away. The RL agent received an un-normalized raw `back_gap` of 600m, resulting in an extreme `+19.0` standard deviation outlier input that saturated the neural network and forced it into a `-1.0` braking output.
* **Fix:** Applied an **Inverse Distance** transformation to the observation: `inverse_back_gap = 1.0 / max(back_gap, 1.0)`. Massive distances (600m) now smoothly decay to `0.0016`, neatly bounded within `[0, 1]` without manually capping/lying to the sensors.
* **Results:** `TD3_25` finished 1,000,000 timesteps with extreme stability (Critic Loss: `0.0015`). The inverse distance math completely solved the last-car braking issue.
  * **Collisions:** 0 (Survived full 500-step analytics rollout)
  * **Energy Consumption:** 8.46 kWh
  * **Mean Speed:** 7.13 m/s (Target: 15.0 m/s)
  * **Mean Gap:** 62.67 m

---

## 📈 TensorBoard History

Stable Baselines 3 automatically logs training metrics in `tensorboard_logs/`. 

### 1. Phase 1 (Baseline)
* **TD3_1 to TD3_11**: Base integration and setup.

### 2. Phase 2 (Comfort & Efficiency)
* **TD3_12 to TD3_15 (Phase 2.1)**: EMA smoothing, 4.37 kWh energy.
* **TD3_16 to TD3_17 (Phase 2.2)**: "Reward hacking" phase.
* **TD3_18 to TD3_19 (Phase 2.3)**: Action Delta breakthrough.

### 3. Phase 3 (Geometry & Physics Fixes)
* **TD3_20 to TD3_21 (Phase 3.1)**: Geometry fixes.
* **TD3_22 to TD3_23 (Phase 3.2)**: Reward re-balancing and Domain Randomization.
* **TD3_24 (Phase 3.3)**: VecNormalize reward scaling. Critic stabilized, perfect run.
* **TD3_25 (Phase 3.4 - Active)**: Inverse Distance transformation.

---

## 📁 Log File Locations & Storage Paths

1. **Main Project Workspace Training Log**:
   * File: [l_training_log.md](file:///c:/Users/aryaa/Desktop/Arya/Projects/ACC%20and%20BCC%20Python%20Code/Dynamic-Integration-of-Bilateral-and-Adaptive-Cruise-Control/l_training_log.md)
2. **TensorBoard Rollout Logs**:
   * Directory: `tensorboard_logs/` (active run: `./tensorboard_logs/TD3_25`)
   * Inspect interactively via: `tensorboard --logdir tensorboard_logs`
