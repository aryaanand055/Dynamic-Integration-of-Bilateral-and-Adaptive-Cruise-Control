# Traffic Simulation: ACC, BCC, and RL Integration

This project simulates traffic flow using car-following models, including Adaptive Cruise Control (ACC), Bilateral Cruise Control (BCC), an integrated model of ACC + BCC with dynamic switching, and a Reinforcement Learning (RL) agent trained using TD3. The simulation is visualized using a Tkinter-based GUI and includes advanced features like energy consumption tracking, collision handling, and custom velocity profiles.

## Project Structure
The codebase is modularized into several logical directories:

- `simulation/`: Contains the core physics engine and simulation loops.
  - `car.py`: Defines the `Car` class and its dynamics (energy calculation, collision physics).
  - `road.py`: Defines the `Road` boundaries.
  - `city.py`: The base simulation manager for ACC, BCC, and ACC+BCC models.
  - `rl_city.py`: An extension of the city manager that supports RL-controlled vehicles.
- `gui/`: Contains the graphical user interfaces and rendering logic.
  - `control_window.py`: The base GUI controller for running and plotting the simulation.
  - `rl_control_window.py`: An extended GUI that includes the RL TD3 Agent alongside standard models.
  - `transportation_painter.py`: Handles canvas drawing and visualizations.
- `env/`: Contains the OpenAI Gym environment for training the RL model.
  - `accel_traffic_env.py`: The custom Gym environment used to train the TD3 agent.
- `data/`: Contains CSV files for custom velocity profiles.
- `train_td3_accel.py`: Script to train the TD3 RL agent from scratch.
- `run_headless.py`: Runs the simulation without a GUI for evaluation and benchmarking.

## How It Works
- The simulation creates a number of cars on a circular road.
- Each car's acceleration is determined by the selected car-following model (ACC, BCC, ACC+BCC, or RL).
- The GUI allows you to set parameters such as the number of cars, control gains (`kd`, `kv`, `kc`), desired velocity, minimum distance (`min_dis`), minimum gap for collision (`min_gap`), and more.
- Four separate simulations run simultaneously to compare ACC, BCC, ACC+BCC, and RL models side-by-side.

## How to Run
1. **Requirements**: Python 3.x. Requires `stable-baselines3`, `gymnasium`, `numpy`, and `matplotlib`.
2. **Start the Simulation**:
   - Run the RL-enabled control window as a module from the root directory:
     ```sh
     python -m gui.rl_control_window
     ```
3. **Controls**:
   - **Run**: Starts the simulation.
   - **Stop Lead / Resume Lead**: Pauses/resumes the lead car to test shockwave propagation.
   - **Show Graphs**: Generates matplotlib velocity and acceleration profiles for all models.

## Key Features

### 1. Reinforcement Learning (TD3) Integration
The project features a custom-trained Reinforcement Learning agent capable of following traffic smoothly and avoiding collisions.
- **Algorithm**: Twin Delayed Deep Deterministic Policy Gradient (TD3).
- **Observation Space**: A normalized vector containing Ego Velocity, Ego Acceleration, Front Gap, Front Relative Velocity, Back Gap, Back Relative Velocity, and Desired Velocity.
- **Action Space**: Direct acceleration prediction bounded between `min_a` and `max_a`.
- **Custom Reward Function Design**:
  - **Gap Tracking & Safety**: Heavily penalizes tailgating (`gap_error^2`). Includes a linear "pull-forward" penalty to actively encourage the agent to catch up if it falls more than 1.5x the desired gap behind the leader.
  - **String Stability**: Penalizes the agent for *amplifying* the acceleration/deceleration of the car in front, ensuring shockwaves die out rather than grow. Crucially, this penalty is ignored if the agent has fallen far behind (`gap_ratio > 1.5`), allowing it to accelerate aggressively to catch back up without being penalized for "amplifying" acceleration.
  - **Speed Matching**: Gaussian/Linear shaped rewards for matching the desired velocity.
  - **Comfort**: Penalizes excessive jerk and extreme acceleration.
- **Training Parameters**:
  - `learning_rate`: 0.0003
  - `batch_size`: 256
  - `gamma`: 0.99
  - `tau`: 0.005
  - `policy_delay`: 2
  - Action Noise: `NormalActionNoise(mean=0.0, sigma=0.1)`

### 2. Energy Consumption Tracking
- Real-time energy calculation based on inertial force, rolling resistance, and aerodynamic drag.
- Displays total energy consumption in kWh for direct comparison between all models.

### 3. Advanced ACC+BCC Integration Factor
The ACC+BCC model dynamically blends behaviors:
- **Back Gap Ratio**: Activates BCC when rear vehicle is within a critical gap.
- **Front Gap Ratio**: Considers distance to the front vehicle relative to headway threshold.
- Smoothed with hysteresis to prevent rapid mode switching (ACC, INTEGRATED, or BCC modes).

### 4. Collision Handling System
- Applies coefficient of restitution for realistic velocity exchange.
- Visual feedback turns colliding vehicles **orange** for a few seconds.

### 5. Custom Velocity Profiles
- Load custom velocity-time profiles from CSV files (`data/data.csv` for lead, `data/data2.csv` for follower).

### 6. Multiple Model Comparison
- Runs independent simulations in parallel to visualize differences between traditional control theory and reinforcement learning.

## Technical Details

### Simulation Parameters
- **Time Step (dt)**: Default 0.1 seconds
- **Road Length**: 1000 meters (circular)
- **Vehicle Length**: 4 meters

### Control Gains (For ACC/BCC)
- **kd**: Gap control gain (default: 0.9)
- **kv**: Relative velocity gain (default: 0.5)
- **kc**: Desired velocity gain (default: 0.4)

### Safety Parameters
- **Reaction Time**: Default 0.8 seconds
- **Max Acceleration**: 3.0 m/s²
- **Min Acceleration**: -5.0 m/s²
- **Max Jerk**: ±5.0 m/s³

---
For any questions or further customization, reach out to me.