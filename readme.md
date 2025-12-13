# Traffic Simulation: ACC and BCC Python Code

This project simulates traffic flow using car-following models, including Adaptive Cruise Control (ACC), Bilateral Cruise Control (BCC) and an integrated model of ACC + BCC with dynamic switching. The simulation is visualized using a Tkinter-based GUI and includes advanced features like energy consumption tracking, collision handling, and custom velocity profiles.

## Project Structure
- `car.py`: Defines the `Car` class, representing individual vehicles and their dynamics, including energy calculation and collision physics.
- `road.py`: Defines the `Road` class, representing the road on which cars travel.
- `city.py`: Contains the `City` class, which manages the simulation, including cars, roads, and the main logic for updating vehicle states. Implements the integration factor algorithm for ACC+BCC model.
- `control_window.py`: The main GUI controller. Handles user input, simulation parameters, and starts/stops the simulation. Displays real-time energy consumption.
- `transportation_painter.py`: Handles visualization of the simulation using Tkinter.
- `run_headless.py`: Runs the simulation without GUI for batch processing or testing.
- `data.csv`, `data2.csv`: Optional CSV files for custom velocity profiles for lead and follower vehicles.

## How It Works
- The simulation creates a number of cars on a circular road.
- Each car's acceleration is determined by the selected car-following model (ACC, BCC, or ACC+BCC).
- The simulation updates car positions, velocities, and handles collisions at each time step.
- The GUI allows you to set parameters such as the number of cars, control gains (`kd`, `kv`, `kc`), desired velocity, minimum distance (`min_dis`), minimum gap for collision (`min_gap`), and more.
- The simulation is visualized in real time, showing car positions, velocities and the current model under which it is running.
- Three separate simulations run simultaneously to compare ACC, BCC, and ACC+BCC models side-by-side.
- Real-time energy consumption is calculated and displayed for each model.

## How to Run
1. **Requirements**: Python 3.x. No external libraries are required beyond Tkinter (included with standard Python).
2. **Start the Simulation**:
   - Run `control_window.py`:
     ```sh
     python control_window.py
     ```
   - This opens a window where you can set simulation parameters and control the simulation.
3. **Controls**:
   - **Run**: Starts the simulation with the current parameters.
   - **Stop Lead**: Stops the lead (ego) car in all three models.
   - **Resume Lead**: Resumes the lead car's movement.
   - **Stop Following**: Stops the following car.
   - **Resume Following**: Resumes the following car's movement.
   - **Plot Vel and Acc Profiles**: Plots velocity and acceleration time profiles for all three models (ACC, BCC, ACC+BCC) with color-coded cars (red=lead, green=last, blue=others).
   - **Enable Ego Velocity Profile**: Checkbox to enable custom velocity profiles loaded from CSV files (`data.csv` for lead, `data2.csv` for follower).
   - You can adjust parameters such as number of cars, control gains, desired velocity, minimum distance, minimum gap, and time step before running the simulation.

## Key Parameters
- **kd**: Gap control gain (how strongly a car reacts to the distance to the car in front).
- **kv**: Relative velocity gain (how strongly a car reacts to the speed difference with the car in front).
- **kc**: Desired velocity gain (how strongly a car tries to reach the desired speed).
- **v_des**: Desired velocity for all cars.
- **min_dis**: Desired following distance (buffer distance between cars).
- **min_gap**: Minimum allowed gap for collision detection and handling.
- **reaction_time**: Time delay in driver response.
- **max_a / min_a**: Maximum and minimum allowed acceleration.
- **dt**: Simulation time step (in seconds).

## Key Features

### 1. Energy Consumption Tracking
- Real-time energy calculation for each vehicle based on physics-based model
- Considers three main forces:
  - **Inertial Force**: `F_inertia = mass × acceleration`
  - **Rolling Resistance**: `F_roll = Cr × mass × gravity`
  - **Aerodynamic Drag**: `F_drag = 0.5 × Cd × air_density × frontal_area × velocity²`
- Total energy consumption displayed in kWh for each model
- Allows comparison of energy efficiency between ACC, BCC, and ACC+BCC models
- Vehicle parameters:
  - Mass: 1800 kg
  - Frontal Area: 2.2 m²
  - Drag Coefficient (Cd): 0.29
  - Rolling Resistance Coefficient (Cr): 0.015

### 2. Advanced ACC+BCC Integration Factor
The ACC+BCC model uses a sophisticated integration factor (iF) to dynamically blend ACC and BCC behaviors based on multiple weighted factors:

- **Back Gap Ratio** (weight: 6): Activates BCC when rear vehicle is within critical gap `X`
- **Front Gap Ratio** (weight: 6): Considers distance to front vehicle relative to headway threshold
- **Rear Braking Factor** (weight: 2): Increases BCC influence when rear vehicle is braking hard (< -2 m/s²)
- **Closing-in Ratio (Rear)** (weight: 2): Activates when rear vehicle is approaching with positive relative velocity
- **Closing-in Ratio (Front)** (weight: 3): Reduces BCC when closing in on front vehicle

**Integration Factor Calculation:**
- Critical gap `X = Gfront_min + Le + Grear_min` where:
  - `Gfront_min = ve × Tr + (vl - ve)² / (2 × ae) + Lb`
  - `Grear_min = vf × Tr + (vf - ve)² / (2 × af) + Lb`
- Normalized iF is smoothed with hysteresis (α = 0.009) to prevent rapid mode switching
- Mode classification: ACC (iF < 0.1), INTEGRATED (0.1 ≤ iF ≤ 0.8), BCC (iF > 0.8)

### 3. Collision Handling System
- **Collision Detection**: Detects when vehicles overlap (gap < vehicle length)
- **Collision Physics**: Applies coefficient of restitution (CoR = 0.2) for realistic velocity exchange
- **Visual Feedback**: Colliding vehicles turn **orange** (not yellow) for 4 seconds (40 time steps)
- **Position Correction**: Automatically adjusts vehicle positions to prevent overlap with `min_gap` buffer
- **Color Restoration**: Vehicles return to original colors after collision timer expires

### 4. Custom Velocity Profiles
- **CSV-based Profiles**: Load custom velocity-time profiles from CSV files
- **Time Interpolation**: Smoothly interpolates between velocity data points
- **Dual Profiles**: Separate profiles for lead vehicle (`data.csv`) and follower vehicle (`data2.csv`)
- **GUI Toggle**: Enable/disable velocity profiles via checkbox
- **CSV Format**: Requires `time` and `velocity` columns

### 5. Jerk Limiting
- **Maximum Jerk**: Limited to ±5 m/s³ for realistic and comfortable acceleration changes
- **Smoothing**: Prevents unrealistic sudden changes in acceleration
- **Applied to All Models**: ACC, BCC, and ACC+BCC all implement jerk constraints

### 6. Vehicle State Tracking
Each vehicle maintains comprehensive history for analysis:
- **Position History**: Records position at each time step
- **Velocity History**: Tracks velocity changes over time
- **Acceleration History**: Logs acceleration profiles
- **Gap History**: Stores inter-vehicular distances
- **Switch Events**: Records when ACC+BCC model switches modes
- **Mode Tracking**: Current control mode (ACC, BCC, or INTEGRATED)

### 7. Multiple Model Comparison
- **Simultaneous Simulation**: Runs three independent simulations in parallel
- **Visual Comparison**: Side-by-side display of ACC, BCC, and ACC+BCC behaviors
- **Energy Comparison**: Real-time energy consumption for each model
- **Performance Analysis**: Compare velocity and acceleration profiles across models

## Notes
- The simulation uses a circular road, so cars wrap around when reaching the end.
- Visualization shows each car as a rectangle, with the lead car in red, the last car in green, and others in blue.
- **Collisions** are indicated by the car turning **orange** with a 4-second visual indicator.
- In the BCC model, the last car always follows ACC logic for stability.
- In the ACC+BCC model, the last car uses pure ACC while other vehicles dynamically switch based on the integration factor.
- The simulation automatically runs for 60 seconds before plotting velocity and acceleration profiles.


## Customization
- **Car-Following Logic**: Modify `city.py` to experiment with different traffic models or add custom car behaviors
- **Integration Factor**: Adjust weights and thresholds in `calculate_integration_factor()` method in `city.py`
- **Energy Model**: Customize vehicle parameters (mass, drag coefficient, etc.) in `car.py`
- **Collision Physics**: Modify coefficient of restitution (CoR) in `car.py` for different collision behaviors
- **Visualization**: Adjust the rendering in `transportation_painter.py` as needed
- **Road Network**: The road model can be extended for more complex networks if desired
- **Jerk Limits**: Adjust `max_jerk` parameter in `city.py` for different comfort levels
- **Velocity Profiles**: Create custom CSV files with different time-velocity patterns

## Technical Details

### Simulation Parameters
- **Time Step (dt)**: Default 0.1 seconds (adjustable)
- **Road Length**: 1000 meters (circular)
- **Vehicle Length**: 4 meters
- **Update Frequency**: 100 ms (10 Hz)

### Control Gains
- **kd**: Gap control gain (default: 0.9)
- **kv**: Relative velocity gain (default: 0.5)
- **kc**: Desired velocity gain (default: 0.4)

### Safety Parameters
- **Reaction Time**: Default 0.8 seconds
- **Headway Time**: Default 1 second (used in ACC+BCC for gap threshold)
- **Max Acceleration**: 3.0 m/s²
- **Min Acceleration**: -5.0 m/s²
- **Max Jerk**: ±5.0 m/s³

## Performance & Analysis
- The simulation tracks minimum and maximum gaps across all vehicles
- Velocity and acceleration profiles can be plotted after simulation
- Energy consumption provides quantitative comparison between models
- Vehicle history data enables detailed post-simulation analysis

---
For any questions or further customization, reach out to me.
ē