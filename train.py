# train.py
# 
# To run in Google Colab:
# 1. Upload the following files to the Colab runtime (Files tab on the left):
#    - traffic_env.py
#    - city.py
#    - car.py
#    - road.py 
#    - any other dependencies (e.g., data files if used, though traffic_env seems self-contained)
#
# 2. Install the necessary packages by running this in a code cell:
#    !pip install gymnasium stable-baselines3 shimmy
#
# 3. Run this script:
#    !python train.py

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import os
import sys

# Add current directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from traffic_env import TrafficControlEnv


# --- Colab Instructions ---
# If running in Google Colab, uncomment the following lines to install dependencies:
# !pip install gymnasium stable-baselines3 shimmy numpy
#
# Also, ensure you have uploaded the following files to the Colab runtime:
# - traffic_env.py
# - car.py
# - city.py
# - road.py
# - train.py (this file)
# ---------------------------

def make_env():
    return TrafficControlEnv()

def train():
    # Create the environment
    # We wrap it in a DummyVecEnv as SB3 expects a vectorized environment
    env = DummyVecEnv([make_env])

    # Initialize the TD3 agent
    # Policy: MlpPolicy (standard feedforward neural network)
    # Learning rate: Default is 1e-3, can be tuned
    # Action noise: TD3 benefits from action noise for exploration
    # gamma: Discount factor (0.99 is standard)
    # tau: Target network update rate
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=None, 
        policy_kwargs=dict(net_arch=[400, 300]) # Example architecture
    )

    print("Starting training...")
    # Train the agent
    model.learn(total_timesteps=10000, log_interval=10)
    print("Training finished.")

    # Save the model
    model.save("td3_traffic_agent")
    print("Model saved as 'td3_traffic_agent.zip'")

    # Evaluate the trained agent
    print("\nEvaluating model...")
    # Reset the underlying environment
    obs = env.reset()
    total_reward = 0
    
    # Run evaluation
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards[0]
        
        if dones[0]:
            print(f"Episode finished after {i+1} steps.")
            obs = env.reset()
            break
            
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    train()
