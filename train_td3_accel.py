import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import numpy as np
import os
import sys

# Add current directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from env.accel_traffic_env import AccelTrafficEnv

def train():
    # Instantiate the decentralized VecEnv natively
    env = AccelTrafficEnv()
    # Normalize rewards (not obs, since we do that manually) to stabilize critic training
    # clip_reward=10.0 prevents extreme outliers from destabilizing the Q-function
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    env = VecMonitor(env)

    # Initialize action noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Initialize the TD3 agent
    # We use MlpPolicy since our observations are flattened arrays
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,       # Reduced from 0.001 for stability over 2M steps
        buffer_size=500000,          # Increased from 100k to retain more diverse experiences
        learning_starts=5000,        # More initial random exploration before learning
        batch_size=256,              # Larger batches for smoother gradient estimates
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,            # Changed from -1: one gradient update per env step (prevents divergence)
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[400, 300]),
        tensorboard_log="./tensorboard_logs/"
    )

    print("Starting TD3 direct acceleration training (Option 2 tuned rewards)...")
    # Train the agent
    model.learn(total_timesteps=1000000, log_interval=100)
    print("Training finished.")

    # Save the model
    model_path = os.path.join(current_dir, "td3_accel_agent")
    model.save(model_path)
    print(f"Model saved as '{model_path}.zip'")

    # Evaluate the trained agent
    print("\nEvaluating model...")
    obs = env.reset()
    total_reward = 0
    
    # Run a short evaluation episode
    for i in range(1000):
        # Predict the action (now expects to predict for 15 agents separately)
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(actions)
        total_reward += np.sum(rewards)
        
        if any(dones):
            print(f"Episode terminated early due to collision after {i+1} steps.")
            break
            
    print(f"Total Evaluation Reward: {total_reward}")

if __name__ == "__main__":
    train()
