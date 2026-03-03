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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from traffic_env import TrafficControlEnv


def make_env(curriculum_step=1):
    """Factory function to create environment with curriculum learning."""
    return TrafficControlEnv(curriculum_step=curriculum_step)

def train():
    print("=" * 60)
    print("TD3 Agent Training for Traffic Control")
    print("=" * 60)
    
    # Start with curriculum step 1 (5 cars)
    curriculum_step = 1
    total_timesteps = 100000
    eval_freq = 5000
    
    # Create initial environment with curriculum
    env = DummyVecEnv([lambda: make_env(curriculum_step=curriculum_step)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(curriculum_step=curriculum_step)])
    
    # Initialize TD3 agent with tuned hyperparameters
    model = TD3(
        "MlpPolicy", 
        env,
        learning_rate=0.001,      # Standard learning rate
        buffer_size=50000,        # Smaller buffer to focus on recent experience
        learning_starts=10000,    # Let agent explore more before learning
        batch_size=64,            # Smaller batch for stability
        tau=0.005,                # Target network update rate
        gamma=0.99,               # Discount factor
        train_freq=(1, "episode"),
        gradient_steps=-1,        # Train after each episode
        action_noise=None,
        policy_kwargs=dict(net_arch=[256, 256]),  # Smaller network
        verbose=1,
        tensorboard_log="./td3_traffic_logs/"
    )
    
    # Callbacks for saving and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="td3_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    
    print(f"\nTraining parameters:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Curriculum step: {curriculum_step} (num_cars={min(5 + curriculum_step, 15)})")
    print(f"  Learning rate: 0.0005")
    print(f"  Buffer size: 50,000")
    print(f"  Batch size: 64")
    print(f"  Policy network: [256, 256]")
    print(f"  Action bounds: [0.1, 1.5]")
    print("\nStarting training...\n")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=100
    )
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)
    
    # Save final model
    model.save("td3_traffic_agent_final")
    print("\nModel saved as 'td3_traffic_agent_final.zip'")
    
    # Evaluate on final trained model
    print("\nFinal Evaluation (10 episodes)...")
    obs = eval_env.reset()
    total_returns = []
    
    for ep in range(10):
        obs = eval_env.reset()
        episode_return = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = eval_env.step(action)
            episode_return += rewards[0]
            steps += 1
            done = dones[0]
        
        total_returns.append(episode_return)
        min_gap = info[0].get("min_gap", 0) if isinstance(info[0], dict) else 0
        print(f"  Episode {ep+1}: Return={episode_return:.1f}, Steps={steps}, Min Gap={min_gap:.2f}m")
    
    avg_return = np.mean(total_returns)
    print(f"\nAverage Return: {avg_return:.1f}")
    print(f"Success Rate: {sum(1 for r in total_returns if r > -5000) / len(total_returns) * 100:.1f}%")
    
    print("\nTraining complete! Check './models/best/' for the best model.")

if __name__ == "__main__":
    train()
