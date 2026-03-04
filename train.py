# train.py - TD3 Training for Traffic Controller
# 
# For Google Colab:
# !pip install gymnasium stable-baselines3
# !python train.py

import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from traffic_env import TrafficControlEnv


def make_env(num_cars=6):
    """Factory function to create environment."""
    return TrafficControlEnv(num_cars=num_cars)


def sanity_check_env():
    """Test environment with default weights before training."""
    print("=" * 60)
    print("Environment Sanity Check")
    print("=" * 60)
    
    env = TrafficControlEnv(num_cars=6)
    obs, _ = env.reset()
    
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test with default ACC-like weights
    default_action = np.array([0.9, 0.0, 0.6, 0.0, 0.4], dtype=np.float32)
    
    total_reward = 0
    steps = 0
    min_gaps = []
    
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(default_action)
        total_reward += reward
        steps += 1
        min_gaps.append(info['min_gap'])
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {steps}")
            print(f"  Termination reason: {'Collision' if terminated else 'Max steps'}")
            break
    
    print(f"\nSanity check results:")
    print(f"  Steps completed: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Min gap range: [{min(min_gaps):.2f}, {max(min_gaps):.2f}]m")
    print(f"  Final avg speed: {info['avg_speed']:.2f} m/s")
    
    success = steps >= 50 and min(min_gaps) > 0.5
    print(f"\nSanity check: {'PASSED ✓' if success else 'FAILED ✗'}")
    
    return success


def train():
    print("=" * 60)
    print("TD3 Agent Training for Traffic Control")
    print("=" * 60)
    
    # Run sanity check first
    if not sanity_check_env():
        print("\n⚠ Warning: Sanity check failed. Training may not converge.")
        print("Check gap calculations and initial conditions.")
    
    print("\n" + "=" * 60)
    print("Starting TD3 Training")
    print("=" * 60)
    
    # Configuration
    num_cars = 6
    total_timesteps = 50000  # Start smaller for faster iteration
    
    # Create environments
    env = DummyVecEnv([lambda: make_env(num_cars=num_cars)])
    eval_env = DummyVecEnv([lambda: make_env(num_cars=num_cars)])
    
    # Action noise for exploration (CRITICAL for TD3!)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions)  # 20% noise relative to action range
    )
    
    # Initialize TD3 with PROPER hyperparameters
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,           # Standard TD3 learning rate
        buffer_size=50000,            # Replay buffer size
        learning_starts=1000,         # Start learning after 1000 steps (not 10000!)
        batch_size=256,               # Larger batches for stability
        tau=0.005,                    # Target network update rate
        gamma=0.99,                   # Discount factor
        train_freq=(1, "step"),       # Train EVERY step, not every episode!
        gradient_steps=1,             # One gradient step per env step
        action_noise=action_noise,    # CRITICAL: Add exploration noise
        policy_kwargs=dict(
            net_arch=[256, 256]       # Two hidden layers
        ),
        verbose=1,
        tensorboard_log="./td3_logs/"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="td3_traffic"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=2500,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Number of cars: {num_cars}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: 3e-4")
    print(f"  Buffer size: 50,000")
    print(f"  Batch size: 256")
    print(f"  Learning starts: 1,000 steps")
    print(f"  Train frequency: Every step")
    print(f"  Action noise: Normal(σ=0.2)")
    print(f"  Network: [256, 256]")
    print("\n")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=100
    )
    
    # Save final model
    model.save("td3_traffic_final")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("Model saved as 'td3_traffic_final.zip'")
    
    # Final evaluation
    print("\nFinal Evaluation (10 episodes)...")
    evaluate_model(model, eval_env)


def evaluate_model(model, env):
    """Evaluate trained model."""
    total_returns = []
    
    for ep in range(10):
        obs = env.reset()
        episode_return = 0
        steps = 0
        done = False
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_return += rewards[0]
            steps += 1
            done = dones[0]
        
        total_returns.append(episode_return)
        info = infos[0] if infos else {}
        min_gap = info.get("min_gap", 0)
        collision = info.get("collision", False)
        
        status = "COLLISION" if collision else ("TIMEOUT" if steps >= 500 else "OK")
        print(f"  Episode {ep+1}: Return={episode_return:.1f}, Steps={steps}, "
              f"Min Gap={min_gap:.2f}m, Status={status}")
    
    avg_return = np.mean(total_returns)
    success_rate = sum(1 for r in total_returns if r > -50) / len(total_returns) * 100
    
    print(f"\nResults:")
    print(f"  Average Return: {avg_return:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    return avg_return, success_rate


if __name__ == "__main__":
    train()
